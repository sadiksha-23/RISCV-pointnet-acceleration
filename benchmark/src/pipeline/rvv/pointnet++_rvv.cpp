#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <riscv_vector.h>

extern "C" {
    #include <gem5/m5ops.h>
}

// -------------------------------------------------------------
// 1. DOWNSAMPLING & GROUPING KERNELS (RVV)
// -------------------------------------------------------------

static inline void find_vector_maximum(vfloat32m1_t values, size_t vl, float &maximum_value, int &maximum_lane) {
    // FPS distances are nonnegative, so -1 is a safe starting value.
    vfloat32m1_t initial = __riscv_vfmv_v_f_f32m1(-1.0f, 1);

    // Find the largest value across all active vector lanes.
    vfloat32m1_t reduced = __riscv_vfredmax_vs_f32m1_f32m1(values, initial, vl);
    maximum_value = __riscv_vfmv_f_s_f32m1_f32(reduced);

    // Mark all lanes containing the maximum value.
    vbool32_t maximum_mask = __riscv_vmfeq_vf_f32m1_b32(values, maximum_value, vl);

    // Select the first lane containing that maximum.
    maximum_lane = static_cast<int>(__riscv_vfirst_m_b32(maximum_mask, vl));
}

void farthestPointSampling_rvv(int b, int n, int m, const float * __restrict__ dataset, float * __restrict__ temp, int * __restrict__ idxs) {
    if (m <= 0) return;

    const ptrdiff_t stride = 3 * sizeof(float);

    for (int i = 0; i < b; ++i) {
        int old = 0;
        idxs[0] = old;

        // Initialize the minimum-distance buffer.
        int rem_init = n;
        int init_offset = 0;

        for (size_t vl; rem_init > 0; rem_init -= vl, init_offset += vl) {
            vl = __riscv_vsetvl_e32m1(rem_init);
            vfloat32m1_t v_inf = __riscv_vfmv_v_f_f32m1(1e38f, vl);
            __riscv_vse32_v_f32m1(temp + init_offset, v_inf, vl);
        }

        const float *cur_dataset = dataset + i * n * 3;

        // Select the remaining m - 1 centroid points.
        for (int j = 1; j < m; ++j) {
            int besti = 0;
            float best = -1.0f;

            // Coordinates of the previously selected centroid.
            float x1 = cur_dataset[old * 3 + 0];
            float y1 = cur_dataset[old * 3 + 1];
            float z1 = cur_dataset[old * 3 + 2];

            int rem_n = n;
            int k_base = 0;

            // Process candidate points in RVV chunks.
            for (size_t vl; rem_n > 0; rem_n -= vl, k_base += vl) {
                vl = __riscv_vsetvl_e32m1(rem_n);

                // Load candidate x, y and z coordinates.
                vfloat32m1_t vx2 = __riscv_vlse32_v_f32m1(cur_dataset + k_base * 3 + 0, stride, vl);
                vfloat32m1_t vy2 = __riscv_vlse32_v_f32m1(cur_dataset + k_base * 3 + 1, stride, vl);
                vfloat32m1_t vz2 = __riscv_vlse32_v_f32m1(cur_dataset + k_base * 3 + 2, stride, vl);

                // Calculate coordinate differences.
                vfloat32m1_t vdx = __riscv_vfsub_vf_f32m1(vx2, x1, vl);
                vfloat32m1_t vdy = __riscv_vfsub_vf_f32m1(vy2, y1, vl);
                vfloat32m1_t vdz = __riscv_vfsub_vf_f32m1(vz2, z1, vl);

                // Calculate squared Euclidean distances.
                vfloat32m1_t vd = __riscv_vfmul_vv_f32m1(vdx, vdx, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdy, vdy, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdz, vdz, vl);

                // Update each point's minimum distance to a selected centroid.
                vfloat32m1_t v_temp = __riscv_vle32_v_f32m1(temp + k_base, vl);
                v_temp = __riscv_vfmin_vv_f32m1(v_temp, vd, vl);
                __riscv_vse32_v_f32m1(temp + k_base, v_temp, vl);

                // Find the maximum updated distance and its lane using RVV.
                float chunk_best;
                int chunk_best_lane;
                find_vector_maximum(v_temp, vl, chunk_best, chunk_best_lane);

                // Compare this chunk's winner with the overall winner.
                if (chunk_best > best) {
                    best = chunk_best;
                    besti = k_base + chunk_best_lane;
                }
            }

            // The farthest point becomes the next centroid.
            old = besti;
            idxs[j] = old;
        }

        temp += n;
        idxs += m;
    }
}

void gatherPoint_rvv(int b, int n, int m, const float * __restrict__ inp, const int * __restrict__ idx, float * __restrict__ out) {
    for (int i = 0; i < b; ++i) {
        const float *cur_inp = inp + i * n * 3;
        const int   *cur_idx = idx + i * m;
        float       *cur_out = out + i * m * 3;

        for (int j = 0; j < m; ++j) {
            int a = cur_idx[j];
            size_t vl = __riscv_vsetvl_e32m1(3);
            vfloat32m1_t v_coords = __riscv_vle32_v_f32m1(cur_inp + a * 3, vl);
            __riscv_vse32_v_f32m1(cur_out + j * 3, v_coords, vl);
        }
    }
}

static inline void fill_indices_rvv(int *output, int count, int value) {
    int remaining = count;
    int offset = 0;

    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remaining);
        vuint32m1_t values = __riscv_vmv_v_x_u32m1(static_cast<uint32_t>(value), vl);
        __riscv_vse32_v_u32m1(reinterpret_cast<uint32_t *>(output + offset), values, vl);

        remaining -= static_cast<int>(vl);
        offset += static_cast<int>(vl);
    }
}

void query_ball_point_rvv(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    float radius2 = radius * radius;
    const ptrdiff_t stride = 3 * sizeof(float);

    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < m; ++j) {
            int cnt = 0;
            int *current_output = idx + (i * m + j) * nsample;

            float x2 = xyz2[(i * m + j) * 3 + 0];
            float y2 = xyz2[(i * m + j) * 3 + 1];
            float z2 = xyz2[(i * m + j) * 3 + 2];

            const float *curr_xyz1 = xyz1 + i * n * 3;
            int k_base = 0;
            int rem_n = n;

            // Process candidate points in RVV chunks.
            for (size_t vl; rem_n > 0; rem_n -= vl, k_base += vl, curr_xyz1 += vl * 3) {
                if (cnt == nsample) break;

                vl = __riscv_vsetvl_e32m1(rem_n);

                // Load candidate coordinates.
                vfloat32m1_t vx1 = __riscv_vlse32_v_f32m1(curr_xyz1 + 0, stride, vl);
                vfloat32m1_t vy1 = __riscv_vlse32_v_f32m1(curr_xyz1 + 1, stride, vl);
                vfloat32m1_t vz1 = __riscv_vlse32_v_f32m1(curr_xyz1 + 2, stride, vl);

                // Calculate coordinate differences.
                vfloat32m1_t vdx = __riscv_vfrsub_vf_f32m1(vx1, x2, vl);
                vfloat32m1_t vdy = __riscv_vfrsub_vf_f32m1(vy1, y2, vl);
                vfloat32m1_t vdz = __riscv_vfrsub_vf_f32m1(vz1, z2, vl);

                // Calculate squared distances.
                vfloat32m1_t vd = __riscv_vfmul_vv_f32m1(vdx, vdx, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdy, vdy, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdz, vdz, vl);

                // Mark lanes whose distance is smaller than radius².
                vbool32_t within_radius = __riscv_vmflt_vf_f32m1_b32(vd, radius2, vl);

                // Count matching lanes.
                size_t match_count = __riscv_vcpop_m_b32(within_radius, vl);
                if (match_count == 0) continue;

                // Generate complete candidate indices for this chunk.
                vuint32m1_t lane_numbers = __riscv_vid_v_u32m1(vl);
                vuint32m1_t candidate_indices = __riscv_vadd_vx_u32m1(lane_numbers, static_cast<uint32_t>(k_base), vl);

                // Move all matching indices to the beginning of the vector.
                vuint32m1_t matching_indices = __riscv_vcompress_vm_u32m1(candidate_indices, within_radius, vl);

                // Preserve PointNet++ behavior by filling empty positions with the first match.
                if (cnt == 0) {
                    uint32_t first_index = __riscv_vmv_x_s_u32m1_u32(matching_indices);
                    fill_indices_rvv(current_output, nsample, static_cast<int>(first_index));
                }

                // Store only as many matching indices as are still needed.
                int remaining_slots = nsample - cnt;
                int indices_to_store = std::min(remaining_slots, static_cast<int>(match_count));

                __riscv_vse32_v_u32m1(
                    reinterpret_cast<uint32_t *>(current_output + cnt),
                    matching_indices,
                    static_cast<size_t>(indices_to_store)
                );

                cnt += indices_to_store;
            }

            pts_cnt[i * m + j] = cnt;
        }
    }
}

void group_point_rvv(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < nsample; ++k) {
                int ii = idx[(i * m + j) * nsample + k];

                const float *src_feature = points + (i * n + ii) * c;
                float *dst_feature = out + ((i * m + j) * nsample + k) * c;

                int rem_c = c;
                int c_offset = 0;

                // Strip-mine over feature channels
                for (size_t vl; rem_c > 0; rem_c -= vl, c_offset += vl) {
                    vl = __riscv_vsetvl_e32m1(rem_c);
                    vfloat32m1_t v_feat = __riscv_vle32_v_f32m1(src_feature + c_offset, vl);
                    __riscv_vse32_v_f32m1(dst_feature + c_offset, v_feat, vl);
                }
            }
        }
    }
}

// -------------------------------------------------------------
// 2. FEATURE EXTRACTION & REDUCTION KERNELS (RVV)
// -------------------------------------------------------------

void conv2d_mlp_bn_relu_rvv(int b, int n, int k, int c_in, int c_out, 
                            const float *X, const float *W, const float *bias,
                            const float *scale, const float *shift, 
                            float *Y) {
    
    int total_points = b * n * k;

    for (int p = 0; p < total_points; ++p) {
        const float *x_pt = X + p * c_in;
        float *y_pt = Y + p * c_out;

        int rem_co = c_out;
        int co_offset = 0;

        // Strip-mine across output feature channels
        for (size_t vl; rem_co > 0; rem_co -= vl, co_offset += vl) {
            vl = __riscv_vsetvl_e32m1(rem_co);

            // 1. Initialize vector accumulator with bias values
            vfloat32m1_t v_acc = __riscv_vle32_v_f32m1(bias + co_offset, vl);

            // 2. Perform GEMM: sum += X[p, ci] * W[ci, co]
            for (int ci = 0; ci < c_in; ++ci) {
                float x_val = x_pt[ci];
                const float *w_row = W + ci * c_out + co_offset;
                
                vfloat32m1_t v_w = __riscv_vle32_v_f32m1(w_row, vl);
                v_acc = __riscv_vfmacc_vf_f32m1(v_acc, x_val, v_w, vl);
            }

            // 3. Apply Batch Normalization: bn_out = sum * scale + shift
            vfloat32m1_t v_scale = __riscv_vle32_v_f32m1(scale + co_offset, vl);
            vfloat32m1_t v_shift = __riscv_vle32_v_f32m1(shift + co_offset, vl);
            
            vfloat32m1_t v_bn = __riscv_vfmul_vv_f32m1(v_acc, v_scale, vl);
            v_bn = __riscv_vfadd_vv_f32m1(v_bn, v_shift, vl);

            // 4. Apply ReLU Activation: max(0.0f, bn_out)
            vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t v_out = __riscv_vfmax_vv_f32m1(v_bn, v_zero, vl);

            // 5. Store final features to output array
            __riscv_vse32_v_f32m1(y_pt + co_offset, v_out, vl);
        }
    }
}

void maxpool_rvv(int b, int n, int k, int c, const float *input_features, float *output_features) {
    for (int batch = 0; batch < b; ++batch) {
        for (int pt = 0; pt < n; ++pt) {
            
            int rem_c = c;
            int ch_offset = 0;

            // Process feature channels in vector chunks
            for (size_t vl; rem_c > 0; rem_c -= vl, ch_offset += vl) {
                vl = __riscv_vsetvl_e32m1(rem_c);

                vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(-1e10f, vl);

                for (int nb = 0; nb < k; ++nb) {
                    const float *src = input_features + batch * (n * k * c) + pt * (k * c) + nb * c + ch_offset;
                    vfloat32m1_t v_in = __riscv_vle32_v_f32m1(src, vl);
                    vmax = __riscv_vfmax_vv_f32m1(vmax, v_in, vl);
                }

                float *dst = output_features + batch * (n * c) + pt * c + ch_offset;
                __riscv_vse32_v_f32m1(dst, vmax, vl);
            }
        }
    }
}

// -------------------------------------------------------------
// 3. FEATURE PROPAGATION (UPSAMPLING) KERNELS (RVV)
// -------------------------------------------------------------

static inline vfloat32m1_t take_vector_minimum(vfloat32m1_t values, size_t vl, float &minimum_value, int &minimum_lane) {
    const float infinity = 1e30f;

    // Find the smallest value across the active vector lanes.
    vfloat32m1_t initial = __riscv_vfmv_v_f_f32m1(infinity, 1);
    vfloat32m1_t reduced = __riscv_vfredmin_vs_f32m1_f32m1(values, initial, vl);
    minimum_value = __riscv_vfmv_f_s_f32m1_f32(reduced);

    // Find the first lane containing the minimum value.
    vbool32_t minimum_mask = __riscv_vmfeq_vf_f32m1_b32(values, minimum_value, vl);
    minimum_lane = static_cast<int>(__riscv_vfirst_m_b32(minimum_mask, vl));

    // Select only that lane.
    vuint32m1_t lane_numbers = __riscv_vid_v_u32m1(vl);
    vbool32_t selected_lane = __riscv_vmseq_vx_u32m1_b32(lane_numbers, static_cast<uint32_t>(minimum_lane), vl);

    // Remove the selected lane so the next call finds the next minimum.
    return __riscv_vfmerge_vfm_f32m1(values, infinity, selected_lane, vl);
}

void threenn_rvv(int b, int n, int m, const float *xyz1, const float *xyz2, float *dist, int *idx) {
    const ptrdiff_t stride = 3 * sizeof(float);

    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < n; ++j) {
            // Load the current target point.
            float x1 = xyz1[j * 3 + 0];
            float y1 = xyz1[j * 3 + 1];
            float z1 = xyz1[j * 3 + 2];

            // Store the three smallest distances and their indices.
            float best1 = 1e30f;
            float best2 = 1e30f;
            float best3 = 1e30f;

            int besti1 = 0;
            int besti2 = 0;
            int besti3 = 0;

            const float *curr_xyz2 = xyz2;
            int k_base = 0;
            int rem_m = m;

            // Process candidate points in RVV chunks.
            for (size_t vl; rem_m > 0; rem_m -= vl, k_base += vl, curr_xyz2 += vl * 3) {
                vl = __riscv_vsetvl_e32m1(rem_m);

                // Load candidate x, y and z coordinates.
                vfloat32m1_t vx2 = __riscv_vlse32_v_f32m1(curr_xyz2 + 0, stride, vl);
                vfloat32m1_t vy2 = __riscv_vlse32_v_f32m1(curr_xyz2 + 1, stride, vl);
                vfloat32m1_t vz2 = __riscv_vlse32_v_f32m1(curr_xyz2 + 2, stride, vl);

                // Calculate coordinate differences.
                vfloat32m1_t vdx = __riscv_vfrsub_vf_f32m1(vx2, x1, vl);
                vfloat32m1_t vdy = __riscv_vfrsub_vf_f32m1(vy2, y1, vl);
                vfloat32m1_t vdz = __riscv_vfrsub_vf_f32m1(vz2, z1, vl);

                // Calculate squared distances: dx² + dy² + dz².
                vfloat32m1_t vd = __riscv_vfmul_vv_f32m1(vdy, vdy, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdx, vdx, vl);
                vd = __riscv_vfmacc_vv_f32m1(vd, vdz, vdz, vl);

                // Keep the distances in the vector and extract up to three minima.
                vfloat32m1_t remaining_distances = vd;
                int candidates_to_check = (vl < 3) ? static_cast<int>(vl) : 3;

                for (int candidate = 0; candidate < candidates_to_check; ++candidate) {
                    float d;
                    int minimum_lane;

                    remaining_distances = take_vector_minimum(remaining_distances, vl, d, minimum_lane);
                    int curr_k = k_base + minimum_lane;

                    // Original top-three update logic.
                    if (d < best1) {
                        best3 = best2;
                        besti3 = besti2;
                        best2 = best1;
                        besti2 = besti1;
                        best1 = d;
                        besti1 = curr_k;
                    } else if (d < best2) {
                        best3 = best2;
                        besti3 = besti2;
                        best2 = d;
                        besti2 = curr_k;
                    } else if (d < best3) {
                        best3 = d;
                        besti3 = curr_k;
                    }
                }
            }

            // Store the final three nearest neighbors.
            dist[j * 3 + 0] = best1;
            idx[j * 3 + 0] = besti1;

            dist[j * 3 + 1] = best2;
            idx[j * 3 + 1] = besti2;

            dist[j * 3 + 2] = best3;
            idx[j * 3 + 2] = besti3;
        }

        // Move pointers to the next batch.
        xyz1 += n * 3;
        xyz2 += m * 3;
        dist += n * 3;
        idx += n * 3;
    }
}

void get_weights_rvv(int b, int n, const float *dist, float *weight) {
    const ptrdiff_t stride = 3 * sizeof(float);

    for (int i = 0; i < b; ++i) {
        const float *curr_dist = dist + (i * n * 3);
        float *curr_weight = weight + (i * n * 3);
        int rem_n = n;

        for (size_t vl; rem_n > 0; rem_n -= vl, curr_dist += vl * 3, curr_weight += vl * 3) {
            vl = __riscv_vsetvl_e32m1(rem_n);

            vfloat32m1_t vd0 = __riscv_vlse32_v_f32m1(curr_dist + 0, stride, vl);
            vfloat32m1_t vd1 = __riscv_vlse32_v_f32m1(curr_dist + 1, stride, vl);
            vfloat32m1_t vd2 = __riscv_vlse32_v_f32m1(curr_dist + 2, stride, vl);

            // Clamp distances before taking reciprocals, matching the scalar kernel.
            vfloat32m1_t veps = __riscv_vfmv_v_f_f32m1(1e-10f, vl);
            vd0 = __riscv_vfmax_vv_f32m1(vd0, veps, vl);
            vd1 = __riscv_vfmax_vv_f32m1(vd1, veps, vl);
            vd2 = __riscv_vfmax_vv_f32m1(vd2, veps, vl);

            vfloat32m1_t vw0 = __riscv_vfrdiv_vf_f32m1(vd0, 1.0f, vl);
            vfloat32m1_t vw1 = __riscv_vfrdiv_vf_f32m1(vd1, 1.0f, vl);
            vfloat32m1_t vw2 = __riscv_vfrdiv_vf_f32m1(vd2, 1.0f, vl);

            vfloat32m1_t vsum = __riscv_vfadd_vv_f32m1(vw0, vw1, vl);
            vsum = __riscv_vfadd_vv_f32m1(vsum, vw2, vl);

            vw0 = __riscv_vfdiv_vv_f32m1(vw0, vsum, vl);
            vw1 = __riscv_vfdiv_vv_f32m1(vw1, vsum, vl);
            vw2 = __riscv_vfdiv_vv_f32m1(vw2, vsum, vl);

            __riscv_vsse32_v_f32m1(curr_weight + 0, stride, vw0, vl);
            __riscv_vsse32_v_f32m1(curr_weight + 1, stride, vw1, vl);
            __riscv_vsse32_v_f32m1(curr_weight + 2, stride, vw2, vl);
        }
    }
}

void interpolate_rvv(int b, int m, int c, int n, const float *points, const int *idx, const float *weight, float *out) {
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < n; ++j) {
            float w1 = weight[j * 3 + 0];
            float w2 = weight[j * 3 + 1];
            float w3 = weight[j * 3 + 2];

            int i1 = idx[j * 3 + 0];
            int i2 = idx[j * 3 + 1];
            int i3 = idx[j * 3 + 2];

            const float *p1 = points + i1 * c;
            const float *p2 = points + i2 * c;
            const float *p3 = points + i3 * c;
            float *out_pt   = out + j * c;

            int rem_c = c;
            int c_offset = 0;

            for (size_t vl; rem_c > 0; rem_c -= vl, c_offset += vl) {
                vl = __riscv_vsetvl_e32m1(rem_c);

                vfloat32m1_t vf1 = __riscv_vle32_v_f32m1(p1 + c_offset, vl);
                vfloat32m1_t vf2 = __riscv_vle32_v_f32m1(p2 + c_offset, vl);
                vfloat32m1_t vf3 = __riscv_vle32_v_f32m1(p3 + c_offset, vl);

                vfloat32m1_t v_out = __riscv_vfmul_vf_f32m1(vf1, w1, vl);
                v_out = __riscv_vfmacc_vf_f32m1(v_out, w2, vf2, vl);
                v_out = __riscv_vfmacc_vf_f32m1(v_out, w3, vf3, vl);

                __riscv_vse32_v_f32m1(out_pt + c_offset, v_out, vl);
            }
        } 

        points += m * c;
        idx    += n * 3;
        weight += n * 3;
        out    += n * c;
    }
}

// -------------------------------------------------------------
// MAIN DRIVER & PROFILING PIPELINE
// -------------------------------------------------------------

int main() {
    int b = 5;
    int n = 1024;
    int m = 128;
    int nsample = 8;
    int c_in    = 64;
    int c_out   = 128;
    int classes = 4;
    float radius = 0.2f;
    float epsilon = 1e-5f;

    // Buffer Allocations
    float *dataset      = new float[b * n * 3];
    float *points       = new float[b * n * c_in];
    float *temp         = new float[b * n];
    int   *fps_idx      = new int[b * m];
    float *new_xyz      = new float[b * m * 3];
    int   *ball_idx     = new int[b * m * nsample];
    int   *pts_cnt      = new int[b * m];
    float *grouped_out  = new float[b * m * nsample * c_in];

    // Stage 1 MLP Weights & Precomputed BN Parameters (64 -> 128)
    float *W1     = new float[c_in * c_out];
    float *bias1  = new float[c_out];
    float *mean1  = new float[c_out];
    float *var1   = new float[c_out];
    float *gamma1 = new float[c_out];
    float *beta1  = new float[c_out];
    float *scale1 = new float[c_out];
    float *shift1 = new float[c_out];
    float *mlp_out = new float[b * m * nsample * c_out];

    // Max Pooling Output
    float *pooled_summary = new float[b * m * c_out];

    // Feature Propagation Buffers
    float *nn_dist   = new float[b * n * 3];
    int   *nn_idx    = new int[b * n * 3];
    float *nn_weight = new float[b * n * 3];
    float *interpolated_out = new float[b * n * c_out];

    // Stage 2 Classification MLP Weights & Precomputed BN Parameters (128 -> 4)
    float *W2     = new float[c_out * classes];
    float *bias2  = new float[classes];
    float *mean2  = new float[classes];
    float *var2   = new float[classes];
    float *gamma2 = new float[classes];
    float *beta2  = new float[classes];
    float *scale2 = new float[classes];
    float *shift2 = new float[classes];
    float *final_logits = new float[b * n * classes];

    // Fast deterministic test data setup
    for (int i = 0; i < b * n * 3; ++i) dataset[i] = (float)(i % 100) * 0.01f;
    for (int i = 0; i < b * n * c_in; ++i) points[i] = (float)(i % 200) * 0.005f;

    for (int i = 0; i < c_in * c_out; ++i) W1[i] = (float)(i % 50) * 0.02f;
    for (int i = 0; i < c_out; ++i) {
        bias1[i]  = 0.01f;
        mean1[i]  = 0.05f;
        var1[i]   = 1.00f;
        gamma1[i] = 1.00f;
        beta1[i]  = 0.00f;
        
        float inv_std = 1.0f / std::sqrt(var1[i] + epsilon);
        scale1[i] = gamma1[i] * inv_std;
        shift1[i] = beta1[i] - (mean1[i] * scale1[i]);
    }

    for (int i = 0; i < c_out * classes; ++i) W2[i] = (float)(i % 30) * 0.03f;
    for (int i = 0; i < classes; ++i) {
        bias2[i]  = 0.01f;
        mean2[i]  = 0.05f;
        var2[i]   = 1.00f;
        gamma2[i] = 1.00f;
        beta2[i]  = 0.00f;

        float inv_std = 1.0f / std::sqrt(var2[i] + epsilon);
        scale2[i] = gamma2[i] * inv_std;
        shift2[i] = beta2[i] - (mean2[i] * scale2[i]);
    }

    memset(fps_idx, 0, sizeof(int) * b * m);
    memset(ball_idx, 0, sizeof(int) * b * m * nsample);
    memset(pts_cnt, 0, sizeof(int) * b * m);
    memset(nn_idx, 0, sizeof(int) * b * n * 3);

    // =========================================================
    // --- gem5 STATS RESET: START END-TO-END RVV PIPELINE ---
    // =========================================================
    m5_reset_stats(0, 0);

    // 1. Spatial Downsampling & Grouping (RVV)
    farthestPointSampling_rvv(b, n, m, dataset, temp, fps_idx);
    gatherPoint_rvv(b, n, m, dataset, fps_idx, new_xyz);
    query_ball_point_rvv(b, n, m, radius, nsample, dataset, new_xyz, ball_idx, pts_cnt);
    group_point_rvv(b, n, c_in, m, nsample, points, ball_idx, grouped_out);

    // 2. Feature Extraction & Reduction (RVV Set Abstraction MLP)
    conv2d_mlp_bn_relu_rvv(b, m, nsample, c_in, c_out, grouped_out, W1, bias1, scale1, shift1, mlp_out);
    maxpool_rvv(b, m, nsample, c_out, mlp_out, pooled_summary);

    // 3. Spatial Upsampling & Feature Propagation (RVV)
    threenn_rvv(b, n, m, dataset, new_xyz, nn_dist, nn_idx);
    get_weights_rvv(b, n, nn_dist, nn_weight);
    interpolate_rvv(b, m, c_out, n, pooled_summary, nn_idx, nn_weight, interpolated_out);

    // 4. Final Classification Head (RVV Feature Propagation MLP)
    conv2d_mlp_bn_relu_rvv(b, n, 1, c_out, classes, interpolated_out, W2, bias2, scale2, shift2, final_logits);

    // =========================================================
    // --- gem5 STATS DUMP: END END-TO-END RVV PIPELINE ---
    // =========================================================
    m5_dump_stats(0, 0);

    // Prevent Dead Code Elimination
    printf("PointNet++ RVV End-to-End Pipeline Completed.\n");
    printf("Sample Check: Point 0 Class 0 Logit: %f\n", final_logits[0]);

    // Memory Cleanup
    delete[] dataset;
    delete[] points;
    delete[] temp;
    delete[] fps_idx;
    delete[] new_xyz;
    delete[] ball_idx;
    delete[] pts_cnt;
    delete[] grouped_out;
    delete[] W1;
    delete[] bias1;
    delete[] mean1;
    delete[] var1;
    delete[] gamma1;
    delete[] beta1;
    delete[] scale1;
    delete[] shift1;
    delete[] mlp_out;
    delete[] pooled_summary;
    delete[] nn_dist;
    delete[] nn_idx;
    delete[] nn_weight;
    delete[] interpolated_out;
    delete[] W2;
    delete[] bias2;
    delete[] mean2;
    delete[] var2;
    delete[] gamma2;
    delete[] beta2;
    delete[] scale2;
    delete[] shift2;
    delete[] final_logits;

    return 0;
}