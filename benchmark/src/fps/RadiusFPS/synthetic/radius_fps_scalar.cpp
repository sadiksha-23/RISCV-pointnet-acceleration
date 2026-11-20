#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

extern "C" {
    #include <gem5/m5ops.h>
}


// Store one point and the voxel it belongs to
struct PointRecord {
    float x;
    float y;
    float z;

    int original_index;

    int vx;
    int vy;
    int vz;
};


// Store information about one voxel
struct Voxel {
    int start;
    int count;

    float cx;
    float cy;
    float cz;

    float radius;

    float max_dist;

    int best_point;
    int best_original_index;
};


// Used by qsort to put points from the same voxel together
static int comparePoints(const void *a, const void *b) {
    const PointRecord *pa = (const PointRecord *)a;
    const PointRecord *pb = (const PointRecord *)b;

    // Compare voxel x
    if ((*pa).vx != (*pb).vx) {
        if ((*pa).vx < (*pb).vx)
            return -1;
        else
            return 1;
    }

    // Compare voxel y
    if ((*pa).vy != (*pb).vy) {
        if ((*pa).vy < (*pb).vy)
            return -1;
        else
            return 1;
    }

    // Compare voxel z
    if ((*pa).vz != (*pb).vz) {
        if ((*pa).vz < (*pb).vz)
            return -1;
        else
            return 1;
    }

    // Same voxel: keep smaller original index first
    if ((*pa).original_index != (*pb).original_index) {
        if ((*pa).original_index < (*pb).original_index)
            return -1;
        else
            return 1;
    }

    return 0;
}


// Build the voxel structure
static int buildVoxels(const float *dataset, int n, float voxel_size, PointRecord *points, Voxel *voxels, float min_x, float min_y, float min_z) {

    // Go through every point
    for (int i = 0; i < n; i++) {

        // Read point coordinates
        float x = dataset[i * 3 + 0];
        float y = dataset[i * 3 + 1];
        float z = dataset[i * 3 + 2];

        // Store coordinates
        points[i].x = x;
        points[i].y = y;
        points[i].z = z;

        // Remember original point index
        points[i].original_index = i;

        // Find which voxel the point belongs to
        points[i].vx = (int)floorf((x - min_x) / voxel_size);
        points[i].vy = (int)floorf((y - min_y) / voxel_size);
        points[i].vz = (int)floorf((z - min_z) / voxel_size);
    }

    // Put points belonging to the same voxel next to each other
    qsort(points, n, sizeof(PointRecord), comparePoints);

    // Number of voxels created
    int voxel_count = 0;

    // First point of current voxel
    int start = 0;

    // Keep creating voxels until all points are processed
    while (start < n) {

        // Voxel index of the current group
        int vx = points[start].vx;
        int vy = points[start].vy;
        int vz = points[start].vz;

        // Start checking at the next point
        int end = start + 1;

        // Keep moving while points belong to this same voxel
        while (end < n &&
               points[end].vx == vx &&
               points[end].vy == vy &&
               points[end].vz == vz) {
            end++;
        }

        // Current voxel
        Voxel *v = &voxels[voxel_count];

        // First point inside this voxel
        (*v).start = start;

        // Number of points inside this voxel
        (*v).count = end - start;

        // Calculate voxel center
        (*v).cx = min_x + ((float)vx + 0.5f) * voxel_size;
        (*v).cy = min_y + ((float)vy + 0.5f) * voxel_size;
        (*v).cz = min_z + ((float)vz + 0.5f) * voxel_size;

        // Radius of sphere that covers the voxel
        (*v).radius = 0.5f * sqrtf(3.0f) * voxel_size;

        // Distance information is not known yet
        (*v).max_dist = 1e38f;

        // Temporarily use first point as farthest point
        (*v).best_point = start;
        (*v).best_original_index = points[start].original_index;

        // One voxel created
        voxel_count++;

        // Move to next voxel
        start = end;
    }

    return voxel_count;
}


// Scalar RadiusFPS
void radiusFarthestPointSampling(int n, int m, PointRecord *points, Voxel *voxels, int voxel_count, float *temp, int *idxs) {

    if (m <= 0) {
        return;
    }

    // Initialize all current minimum distances
    for (int i = 0; i < n; i++) {
        temp[i] = 1e38f;
    }

    // Standard FPS starts from original point 0
    int old = 0;

    // Points were reordered, so find original point 0
    for (int i = 0; i < n; i++) {
        if (points[i].original_index == 0) {
            old = i;
            break;
        }
    }

    // First sampled point
    idxs[0] = 0;

    // FPS iterations
    for (int j = 1; j < m; j++) {

        // Current selected point
        float x1 = points[old].x;
        float y1 = points[old].y;
        float z1 = points[old].z;

        // Check every voxel
        for (int v = 0; v < voxel_count; v++) {

            Voxel *voxel = &voxels[v];

            // Difference from selected point to voxel center
            float cx = (*voxel).cx - x1;
            float cy = (*voxel).cy - y1;
            float cz = (*voxel).cz - z1;

            // Distance to voxel center
            float center_dist = sqrtf(cx * cx + cy * cy + cz * cz);

            // Closest any point inside this voxel could possibly be
            float lower_bound = center_dist - (*voxel).radius;

            if (lower_bound < 0.0f) {
                lower_bound = 0.0f;
            }

            // temp stores squared distance
            float lower_bound_sq = lower_bound * lower_bound;

            // Skip whole voxel if nothing inside can improve
            if (lower_bound_sq >= (*voxel).max_dist) {
                continue;
            }

            // Find new farthest point inside this voxel
            float new_voxel_max = -1.0f;
            int voxel_best_point = (*voxel).start;
            int voxel_best_original = points[(*voxel).start].original_index;

            // End position of this voxel
            int end = (*voxel).start + (*voxel).count;

            // Check every point in this voxel
            for (int k = (*voxel).start; k < end; k++) {

                // x difference
                float dx = points[k].x - x1;
                float dx2 = dx * dx;

                // If x alone does not rule this point out
                if (dx2 < temp[k]) {

                    // y difference
                    float dy = points[k].y - y1;
                    float dy2 = dy * dy;

                    // If y alone does not rule this point out
                    if (dy2 < temp[k]) {

                        // z difference
                        float dz = points[k].z - z1;
                        float dz2 = dz * dz;

                        // If z alone does not rule this point out
                        if (dz2 < temp[k]) {

                            // Full squared distance
                            float d = dx2 + dy2 + dz2;

                            // Update minimum distance
                            if (d < temp[k]) {
                                temp[k] = d;
                            }
                        }
                    }
                }

                // Find farthest point inside this voxel
                if (temp[k] > new_voxel_max) {
                    new_voxel_max = temp[k];
                    voxel_best_point = k;
                    voxel_best_original = points[k].original_index;
                }

                // Same distance: use smaller original index
                else if (temp[k] == new_voxel_max) {
                    if (points[k].original_index < voxel_best_original) {
                        voxel_best_point = k;
                        voxel_best_original = points[k].original_index;
                    }
                }
            }

            // Store winner of this voxel
            (*voxel).max_dist = new_voxel_max;
            (*voxel).best_point = voxel_best_point;
            (*voxel).best_original_index = voxel_best_original;
        }

        // Find global farthest point using voxel winners
        float best = -1.0f;
        int best_point = -1;
        int best_original = -1;

        // Compare every voxel winner
        for (int v = 0; v < voxel_count; v++) {

            if (voxels[v].max_dist > best) {
                best = voxels[v].max_dist;
                best_point = voxels[v].best_point;
                best_original = voxels[v].best_original_index;
            }

            // Same distance: choose smaller original index
            else if (voxels[v].max_dist == best) {
                if (best_original < 0 || voxels[v].best_original_index < best_original) {
                    best_point = voxels[v].best_point;
                    best_original = voxels[v].best_original_index;
                }
            }
        }

        // This point becomes the next selected point
        old = best_point;

        // Store original index
        idxs[j] = points[old].original_index;
    }
}


int main() {
    int b = 1, n = 100000, m = 1024;

    // Same synthetic dataset allocation as original FPS
    float *dataset = new float[b * n * 3];
    float *temp = new float[b * n];
    int *idxs = new int[b * m];

    // Same synthetic data generation as original FPS
    for (int i = 0; i < b * n * 3; i++) {
        dataset[i] = (float)(i % 100) * 0.01f;
    }

    // Same initialization as original FPS
    memset(temp, 0, sizeof(float) * b * n);
    memset(idxs, 0, sizeof(int) * b * m);

    // Find minimum x, y, z for voxel grid
    float min_x = dataset[0];
    float min_y = dataset[1];
    float min_z = dataset[2];

    for (int i = 1; i < n; i++) {
        float x = dataset[i * 3 + 0];
        float y = dataset[i * 3 + 1];
        float z = dataset[i * 3 + 2];

        if (x < min_x) {
            min_x = x;
        }

        if (y < min_y) {
            min_y = y;
        }

        if (z < min_z) {
            min_z = z;
        }
    }

    // Initial voxel size
    float voxel_size = 0.1f;

    // RadiusFPS-specific arrays
    PointRecord *points = new PointRecord[n];
    Voxel *voxels = new Voxel[n];

    // Build voxels before measured region
    int voxel_count = buildVoxels(dataset, n, voxel_size, points, voxels, min_x, min_y, min_z);

    // Start gem5 measurement
    m5_reset_stats(0, 0);

    // Run RadiusFPS
    radiusFarthestPointSampling(n, m, points, voxels, voxel_count, temp, idxs);

    // End gem5 measurement
    m5_dump_stats(0, 0);

    // Same basic output as original FPS
    printf("First FPS index: %d\n", idxs[0]);
    printf("Last FPS index: %d\n", idxs[m - 1]);

    // Extra information useful for RadiusFPS
    printf("Voxel count: %d\n", voxel_count);
    printf("Voxel size: %f\n", voxel_size);

    delete[] dataset;
    delete[] temp;
    delete[] idxs;
    delete[] points;
    delete[] voxels;

    return 0;
}