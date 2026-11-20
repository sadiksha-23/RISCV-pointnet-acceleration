#include <algorithm>
#include <climits>
#include <cerrno>
#include <fstream>
#include <vector>

#include <stdio.h>      
#include <stdlib.h>     
#include <math.h>       
#include <float.h>      

using namespace std;

extern "C" {
    #include <gem5/m5ops.h>
}


struct PointRecord {

    // x, y, and z coordinate of the point
    float x;
    float y;
    float z;

    // Original position of this point in the KITTI file
    int original_index;

    // x, y, z index of the voxel this point belongs to
    int vx;
    int vy;
    int vz;
};


// Store information about voxel
struct Voxel {

    // First point belonging to this voxel
    // after points are sorted by voxel
    int start;

    // Number of points inside this voxel
    int count;

    // x, y, z coordinate of voxel center
    float cx;
    float cy;
    float cz;

    // Radius of sphere that completely covers the voxel
    float radius;

    // Largest current FPS distance among points in this voxel
    float max_dist;

    // Position in the sorted points array of the farthest point
    int best_point;

    // Original KITTI index of that farthest point
    int best_original_index;
};


static bool comparePoints(const PointRecord &a, const PointRecord &b) {
    // Compare voxel x
    if (a.vx != b.vx) {
        return a.vx < b.vx;
    }

    // Compare voxel y
    if (a.vy != b.vy) {
        return a.vy < b.vy;
    }

    // Compare voxel z
    if (a.vz != b.vz) {
        return a.vz < b.vz;
    }

    // Same voxel: keep smaller original index first
    return a.original_index < b.original_index;
}


static int buildVoxels(const float *dataset, int n, float voxel_size, PointRecord *points, Voxel *voxels, float min_x, float min_y, float min_z) {
    // Going through every point
    for (int i = 0; i < n; i++) {

        // Get x, y, z from dataset
        float x = dataset[i * 3 + 0];
        float y = dataset[i * 3 + 1];
        float z = dataset[i * 3 + 2];

        // Save coordinates in points array
        points[i].x = x;
        points[i].y = y;
        points[i].z = z;

        // Save original point index
        points[i].original_index = i;

        // Find which voxel this point belongs to
        points[i].vx = (int)floorf((x - min_x) / voxel_size);

        points[i].vy = (int)floorf((y - min_y) / voxel_size);

        points[i].vz = (int)floorf((z - min_z) / voxel_size);
    }


    // Sort points by voxel index
    // Points in the same voxel become next to each other
    std::sort(points, points + n, comparePoints);

    // Number of voxels we have created
    int voxel_count = 0;

    // First point of the current voxel
    int start = 0;

    // Keep making voxels until every point is processed
    while (start < n) {

        // Voxel index of the point at "start"
        int vx = points[start].vx;
        int vy = points[start].vy;
        int vz = points[start].vz;

        // Start checking from the next point
        int end = start + 1;

        // Find how many consecutive points are in this same voxel
        while (
            end < n &&
            points[end].vx == vx &&
            points[end].vy == vy &&
            points[end].vz == vz
        ) {
            end++;
        }

        // Get the current voxel
        Voxel *v = &voxels[voxel_count];

        // First point belonging to this voxel
        (*v).start = start;

        // Number of points belonging to this voxel
        (*v).count = end - start;

        // Find center x of voxel
        (*v).cx = min_x + ((float)vx + 0.5f) * voxel_size;

        // Find center y of voxel
        (*v).cy = min_y + ((float)vy + 0.5f) * voxel_size;

        // Find center z of voxel
        (*v).cz = min_z + ((float)vz + 0.5f) * voxel_size;

        // Sphere radius that covers the whole voxel
        (*v).radius = 0.5f * sqrtf(3.0f) * voxel_size;

        // FPS distance information is not known yet
        (*v).max_dist = 1e38f;

        // For now, use first point in voxel as best point
        (*v).best_point = start;

        // Save its original KITTI index
        (*v).best_original_index = points[start].original_index;

        // One voxel has now been created
        voxel_count++;

        // Move to the first point of the next voxel
        start = end;
    }

    // Return how many non-empty voxels were created
    return voxel_count;
}


// RadiusFPS algorithm
void radiusFarthestPointSampling(int n, int m, PointRecord *points, Voxel *voxels, int voxel_count, float *temp, int *idxs) {

    // Nothing to sample
    if (m <= 0) {
        return;
    }

    // Initialize every point's minimum distance to a very large number
    for (int i = 0; i < n; i++) {
        temp[i] = 1e38f;
    }

    // Standard FPS starts from original point 0
    // Points were reordered by voxel sorting, so we find where original point 0 is now
    int old = 0;

    for (int i = 0; i < n; i++) {
        if (points[i].original_index == 0) {
            old = i;
            break;
        }
    }

    // First sampled point is original KITTI point 0
    idxs[0] = 0;

    // Start FPS iterations
    for (int j = 1; j < m; j++) {

        // Coordinates of current selected point
        float x1 = points[old].x;
        float y1 = points[old].y;
        float z1 = points[old].z;

        // Go through every voxel
        for (int v = 0; v < voxel_count; v++) {

            // Current voxel
            Voxel *voxel = &voxels[v];

            // Difference between selected point and voxel center
            float cx = (*voxel).cx - x1;
            float cy = (*voxel).cy - y1;
            float cz = (*voxel).cz - z1;

            // Distance from selected point to voxel center
            float center_dist = sqrtf(cx * cx + cy * cy + cz * cz);

            // Closest possible distance from selected point to any point in this voxel
            float lower_bound = center_dist - (*voxel).radius;

            // Distance cannot be negative
            if (lower_bound < 0.0f) {
                lower_bound = 0.0f;
            }

            // Our FPS uses squared distances, so square the lower bound too
            float lower_bound_sq = lower_bound * lower_bound;

            // If this voxel cannot improve any point distance, skip the whole voxel
            if (lower_bound_sq >= (*voxel).max_dist) {
                continue;
            }

            // Largest updated minimum distance in this voxel
            float new_voxel_max = -1.0f;

            // Start by assuming the first point is the farthest
            int voxel_best_point = (*voxel).start;

            // Original KITTI index of current farthest point
            int voxel_best_original = points[(*voxel).start].original_index;

            // Position immediately after the final point in this voxel
            int end = (*voxel).start + (*voxel).count;

            // Go through every point inside this voxel
            for (int k = (*voxel).start; k < end; k++) {

                // x difference from selected point
                float dx = points[k].x - x1;

                // Squared x difference
                float dx2 = dx * dx;

                // Continue only if x alone does not prove this point cannot improve
                if (dx2 < temp[k]) {

                    // y difference from selected point
                    float dy = points[k].y - y1;

                    // Squared y difference
                    float dy2 = dy * dy;

                    // Continue only if y alone does not prove this point cannot improve
                    if (dy2 < temp[k]) {

                        // z difference from selected point
                        float dz = points[k].z - z1;

                        // Squared z difference
                        float dz2 = dz * dz;

                        // Continue only if z alone does not prove this point cannot improve
                        if (dz2 < temp[k]) {

                            // Full squared Euclidean distance
                            float d = dx2 + dy2 + dz2;

                            // Update point's minimum distance if this selected point is closer
                            if (d < temp[k]) {
                                temp[k] = d;
                            }
                        }
                    }
                }

                // Check if this point is now the farthest point in the voxel
                if (temp[k] > new_voxel_max) {
                    new_voxel_max = temp[k];
                    voxel_best_point = k;
                    voxel_best_original = points[k].original_index;
                }

                // If distances are equal, choose smaller original index
                else if (temp[k] == new_voxel_max) {
                    if (points[k].original_index < voxel_best_original) {
                        voxel_best_point = k;
                        voxel_best_original = points[k].original_index;
                    }
                }
            }

            // Save updated maximum distance for this voxel
            (*voxel).max_dist = new_voxel_max;

            // Save farthest point in this voxel
            (*voxel).best_point = voxel_best_point;

            // Save its original KITTI index
            (*voxel).best_original_index = voxel_best_original;
        }

        // Best distance among all voxels
        float best = -1.0f;

        // Sorted-array position of global farthest point
        int best_point = -1;

        // Original KITTI index of global farthest point
        int best_original = -1;

        // Compare the winner from every voxel
        for (int v = 0; v < voxel_count; v++) {

            // This voxel has a farther point than our current best
            if (voxels[v].max_dist > best) {
                best = voxels[v].max_dist;
                best_point = voxels[v].best_point;
                best_original = voxels[v].best_original_index;
            }

            // Equal distance: choose smaller original index
            else if (voxels[v].max_dist == best) {
                if (best_original < 0 || voxels[v].best_original_index < best_original) {
                    best_point = voxels[v].best_point;
                    best_original = voxels[v].best_original_index;
                }
            }
        }

        // This becomes the next selected FPS point
        old = best_point;

        // Store its original KITTI index
        idxs[j] = points[old].original_index;
    }
}

static bool loadKittiFrame(const char *filename, vector<float> &dataset) {
    ifstream file(filename, ios::binary | ios::ate);

    if (!file) {
        fprintf(stderr, "ERROR: Could not open KITTI frame: %s\n", filename);
        return false;
    }

    streamsize bytes = file.tellg();

    if (bytes <= 0 || bytes % (4 * sizeof(float)) != 0) {
        fprintf(stderr, "ERROR: Invalid KITTI frame size: %lld bytes\n", static_cast<long long>(bytes));
        return false;
    }

    file.seekg(0, ios::beg);

    size_t point_count = (size_t)(bytes) / (4 * sizeof(float));
    vector<float> raw(point_count * 4);

    if (!file.read(reinterpret_cast<char *>(raw.data()), bytes)) {
        fprintf(stderr, "ERROR: Failed while reading KITTI frame\n");
        return false;
    }

    dataset.resize(point_count * 3);

    for (size_t i = 0; i < point_count; i++) {
        float x = raw[i * 4 + 0];
        float y = raw[i * 4 + 1];
        float z = raw[i * 4 + 2];

        if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
            fprintf(stderr, "ERROR: Non-finite coordinate at point %zu\n", i);
            return false;
        }

        dataset[i * 3 + 0] = x;
        dataset[i * 3 + 1] = y;
        dataset[i * 3 + 2] = z;
    }

    return true;
}


int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        fprintf(stderr, "Usage: %s <KITTI_frame.bin> [point_count]\n", argv[0]);
        return 1;
    }

    // Same dataset loading as normal FPS
    vector<float> dataset;

    if (!loadKittiFrame(argv[1], dataset)) {
        return 1;
    }

    // Same values as normal FPS
    if (dataset.size() / 3 > (size_t)INT_MAX) {
        fprintf(stderr, "ERROR: Too many points for int indices\n");
        return 1;
    }

    int n = (int)(dataset.size() / 3);

    // Optional: use the first point_count points, e.g. 100000 for the slide baseline.
    // Without this argument, use the entire frame, as in the original file.
    if (argc == 3) {
        char *end = nullptr;
        errno = 0;
        long requested = strtol(argv[2], &end, 10);
        if (errno != 0 || end == argv[2] || *end != '\0' || requested <= 0 || requested > n) {
            fprintf(stderr, "ERROR: point_count must be between 1 and %d\n", n);
            return 1;
        }
        n = (int)requested;
    }
    int m = 128;

    if (n < m) {
        fprintf(stderr, "ERROR: Frame has only %d points but M=%d\n", n, m);
        return 1;
    }

    // Measure minimum-coordinate scan, required array allocations,
    // voxel construction (including sorting), and FPS sampling.
    // File loading and output printing are outside this region.
    m5_reset_stats(0, 0);

    // Find minimum x, y, z of the point cloud.
    // RadiusFPS needs these to create the voxel grid.
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

    // Initial voxel side length.
    // We can test different values later.
    float voxel_size = 2.0f;

    // RadiusFPS-specific arrays.
    // n is enough for voxels because worst case is one voxel per point.
    vector<PointRecord> points((size_t)(n));
    vector<Voxel> voxels((size_t)(n));

    // Same purpose as normal FPS:
    // temp stores each point's current minimum distance.
    vector<float> temp((size_t)(n));

    // Same output array as normal FPS.
    vector<int> idxs((size_t)(m));

    // Build the voxel structure inside the measured region.
    int voxel_count = buildVoxels(dataset.data(), n, voxel_size, points.data(), voxels.data(), min_x, min_y, min_z);

    printf("PREPARATION_COMPLETE\n");
    m5_dump_stats(0, 0);
    m5_reset_stats(0, 0);

    radiusFarthestPointSampling(n, m, points.data(), voxels.data(), voxel_count, temp.data(), idxs.data());

    printf("SAMPLING_COMPLETE\n");
    m5_dump_stats(0, 0);

    // Same output as normal FPS.
    printf("POINT_COUNT=%d\n", n);

    for (int i = 0; i < m; i++) {
        printf("FPS_INDEX[%d]=%d\n", i, idxs[i]);
    }

    return 0;
}