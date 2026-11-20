#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

extern "C" {
    #include <gem5/m5ops.h>
}

using namespace std;

// One KD-tree node
struct Node {
    vector<int> points;
    int far_point;
    int second_far_point;
    int sample_stamp;
    int failed_candidates;
};

unsigned long long distance_checks = 0;
unsigned long long candidate_checks = 0;
unsigned long long node_refreshes = 0;

// Squared distance between two points
float squaredDistance(const float *dataset, int first, int second) {
    float dx = dataset[first * 3 + 0] - dataset[second * 3 + 0];
    float dy = dataset[first * 3 + 1] - dataset[second * 3 + 1];
    float dz = dataset[first * 3 + 2] - dataset[second * 3 + 2];
    return dx * dx + dy * dy + dz * dz;
}

float coordinate(const float *dataset, int point, int axis) {
    return dataset[point * 3 + axis];
}

struct SplitPointComparator {
    const float *dataset;
    int axis;

    SplitPointComparator(const float *data, int split_axis) {
        dataset = data;
        axis = split_axis;
    }

    bool operator()(int first, int second) const {
        float first_value = coordinate(dataset, first, axis);
        float second_value = coordinate(dataset, second, axis);
        if (first_value != second_value) return first_value < second_value;
        return first < second;
    }
};

// Choose x, y, or z for the next split
int chooseSplitAxis(const float *dataset, const vector<int> &indices) {
    float minimum[3];
    float maximum[3];

    for (int axis = 0; axis < 3; axis++) {
        minimum[axis] = coordinate(dataset, indices[0], axis);
        maximum[axis] = coordinate(dataset, indices[0], axis);
    }

    for (int i = 0; i < (int)indices.size(); i++) {
        for (int axis = 0; axis < 3; axis++) {
            float value = coordinate(dataset, indices[i], axis);
            if (value < minimum[axis]) minimum[axis] = value;
            if (value > maximum[axis]) maximum[axis] = value;
        }
    }

    int best_axis = 0;
    float best_range = maximum[0] - minimum[0];

    for (int axis = 1; axis < 3; axis++) {
        float range = maximum[axis] - minimum[axis];
        if (range > best_range) {
            best_range = range;
            best_axis = axis;
        }
    }

    return best_axis;
}

// Split the points into KD-tree leaf nodes
void buildNodes(const float *dataset, vector<int> indices, int requested_nodes, vector<Node> &nodes) {
    if (requested_nodes <= 1 || indices.size() <= 1) {
        Node node;
        node.points = move(indices);
        node.far_point = -1;
        node.second_far_point = -1;
        node.sample_stamp = 1;
        node.failed_candidates = 0;
        nodes.push_back(move(node));
        return;
    }

    int left_nodes = requested_nodes / 2;
    int right_nodes = requested_nodes - left_nodes;
    size_t point_count = indices.size();
    size_t left_count = point_count * (size_t)left_nodes / (size_t)requested_nodes;

    left_count = max(left_count, (size_t)left_nodes);
    left_count = min(left_count, point_count - (size_t)right_nodes);

    int axis = chooseSplitAxis(dataset, indices);

    nth_element(indices.begin(), indices.begin() + left_count, indices.end(), SplitPointComparator(dataset, axis));

    vector<int> left(indices.begin(), indices.begin() + left_count);
    vector<int> right(indices.begin() + left_count, indices.end());

    buildNodes(dataset, move(left), left_nodes, nodes);
    buildNodes(dataset, move(right), right_nodes, nodes);
}

bool betterPoint(int first, int second, const float *temp) {
    if (first < 0) return false;
    if (second < 0) return true;
    if (temp[first] != temp[second]) return temp[first] > temp[second];
    return first < second;
}

// Find the farthest two points in one node
void updateNodeTopTwo(Node &node, const bool *selected, const float *temp) {
    node.far_point = -1;
    node.second_far_point = -1;

    for (int i = 0; i < (int)node.points.size(); i++) {
        int point = node.points[i];
        if (selected[point]) continue;

        if (betterPoint(point, node.far_point, temp)) {
            node.second_far_point = node.far_point;
            node.far_point = point;
        } else if (betterPoint(point, node.second_far_point, temp)) {
            node.second_far_point = point;
        }
    }
}

// Find the nodes containing the best two current candidates
void findTopTwoNodes(const vector<Node> &nodes, const float *temp, int &far_node, int &second_node) {
    far_node = -1;
    second_node = -1;

    for (int i = 0; i < (int)nodes.size(); i++) {
        if (nodes[i].far_point < 0) continue;

        if (far_node < 0 || betterPoint(nodes[i].far_point, nodes[far_node].far_point, temp)) {
            second_node = far_node;
            far_node = i;
        } else if (second_node < 0 || betterPoint(nodes[i].far_point, nodes[second_node].far_point, temp)) {
            second_node = i;
        }
    }
}

// Recheck one candidate against new samples
void validateCandidate(Node &node, int candidate, const float *dataset, const int *idxs, int selected_count, float *temp) {
    for (int sample = node.sample_stamp; sample < selected_count; sample++) {
        float distance = squaredDistance(dataset, candidate, idxs[sample]);
        if (distance < temp[candidate]) temp[candidate] = distance;
        ++distance_checks;
    }

    ++candidate_checks;
}

// Fully update every point in one node
void refreshNode(Node &node, const float *dataset, const int *idxs, int selected_count, const bool *selected, float *temp) {
    for (int sample = node.sample_stamp; sample < selected_count; sample++) {
        int sampled_point = idxs[sample];

        for (int i = 0; i < (int)node.points.size(); i++) {
            int point = node.points[i];
            if (selected[point]) continue;

            float distance = squaredDistance(dataset, point, sampled_point);
            if (distance < temp[point]) temp[point] = distance;
            ++distance_checks;
        }
    }

    node.sample_stamp = selected_count;
    node.failed_candidates = 0;
    updateNodeTopTwo(node, selected, temp);
    ++node_refreshes;
}

// PointISA-style Lazy FPS
void lazyFarthestPointSampling(int b, int n, int m, int node_count, const float * __restrict__ dataset, float * __restrict__ temp, int * __restrict__ idxs) {
    if (m <= 0 || n <= 0) return;

    for (int batch = 0; batch < b; batch++) {
        const float *current_dataset = dataset + batch * n * 3;
        float *current_temp = temp + batch * n;
        int *current_idxs = idxs + batch * m;

        vector<int> all_indices((size_t)n);
        iota(all_indices.begin(), all_indices.end(), 0);

        vector<Node> nodes;
        nodes.reserve((size_t)node_count);
        buildNodes(current_dataset, move(all_indices), node_count, nodes);

        bool *selected = new bool[n];
        memset(selected, 0, sizeof(bool) * n);

        // FPS starts from point 0
        current_idxs[0] = 0;
        selected[0] = true;

        // Initial distance from every point to point 0
        for (int point = 0; point < n; point++) {
            current_temp[point] = squaredDistance(current_dataset, point, 0);
            ++distance_checks;
        }

        // Initialize the best candidates in every node
        for (Node &node : nodes) {
            updateNodeTopTwo(node, selected, current_temp);
        }

        int selected_count = 1;

        while (selected_count < m) {
            int far_node = -1;
            int second_node = -1;

            findTopTwoNodes(nodes, current_temp, far_node, second_node);

            if (far_node < 0) {
                printf("ERROR: No FPS candidate remains.\n");
                break;
            }

            Node &node = nodes[far_node];
            int candidate = node.far_point;

            // Lazily validate the current candidate
            validateCandidate(node, candidate, current_dataset, current_idxs, selected_count, current_temp);

            // Candidate distance may have changed
            updateNodeTopTwo(node, selected, current_temp);

            int checked_far_node = -1;
            int checked_second_node = -1;

            findTopTwoNodes(nodes, current_temp, checked_far_node, checked_second_node);

            bool safe = checked_far_node == far_node && node.far_point == candidate;

            if (safe) {
                current_idxs[selected_count] = candidate;
                ++selected_count;
                selected[candidate] = true;
                current_temp[candidate] = 0.0f;
                node.failed_candidates = 0;
                updateNodeTopTwo(node, selected, current_temp);
            } else {
                ++node.failed_candidates;

                if (node.failed_candidates >= 2) {
                    refreshNode(node, current_dataset, current_idxs, selected_count, selected, current_temp);
                }
            }
        }

        delete[] selected;
    }
}

// Load x, y, and z from one KITTI frame
bool loadKittiFrame(const char *filename, float *&dataset, int &point_count) {
    FILE *file = fopen(filename, "rb");

    if (file == nullptr) {
        printf("ERROR: Could not open KITTI frame: %s\n", filename);
        return false;
    }

    fseek(file, 0, SEEK_END);
    long bytes = ftell(file);
    rewind(file);

    if (bytes <= 0 || bytes % (4 * sizeof(float)) != 0) {
        printf("ERROR: Invalid KITTI frame size: %ld bytes\n", bytes);
        fclose(file);
        return false;
    }

    point_count = (int)(bytes / (4 * sizeof(float)));
    float *raw = new float[point_count * 4];

    size_t values_read = fread(raw, sizeof(float), point_count * 4, file);
    fclose(file);

    if (values_read != (size_t)point_count * 4) {
        printf("ERROR: Failed while reading KITTI frame.\n");
        delete[] raw;
        return false;
    }

    dataset = new float[point_count * 3];

    for (int i = 0; i < point_count; i++) {
        float x = raw[i * 4 + 0];
        float y = raw[i * 4 + 1];
        float z = raw[i * 4 + 2];

        if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
            printf("ERROR: Non-finite coordinate at point %d.\n", i);
            delete[] raw;
            delete[] dataset;
            dataset = nullptr;
            return false;
        }

        dataset[i * 3 + 0] = x;
        dataset[i * 3 + 1] = y;
        dataset[i * 3 + 2] = z;
    }

    delete[] raw;
    return true;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <KITTI_frame.bin>\n", argv[0]);
        return 1;
    }

    // Test settings
    int b = 1;
    int n = 0;
    int m = 128;
    int node_count = 32;
    float *dataset = nullptr;

    if (!loadKittiFrame(argv[1], dataset, n)) return 1;

    if (n < m) {
        printf("ERROR: Frame has only %d points but M=%d.\n", n, m);
        delete[] dataset;
        return 1;
    }

    float *temp = new float[b * n];
    int *idxs = new int[b * m];

    memset(temp, 0, sizeof(float) * b * n);
    memset(idxs, 0, sizeof(int) * b * m);

    m5_reset_stats(0, 0);

    lazyFarthestPointSampling(b, n, m, node_count, dataset, temp, idxs);

    m5_dump_stats(0, 0);

    printf("POINT_COUNT=%d\n", n);
    printf("SAMPLE_COUNT=%d\n", m);
    printf("NODE_COUNT=%d\n", node_count);
    printf("DISTANCE_CHECKS=%llu\n", distance_checks);
    printf("CANDIDATE_CHECKS=%llu\n", candidate_checks);
    printf("NODE_REFRESHES=%llu\n", node_refreshes);

    for (int i = 0; i < m; i++) {
        printf("FPS_INDEX[%d]=%d\n", i, idxs[i]);
    }

    delete[] dataset;
    delete[] temp;
    delete[] idxs;

    return 0;
}