#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
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

// Choose x, y, or z for the next split
int chooseSplitAxis(const float *dataset, const vector<int> &indices) {
    float minimum[3] = {numeric_limits<float>::infinity(), numeric_limits<float>::infinity(), numeric_limits<float>::infinity()};
    float maximum[3] = {-numeric_limits<float>::infinity(), -numeric_limits<float>::infinity(), -numeric_limits<float>::infinity()};

    for (int point : indices) {
        for (int axis = 0; axis < 3; axis++) {
            float value = coordinate(dataset, point, axis);
            minimum[axis] = min(minimum[axis], value);
            maximum[axis] = max(maximum[axis], value);
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
    size_t left_count = point_count * (size_t)(left_nodes) / (size_t)(requested_nodes);

    left_count = max(left_count, (size_t)(left_nodes));
    left_count = min(left_count, point_count - (size_t)(right_nodes));

    int axis = chooseSplitAxis(dataset, indices);

    auto compare = [&](int first, int second) {
        float first_value = coordinate(dataset, first, axis);
        float second_value = coordinate(dataset, second, axis);
        if (first_value != second_value) return first_value < second_value;
        return first < second;
    };

    nth_element(indices.begin(), indices.begin() + left_count, indices.end(), compare);

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

    for (int point : node.points) {
        if (selected[point]) continue;

        if (betterPoint(point, node.far_point, temp)) {
            node.second_far_point = node.far_point;
            node.far_point = point;
        } else if (betterPoint(point, node.second_far_point, temp)) {
            node.second_far_point = point;
        }
    }
}

void findTopTwoNodes(const vector<Node> &nodes, const float *temp, int &far_node, int &second_node) {
    far_node = -1;
    second_node = -1;

    for (int i = 0; i < (int)(nodes.size()); ++i) {
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

// Update every point in one node
void refreshNode(Node &node, const float *dataset, const int *idxs, int selected_count, const bool *selected, float *temp) {
    for (int sample = node.sample_stamp; sample < selected_count; sample++) {
        int sampled_point = idxs[sample];

        for (int point : node.points) {
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

        vector<int> all_indices((size_t)(n));
        iota(all_indices.begin(), all_indices.end(), 0);

        vector<Node> nodes;
        nodes.reserve((size_t)(node_count));
        buildNodes(current_dataset, move(all_indices), node_count, nodes);

        bool *selected = new bool[n];
        memset(selected, 0, sizeof(bool) * n);

        current_idxs[0] = 0;
        selected[0] = true;

        for (int point = 0; point < n; point++) {
            current_temp[point] = squaredDistance(current_dataset, point, 0);
            ++distance_checks;
        }

        for (Node &node : nodes) updateNodeTopTwo(node, selected, current_temp);

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

            validateCandidate(node, candidate, current_dataset, current_idxs, selected_count, current_temp);

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
                if (node.failed_candidates >= 2) refreshNode(node, current_dataset, current_idxs, selected_count, selected, current_temp);
            }
        }

        delete[] selected;
    }
}

int main() {
    // Test settings
    int b = 1, n = 100000, m = 128, node_count = 32;

    float *dataset = new float[b * n * 3];
    float *temp = new float[b * n];
    int *idxs = new int[b * m];

    for (int i = 0; i < b * n * 3; i++) dataset[i] = static_cast<float>(i % 100) * 0.01f;

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

    for (int i = 0; i < m; i++) printf("FPS_INDEX[%d]=%d\n", i, idxs[i]);

    delete[] dataset;
    delete[] temp;
    delete[] idxs;

    return 0;
}
