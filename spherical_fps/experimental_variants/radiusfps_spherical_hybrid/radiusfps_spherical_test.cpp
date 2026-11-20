#include <fstream>
#include <chrono>
#include <algorithm>
#include <limits>
#include <string>
#include <cstring>
#include <stdint.h>
#include <vector>

#include <stdio.h>      
#include <stdlib.h>     
#include <math.h>       
#include <float.h>      

using namespace std;




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


static int comparePoints(const void *a, const void *b)
{
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
    qsort(points, n, sizeof(PointRecord), comparePoints);

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



// Spherical bounding intervals per original XYZ voxel; exact selected indices are
// unchanged because this provides only an additional *lower* distance bound.
struct SphericalBounds {
    double rlo, rhi, alo, ahi, elo, ehi;
};
static vector<SphericalBounds> g_spherical;
static double g_origin[3] = {0.,0.,0.};
static bool g_use_spherical = false;
static bool g_verify = false;
static uint64_t g_xyz_skips = 0, g_extra_spherical_skips = 0;
static uint64_t g_updated_voxels = 0, g_examined_points = 0, g_full_distances = 0;
static uint64_t g_spherical_evals = 0;
static constexpr double TWO_PI = 6.283185307179586476925286766559;
static constexpr double PI = 3.1415926535897932384626433832795;

static vector<SphericalBounds> buildSphericalBounds(PointRecord *points, Voxel *voxels,
                                                      int voxel_count, const double *origin) {
    vector<SphericalBounds> out((size_t) voxel_count);
    for (int v = 0; v < voxel_count; ++v) {
        auto &b = out[(size_t)v];
        b.rlo=b.alo=b.elo=std::numeric_limits<double>::infinity();
        b.rhi=b.ahi=b.ehi=-std::numeric_limits<double>::infinity();
        int start = voxels[v].start;
        int stop = start + voxels[v].count;
        for (int k=start; k<stop; ++k) {
            double dx=(double)points[k].x-origin[0];
            double dy=(double)points[k].y-origin[1];
            double dz=(double)points[k].z-origin[2];
            double rr=std::sqrt(dx*dx+dy*dy+dz*dz);
            double aa=std::atan2(dy,dx);
            double ee=std::atan2(dz,std::hypot(dx,dy));
            b.rlo=std::min(b.rlo,rr); b.rhi=std::max(b.rhi,rr);
            b.alo=std::min(b.alo,aa); b.ahi=std::max(b.ahi,aa);
            b.elo=std::min(b.elo,ee); b.ehi=std::max(b.ehi,ee);
        }
        // Round out intervals conservatively.
        b.rlo=std::max(0.0,b.rlo-1e-9);
        b.rhi+=1e-9; b.alo-=1e-12; b.ahi+=1e-12;
        b.elo-=1e-12; b.ehi+=1e-12;
    }
    return out;
}

static inline double clampDouble(double x, double lo, double hi) {
    return std::max(lo, std::min(x, hi));
}

// Certified lower bound on min_{p in voxel} ||p-s||^2, based only on the
// spherical intervals spanning all *actual* points in the voxel.
static double sphericalMinimumDistanceSq(const SphericalBounds &b, double sx, double sy, double sz) {
    double vx=sx-g_origin[0],vy=sy-g_origin[1],vz=sz-g_origin[2];
    double h=std::hypot(vx,vy);
    double psi=std::atan2(vy,vx);
    double c=std::max(std::cos(b.alo-psi),std::cos(b.ahi-psi));
    // Azimuth containing psi (mod 2*pi) reaches cos=1.
    double n=std::ceil((b.alo-psi)/TWO_PI);
    if (psi+TWO_PI*n <= b.ahi+1e-13) c=1.0;
    double A=h*c, B=vz;
    // Maximize A*cos(el)+B*sin(el) on [elo,ehi].
    double dotmax=std::max(A*std::cos(b.elo)+B*std::sin(b.elo),
                           A*std::cos(b.ehi)+B*std::sin(b.ehi));
    double angle=std::atan2(B,A);
    for (int k=-1;k<=1;k++) {
        double t=angle+TWO_PI*k;
        if (t>=b.elo-1e-13 && t<=b.ehi+1e-13)
            dotmax=std::max(dotmax,std::hypot(A,B));
    }
    // With fixed dotmax, minimum of r*r + |v|^2 - 2*r*dotmax
    // over radial interval occurs at r=clamp(dotmax,[rlo,rhi]).
    double r=clampDouble(dotmax,b.rlo,b.rhi);
    double d2=r*r+vx*vx+vy*vy+vz*vz-2*r*dotmax;
    // Conservative numerical slack. Never prune an equality.
    return std::max(0.0,d2-1e-5*(1+std::fabs(d2)));
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
                ++g_xyz_skips;
                continue;
            }
            // Only evaluate trigonometric geometry for voxels that the original
            // RadiusFPS lower bound could not eliminate.
            if (g_use_spherical) {
                ++g_spherical_evals;
                double spherical_lb_sq = sphericalMinimumDistanceSq(g_spherical[(size_t)v], x1, y1, z1);
                if (g_verify) {
                    double exact_min = std::numeric_limits<double>::infinity();
                    for (int k=(*voxel).start; k<(*voxel).start+(*voxel).count; ++k) {
                        double dx=(double)points[k].x-x1;
                        double dy=(double)points[k].y-y1;
                        double dz=(double)points[k].z-z1;
                        exact_min=std::min(exact_min,dx*dx+dy*dy+dz*dz);
                    }
                    if (spherical_lb_sq > exact_min + 1e-5*(1+exact_min)) {
                        fprintf(stderr,"UNSAFE SPHERICAL BOUND: voxel=%d lb=%f actualmin=%f\n",v,spherical_lb_sq,exact_min);
                        exit(3);
                    }
                }
                if (spherical_lb_sq >= (double)(*voxel).max_dist) {
                    ++g_extra_spherical_skips;
                    continue;
                }
            }
            ++g_updated_voxels;

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
                ++g_examined_points;

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
                            ++g_full_distances;
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


struct Result {
    double prep_ms=0, sample_ms=0, total_ms=0;
    uint64_t xyz_skips=0, spherical_skips=0, updated_voxels=0, visited_points=0, full_distances=0, spherical_evals=0;
    vector<int> indices;
};
static Result runOnce(const vector<float> &dataset, int n, int m, float voxel_size, bool spherical,
                      const double *origin, bool verify) {
    using Clock=std::chrono::steady_clock;
    auto st=Clock::now();
    float min_x=dataset[0],min_y=dataset[1],min_z=dataset[2];
    for (int i=1;i<n;i++) {
        min_x=std::min(min_x,dataset[(size_t)i*3]);
        min_y=std::min(min_y,dataset[(size_t)i*3+1]);
        min_z=std::min(min_z,dataset[(size_t)i*3+2]);
    }
    vector<PointRecord> points((size_t)n);
    vector<Voxel> voxels((size_t)n);
    vector<float> temp((size_t)n);
    vector<int> indices((size_t)m);
    int voxel_count=buildVoxels(dataset.data(),n,voxel_size,points.data(),voxels.data(),min_x,min_y,min_z);
    g_use_spherical=spherical;
    g_verify=verify;
    for(int i=0;i<3;i++) g_origin[i]=origin[i];
    g_spherical.clear();
    if (spherical) g_spherical=buildSphericalBounds(points.data(),voxels.data(),voxel_count,origin);
    auto prepared=Clock::now();
    g_xyz_skips=g_extra_spherical_skips=g_updated_voxels=g_examined_points=g_full_distances=g_spherical_evals=0;
    radiusFarthestPointSampling(n,m,points.data(),voxels.data(),voxel_count,temp.data(),indices.data());
    auto end=Clock::now();
    Result r;
    r.prep_ms=std::chrono::duration<double,std::milli>(prepared-st).count();
    r.sample_ms=std::chrono::duration<double,std::milli>(end-prepared).count();
    r.total_ms=std::chrono::duration<double,std::milli>(end-st).count();
    r.xyz_skips=g_xyz_skips;
    r.spherical_skips=g_extra_spherical_skips;
    r.updated_voxels=g_updated_voxels;
    r.visited_points=g_examined_points;
    r.full_distances=g_full_distances;
    r.spherical_evals=g_spherical_evals;
    r.indices=std::move(indices);
    return r;
}

static void printResult(const char *name, const Result &r) {
    printf("%-11s PREP_MS=%.3f SAMPLE_MS=%.3f TOTAL_MS=%.3f",name,r.prep_ms,r.sample_ms,r.total_ms);
    printf(" XYZ_SKIPS=%llu SPHERICAL_EXTRA_SKIPS=%llu UPDATED_VOXELS=%llu",
        (unsigned long long)r.xyz_skips,(unsigned long long)r.spherical_skips,
        (unsigned long long)r.updated_voxels);
    printf(" VISITED_POINTS=%llu FULL_DISTANCES=%llu SPHERICAL_EVALS=%llu\n",
        (unsigned long long)r.visited_points,(unsigned long long)r.full_distances,
        (unsigned long long)r.spherical_evals);
}

int main(int argc,char **argv) {
    string filename="datasets/kitti/0000000000.bin";
    string origin_mode="centroid";
    int n=100000,m=128,repeats=3;
    bool verify=false;
    float voxel_size=2.f;
    for(int i=1;i<argc;i++) {
        auto get=[&](const char *flag)->string {
            if(i+1>=argc) {fprintf(stderr,"Missing value for %s\n",flag);exit(2);} return argv[++i];
        };
        if (!strcmp(argv[i],"--dataset")) filename=get("--dataset");
        else if (!strcmp(argv[i],"--n")) n=std::stoi(get("--n"));
        else if (!strcmp(argv[i],"--m")) m=std::stoi(get("--m"));
        else if (!strcmp(argv[i],"--repeats")) repeats=std::stoi(get("--repeats"));
        else if (!strcmp(argv[i],"--voxel-size")) voxel_size=std::stof(get("--voxel-size"));
        else if (!strcmp(argv[i],"--origin")) origin_mode=get("--origin");
        else if (!strcmp(argv[i],"--verify-bounds")) verify=true;
        else {fprintf(stderr,"Unknown argument: %s\n",argv[i]);return 2;}
    }
    if(n<=0 || m<=0 || m>n || repeats<=0 || voxel_size<=0) {fprintf(stderr,"Invalid n/m/repeats/voxel size\n");return 2;}
    vector<float> dataset;
    if(!loadKittiFrame(filename.c_str(),dataset)) return 1;
    if((size_t)n*3>dataset.size()) {fprintf(stderr,"Dataset has fewer than N=%d points\n",n);return 1;}
    double origin[3]={0,0,0};
    if(origin_mode=="first") {
        for(int d=0;d<3;d++)origin[d]=dataset[d];
    } else if(origin_mode=="centroid") {
        for(int i=0;i<n;i++)for(int d=0;d<3;d++)origin[d]+=(double)dataset[(size_t)i*3+d];
        for(int d=0;d<3;d++)origin[d]/=n;
    } else if(origin_mode!="sensor") {
        fprintf(stderr,"Invalid origin: %s\n",origin_mode.c_str());return 2;
    }
    printf("Dataset=%s N=%d M=%d voxel_size=%.3f origin=%s repeats=%d verify_bounds=%s\n",
        filename.c_str(),n,m,voxel_size,origin_mode.c_str(),repeats,verify?"yes":"no");
    vector<double> base_prep,base_sample,base_total,polar_prep,polar_sample,polar_total;
    for(int rep=0;rep<repeats;rep++) {
        Result baseline=runOnce(dataset,n,m,voxel_size,false,origin,false);
        Result polar=runOnce(dataset,n,m,voxel_size,true,origin,verify);
        for(int i=0;i<m;i++) if(baseline.indices[(size_t)i]!=polar.indices[(size_t)i]) {
            printf("FAIL: selection #%d: RadiusFPS=%d RadiusFPS+spherical=%d\n",i+1,baseline.indices[(size_t)i],polar.indices[(size_t)i]);
            return 3;
        }
        static const int first10[]={0,3052,20530,853,29561,30686,7897,14320,30792,6898};
        if(n==100000 && m>=10 && filename.find("0000000000.bin")!=string::npos) {
            for(int i=0;i<10;i++)if(baseline.indices[(size_t)i]!=first10[i]) {
                printf("WARNING: expected KITTI first10 differ at step %d: got %d reference %d\n",i+1,baseline.indices[(size_t)i],first10[i]);
            }
        }
        printf("REPEAT %d/%d\n",rep+1,repeats);
        printResult("RadiusFPS",baseline);
        printResult("Radius+SPH",polar);
        printf("EXACT_INDICES_MATCH=PASS (%d/%d)\n",m,m);
        base_prep.push_back(baseline.prep_ms);base_sample.push_back(baseline.sample_ms);base_total.push_back(baseline.total_ms);
        polar_prep.push_back(polar.prep_ms);polar_sample.push_back(polar.sample_ms);polar_total.push_back(polar.total_ms);
    }
    auto med=[](vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];};
    double bt=med(base_total),pt=med(polar_total);
    printf("MEDIAN baseline: prep=%.3fms sample=%.3fms total=%.3fms\n",med(base_prep),med(base_sample),bt);
    printf("MEDIAN spherical: prep=%.3fms sample=%.3fms total=%.3fms\n",med(polar_prep),med(polar_sample),pt);
    printf("END_TO_END_SPEEDUP=%.4fx (above 1 faster; below 1 slower)\n",bt/pt);
    printf("EXACT_INDICES_ALL_REPEATS=PASS\n");
    return 0;
}
