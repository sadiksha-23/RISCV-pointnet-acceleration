// Native-only independent reference: exact brute-force FPS (seed 0, lowest index ties).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s <KITTI.bin> <N> <M>\n", argv[0]);
        return 2;
    }
    const int n=std::atoi(argv[2]), m=std::atoi(argv[3]);
    if (n<=0 || m<=0 || m>n) return 2;
    std::ifstream file(argv[1],std::ios::binary | std::ios::ate);
    if (!file) return 2;
    const std::streamoff len=file.tellg();
    if (len<static_cast<std::streamoff>(n)*4*sizeof(float)) return 2;
    file.seekg(0);
    std::vector<float> xyz(static_cast<size_t>(n)*3);
    for (int i=0;i<n;++i) {
        float raw[4];
        if (!file.read(reinterpret_cast<char*>(raw),sizeof(raw))) return 2;
        for (int j=0;j<3;++j) {
            if (!std::isfinite(raw[j])) return 2;
            xyz[3*i+j]=raw[j];
        }
    }
    std::vector<float> nearest(static_cast<size_t>(n),1e38f);
    std::vector<int> selected(static_cast<size_t>(m));
    selected[0]=0;
    for (int step=1;step<m;++step) {
        int newest=selected[step-1];
        const size_t p=static_cast<size_t>(newest)*3;
        float best=-1;
        int best_idx=-1;
        for (int i=0;i<n;++i) {
            const size_t q=static_cast<size_t>(i)*3;
            const float dx=xyz[q]-xyz[p];
            const float dy=xyz[q+1]-xyz[p+1];
            const float dz=xyz[q+2]-xyz[p+2];
            const float d=dx*dx+dy*dy+dz*dz;
            if (d<nearest[i]) nearest[i]=d;
            if (nearest[i]>best || (nearest[i]==best && (best_idx<0||i<best_idx))) {
                best=nearest[i];best_idx=i;
            }
        }
        selected[step]=best_idx;
    }
    for (int i=0;i<m;++i) std::printf("FPS_INDEX[%d]=%d\n",i,selected[i]);
    return 0;
}
