#ifndef HT_H
#define HT_H

#include <math.h>
#include "../cuda/ht_kernel.cuh"
#include "../cuda/utils_kernels.cuh"


class HT
{
public:
    explicit HT(float *h_y, int yLen);
    ~HT();
    void computeFlucVec(int *scales, int nScales, float *ht, int threads);
private:
    cudaError_t cudaErr;
    float *d_y = nullptr;
    float *d_t = nullptr;
    int len = 0;
};

#endif
