#ifndef HT_H
#define HT_H

#include <math.h>
#include "../cuda/ht_kernel.cuh"
#include "../cuda/utils_kernels.cuh"


class HT
{
public:
    explicit HT(double *h_y, int yLen);
    ~HT();
    void computeFlucVec(int *scales, int nScales, double *F, int threads, int threads_mfdfa);
    void computeFlucVec_2(int *scales, int nScales, double *F, int threads, int threads_mfdfa);
private:
    cudaError_t cudaErr;
    double *d_y = nullptr;
    double *d_t = nullptr;
    int len = 0;
};

#endif
