#ifndef DCCA_H
#define DCCA_H

#include <math.h>
#include "../cuda/dcca_kernel.cuh"
#include "../cuda/utils_kernels.cuh"


class DCCA
{
public:
    explicit DCCA(float *h_y1, float *h_y2, int yLen);
    ~DCCA();
    void computeFlucVec(int *winSizes, int nWins, float *rho, int threads, bool revSeg=false);
    void computeThresholds(int *winSizes, int nWins, int threads);
private:
    cudaError_t cudaErr;
    float *d_y1;
    float *d_y2;
    float *d_t;
    int len = 0;
};

#endif
