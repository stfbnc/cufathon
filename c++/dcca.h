#ifndef DCCA_H
#define DCCA_H

#include <math.h>
#include "../cuda/dcca_kernel.cuh"
#include "../cuda/utils_kernels.cuh"


class DCCA
{
public:
    explicit DCCA(double *h_y1, double *h_y2, int yLen);
    ~DCCA();
    void computeFlucVec(int *winSizes, int nWins, double *rho, int threads, bool revSeg=false);
private:
    cudaError_t cudaErr;
    double *d_y1;
    double *d_y2;
    double *d_t;
    int len = 0;
};

#endif
