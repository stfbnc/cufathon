#ifndef MFDFA_H
#define MFDFA_H

#include <math.h>
#include "../cuda/mfdfa_kernel.cuh"
#include "../cuda/utils_kernels.cuh"


class MFDFA
{
public:
    explicit MFDFA(float *h_y, int yLen);
    ~MFDFA();
    void computeFlucVec(int *winSizes, int nWins, float *qVals, int nq, float *hq, int threads, bool revSeg=false);
    void computeMultifractalSpectrum(int *winSizes, int nWins, float *qVals, int nq, float *a, float *fa, int threads, bool revSeg=false);
private:
    cudaError_t cudaErr;
    float *d_y;
    float *d_t;
    int len = 0;
};

#endif
