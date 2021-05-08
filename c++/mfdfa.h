#ifndef MFDFA_H
#define MFDFA_H

#include <math.h>
#include "../cuda/mfdfa_kernel.cuh"
#include "../cuda/utils_kernels.cuh"


class MFDFA
{
public:
    explicit MFDFA(double *h_y, int yLen);
    ~MFDFA();
    void computeFlucVec(int *winSizes, int nWins, double *qVals, int nq, double *hq, int threads, bool revSeg=false);
    void computeMultifractalSpectrum(int *winSizes, int nWins, double *qVals, int nq, double *a, double *fa, int threads, bool revSeg=false);
private:
    cudaError_t cudaErr;
    double *d_y;
    double *d_t;
    int len = 0;
};

#endif
