#ifndef DFA_H
#define DFA_H

#include <math.h>
#include "../cuda/dfa_kernel.cuh"
#include "../cuda/utils_kernels.cuh"


class DFA
{
public:
    explicit DFA(double *h_y, int yLen);
    ~DFA();
    void computeFlucVec(int *winSizes, int nWins, double *F, int threads, bool revSeg=false);
    void computeFlucVecInner(int *winSizes, int nWins, double *F, bool revSeg=false);
private:
    cudaError_t cudaErr;
    double *d_y = nullptr;
    double *d_t = nullptr;
    int len = 0;
};

#endif
