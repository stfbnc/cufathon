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
    void computeFlucVec(int *winSizes, int nWins, double *F, double I, double H, int threads, bool revSeg=false);
private:
    cudaError_t cudaErr;
    double *d_y;
    double *d_t;
    int len = 0;
};

#endif
