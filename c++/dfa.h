#ifndef DFA_H
#define DFA_H

#include <math.h>
#include "../cuda/dfa_kernel.cuh"
#include "../cuda/utils_kernels.cuh"


class DFA
{
public:
    explicit DFA(float *h_y, int yLen);
    ~DFA();
    void computeFlucVec(int *winSizes, int nWins, float *F, float I, float H, int threads, bool revSeg=false);
private:
    cudaError_t cudaErr;
    float *d_y;
    float *d_t;
    int len = 0;
};

#endif
