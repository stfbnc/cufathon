#ifndef DFA_H
#define DFA_H

#include "../cuda/dfa_kernel.h"
#include "../cuda/utils_kernels.h"


class DFA
{
public:
    explicit DFA(double *y, int yLen);
    ~DFA();
    void computeFlucVec(int *winSizes, int nWins, double *F, bool revSeg=false);
private:
    double *y = nullptr;
    double *t = nullptr;
    int yLen = 0;
};

#endif
