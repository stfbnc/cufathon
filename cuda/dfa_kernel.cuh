#ifndef DFA_KERNEL
#define DFA_KERNEL

#include "cuda_runtime.h"
#include "cuda.h"
#include "utils_device.cuh"


void cudaDFA(double *y, double *t, int N, int *winSizes, int nWins, double *flucVec);
void cudaDFAInner(double *y, double *t, int N, int *winSizes, int nWins, double *flucVec);

#endif
