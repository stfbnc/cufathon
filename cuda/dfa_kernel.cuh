#ifndef DFA_KERNEL
#define DFA_KERNEL

#include <stdio.h>
#include "cuda_runtime.h"
#include "cuda.h"
#include "utils_device.cuh"
#include "utils_kernels.cuh"


void cudaDFA(double *y, double *t, int N, int *winSizes, int nWins, double *flucVec, double *I, double *H, int nThreads);
void cudaDFAInner(double *y, double *t, int N, int *winSizes, int nWins, double *flucVec, int nThreads);

#endif
