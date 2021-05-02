#ifndef DFA_KERNEL
#define DFA_KERNEL

#include "cuda_runtime.h"
#include "cuda.h"
#include "utils_device.cuh"
#include "utils_kernels.cuh"


void cudaDFA(double *y, double *t, int N, int *winSizes, int nWins, bool revSeg, double *flucVec, double *I, double *H, int nThreads);

#endif
