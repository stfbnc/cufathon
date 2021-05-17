#ifndef DFA_KERNEL
#define DFA_KERNEL

#include "cuda_runtime.h"
#include "cuda.h"
#include "utils_device.cuh"
#include "utils_kernels.cuh"


void cudaDFA(float *y, float *t, int N, int *winSizes, int nWins, bool revSeg, float *flucVec, float *I, float *H, int nThreads);

#endif
