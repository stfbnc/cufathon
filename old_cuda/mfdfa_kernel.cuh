#ifndef MFDFA_KERNEL
#define MFDFA_KERNEL

#include "cuda_runtime.h"
#include "cuda.h"
#include "utils_device.cuh"
#include "utils_kernels.cuh"


void cudaMFDFA(float *y, float *t, int N, int *winSizes, int nWins, float *qVals, int nq, bool revSeg, float *hq, int nThreads);
void cudaMultifractalSpectrum(float *y, float *t, int N, int *winSizes, int nWins, float *qVals, int nq, bool revSeg, float *a, float *fa, int nThreads);

#endif
