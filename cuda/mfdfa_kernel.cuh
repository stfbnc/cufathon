#ifndef MFDFA_KERNEL
#define MFDFA_KERNEL

#include "cuda_runtime.h"
#include "cuda.h"
#include "utils_device.cuh"
#include "utils_kernels.cuh"


void cudaMFDFA(double *y, double *t, int N, int *winSizes, int nWins, double *qVals, int nq, bool revSeg, double *hq, int nThreads);
void cudaMultifractalSpectrum(double *y, double *t, int N, int *winSizes, int nWins, double *qVals, int nq, bool revSeg, double *a, double *fa, int nThreads);

#endif
