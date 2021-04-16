#ifndef MFDFA_KERNEL
#define MFDFA_KERNEL

#include "cuda_runtime.h"
#include "cuda.h"
#include "utils_device.cuh"


void cudaMFDFA(double *y, double *t, int N, int *winSizes, int nWins, double qVal, double *flucVec, int nThreads);
void cudaMFDFA2D(double *y, double *t, int N, int *winSizes, int nWins, double *qVals, int nq, double *flucVec, int nThreads);

#endif
