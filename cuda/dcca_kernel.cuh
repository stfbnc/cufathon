#ifndef DCCA_KERNEL
#define DCCA_KERNEL

#include "cuda_runtime.h"
#include "cuda.h"
#include "utils_device.cuh"
#include "utils_kernels.cuh"


void cudaDCCA(double *y1, double *y2, double *t, int N, int *winSizes, int nWins, bool revSeg, double *rho, int nThreads);

#endif
