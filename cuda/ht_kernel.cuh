#ifndef HT_KERNEL
#define HT_KERNEL

#include "cuda_runtime.h"
#include "cuda.h"
#include "utils_device.cuh"


void cudaHT(double *y, double *t, int N, int *scales, int nScales, double *flucVec, int nThreads, int nThreads_mfdfa);
void cudaHT_2(double *y, double *t, int N, int *scales, int nScales, double *flucVec, int nThreads, int nThreads_mfdfa);

#endif
