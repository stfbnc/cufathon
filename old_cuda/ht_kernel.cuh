#ifndef HT_KERNEL
#define HT_KERNEL

#include "cuda_runtime.h"
#include "cuda.h"
#include "utils_device.cuh"
#include "utils_kernels.cuh"


void cudaHT(float *y, float *t, int N, int *scales, int nScales, float *ht, int nThreads);

#endif
