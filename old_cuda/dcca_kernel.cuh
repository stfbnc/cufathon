#ifndef DCCA_KERNEL
#define DCCA_KERNEL

#include <iostream>
#include <algorithm>
#include "cuda_runtime.h"
#include "cuda.h"
#include "curand.h"
#include "utils_device.cuh"
#include "utils_kernels.cuh"


void cudaDCCA(float *y1, float *y2, float *t, int N, int *winSizes, int nWins, bool revSeg, float *rho, int nThreads);
void cudaDCCAConfInt(int *winSizes, int nWins, int N, int nSim, float confLevel, float *confUp, float *confDown, int nThreads);

#endif
