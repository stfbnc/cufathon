#ifndef UTILS_KERNELS
#define UTILS_KERNELS

#include "cuda_runtime.h"
#include "cuda.h"


void linRange(double *vec, int N, int start);

__global__
void doubleToLog(const double * __restrict__ vec, double * __restrict__ logVec, int N);

__global__
void intToLog(const int * __restrict__ vec, double * __restrict__ logVec, int N);

#endif
