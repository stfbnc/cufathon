#ifndef UTILS_KERNELS
#define UTILS_KERNELS

#include "cuda_runtime.h"
#include "cuda.h"


void linRange(float *vec, int N, int start);

__global__
void floatToLog(const float * __restrict__ vec, float * __restrict__ logVec, int N);

__global__
void intToLog(const int * __restrict__ vec, float * __restrict__ logVec, int N);

#endif
