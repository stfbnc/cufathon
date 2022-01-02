#ifndef __HT_KERNEL__
#define __HT_KERNEL__

#include "utils.cuh"

extern void cudaHT(float *y, int n, int *scales, int n_scales, float *ht, int n_threads);

#endif
