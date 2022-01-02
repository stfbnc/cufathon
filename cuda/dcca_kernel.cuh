#ifndef __DCCA_KERNEL__
#define __DCCA_KERNEL__

#include <iostream>
#include <algorithm>
#include "cuda_runtime.h"
#include "cuda.h"
#include "curand.h"
#include "utils.cuh"

extern void cudaDCCA(float *y1, float *y2, int n, int *win_sizes, int n_wins, bool rev_seg, float *fxx, float *fyy, float *fxy, int n_threads);
extern void cudaDCCAConfInt(int *win_sizes, int n_wins, int n, int n_sim, float conf_level, float *conf_up, float *conf_down, int n_threads);

#endif
