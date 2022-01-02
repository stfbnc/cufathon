#ifndef __MFDFA_KERNEL__
#define __MFDFA_KERNEL__

#include "utils.cuh"

extern void cudaMFDFA(float *y, int n, int *win_sizes, int n_wins, float *q_vals, int nq, bool rev_seg, float *fluc_vec, int n_threads);

#endif
