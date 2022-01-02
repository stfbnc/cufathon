#ifndef __DFA_KERNEL__
#define __DFA_KERNEL__

#include "utils.cuh"

extern void cudaDFA(float *y, int n, int *win_sizes, int n_wins, bool rev_seg, float *fluc_vec, int n_threads);

#endif
