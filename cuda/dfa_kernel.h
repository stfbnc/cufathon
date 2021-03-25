#ifndef DFA_KERNEL
#define DFA_KERNEL

#include "cuda_runtime.h"
#include "cuda.h"
#include "utils_device.h"


void cudaDFA(double *y, double *t, int currWinSize, int Ns, double *f);

#endif
