#ifndef UTILS_DEVICE
#define UTILS_DEVICE

#include <stdio.h>
#include "cuda_runtime.h"
#include "cuda.h"


__device__ inline
void fit(int L, const double * __restrict__ x, const double * __restrict__ y,
         double *ang_coeff, double *intercept)
{
    double sumx = 0.0;
    double sumx2 = 0.0;
    double sumxy = 0.0;
    double sumy = 0.0;
    double sumy2 = 0.0;

    for(int i = 0; i < L; i++)
    {
        sumx += x[i];
        sumx2 += x[i] * x[i];
        sumxy += x[i] * y[i];
        sumy += y[i];
        sumy2 += y[i] * y[i];
    }

    double denom = (L * sumx2 - sumx * sumx);

    if(denom == 0.0)
    {
        *ang_coeff = 0.0;
        *intercept = 0.0;
        return;
    }

    *ang_coeff = (L * sumxy - sumx * sumy) / (double)denom;
    *intercept = (sumy * sumx2 - sumx * sumxy) / (double)denom;
}

__global__
void fit_intX(int L, const int * __restrict__ x, const double * __restrict__ y,
              double *ang_coeff, double *intercept);

#endif
