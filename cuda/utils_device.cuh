#ifndef __UTILS_DEVICE__
#define __UTILS_DEVICE__

//#include <stdio.h>
#include "cuda_runtime.h"
#include "cuda.h"

__device__ inline
void fit(int L, const float * __restrict__ x, const float * __restrict__ y,
         float *ang_coeff, float *intercept)
{
    float sumx = 0.0f;
    float sumx2 = 0.0f;
    float sumxy = 0.0f;
    float sumy = 0.0f;
    float sumy2 = 0.0f;

    for(int i = 0; i < L; i++)
    {
        sumx += x[i];
        sumx2 += x[i] * x[i];
        sumxy += x[i] * y[i];
        sumy += y[i];
        sumy2 += y[i] * y[i];
    }

    float denom = (L * sumx2 - sumx * sumx);

    if(denom == 0.0f)
    {
        *ang_coeff = 0.0f;
        *intercept = 0.0f;
        return;
    }

    *ang_coeff = (L * sumxy - sumx * sumy) / (float)denom;
    *intercept = (sumy * sumx2 - sumx * sumxy) / (float)denom;
}

#endif
