#ifndef __UTILS_DEVICE__
#define __UTILS_DEVICE__

#include <math.h>
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

__device__ inline
void fit(int L, int x_start, const float * __restrict__ y,
         float *ang_coeff, float *intercept)
{
    float sumx = 0.0f;
    float sumx2 = 0.0f;
    float sumxy = 0.0f;
    float sumy = 0.0f;

    for(int i = 0; i < L; i++)
    {
        sumx += (x_start + i);
        sumx2 += (x_start + i) * (x_start + i);
        sumxy += (x_start + i) * y[i];
        sumy += y[i];
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

__device__ inline
void h_fit(int L, const float * __restrict__ x, const float * __restrict__ y,
           float *ang_coeff, float *intercept)
{
    float sumx = 0.0f;
    float sumx2 = 0.0f;
    float sumxy = 0.0f;
    float sumy = 0.0f;

    for(int i = 0; i < L; i++)
    {
        sumx += x[i];
        sumx2 += x[i] * x[i];
        sumxy += x[i] * y[i]; 
        sumy += y[i];
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
