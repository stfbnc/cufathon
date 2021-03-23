#include <stdio.h>
#include "dfa_kernel.h"


__device__
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
void DFAKernel(const double * __restrict__ y, const double * __restrict__ t,
               int currWinSize, int Ns, double * __restrict__ f)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    //int stridex = blockDim.x * gridDim.x;

    //int ty = blockIdx.y * blockDim.y + threadIdx.y;
    //int stridey = blockDim.y * gridDim.y;

    //for(int tid = tx; tid < Ns; tid += stridex)
    if((tx < Ns))// && (ty < currWinSize))
    {
        //f[tx] = 0.0;
        int startLim = tx * currWinSize;
        double m = 0.0, q = 0.0;
        
        fit(currWinSize, t + startLim, y + startLim, &m, &q);

        for(int j = 0; j < currWinSize; j++)
        {
            double var = y[startLim + j] - (q + m * t[startLim + j]);
            //double var = y[startLim + ty] - (q + m * t[startLim + ty]);
            //f[tx * currWinSize + ty] = pow(var, 2.0);
            f[tx] += pow(var, 2.0);
        }
    }
}

void cudaDFA(double *y, double *t, int currWinSize,
             int Ns, double *f)
{
    int threadsPerBlock = 512;
    int blocksPerGrid = (Ns + threadsPerBlock - 1) / threadsPerBlock;
    DFAKernel<<<blocksPerGrid, threadsPerBlock>>>(y, t, currWinSize, Ns, f);
    cudaDeviceSynchronize();
}

