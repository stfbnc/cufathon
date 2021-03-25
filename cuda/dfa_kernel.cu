#include <stdio.h>
#include "dfa_kernel.h"


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

