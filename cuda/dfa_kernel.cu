#include <stdio.h>
#include "dfa_kernel.cuh"


__global__
void DFAKernel(const double * __restrict__ y, const double * __restrict__ t, int N,
               const int * __restrict__ winSizes, int nWins, double * __restrict__ flucVec)
{
    int nWin = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(nWin < nWins)
    {
        int currWinSize = winSizes[nWin];
        int Ns = N / currWinSize;
        double f = 0.0;
        
        for(int i = 0; i < Ns; i++)
        {
            int startLim = i * currWinSize;
            double m = 0.0, q = 0.0;

            fit(currWinSize, t + startLim, y + startLim, &m, &q);

            for(int j = 0; j < currWinSize; j++)
            {
                double var = y[startLim + j] - (q + m * t[startLim + j]);
                f += pow(var, 2.0);
            }
        }

        flucVec[nWin] = sqrt(f / (Ns * currWinSize));
    }
}

void cudaDFA(double *y, double *t, int N, int *winSizes, int nWins, double *flucVec)
{
    int threadsPerBlock = 512;
    int blocksPerGrid = (nWins + threadsPerBlock - 1) / threadsPerBlock;
    DFAKernel<<<blocksPerGrid, threadsPerBlock>>>(y, t, N, winSizes, nWins, flucVec);
    cudaDeviceSynchronize();
}

__global__
void DFAKernelInner(const double * __restrict__ y, const double * __restrict__ t,
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

void cudaDFAInner(double *y, double *t, int currWinSize,
             int Ns, double *f)
{
    int threadsPerBlock = 512;
    int blocksPerGrid = (Ns + threadsPerBlock - 1) / threadsPerBlock;
    DFAKernelInner<<<blocksPerGrid, threadsPerBlock>>>(y, t, currWinSize, Ns, f);
    cudaDeviceSynchronize();
}

