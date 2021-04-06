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
    int nThreads = 512;
    dim3 threadsPerBlock(nThreads);
    dim3 blocksPerGrid((nWins + nThreads - 1) / nThreads);
    DFAKernel<<<blocksPerGrid, threadsPerBlock>>>(y, t, N, winSizes, nWins, flucVec);
    cudaDeviceSynchronize();
}


__global__
void DFAKernelInner(const double * __restrict__ y, const double * __restrict__ t, int N,
                    const int * __restrict__ winSizes, int nWins, double * __restrict__ flucVec)
{
    int nWin = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double s_y[];
    extern __shared__ double s_t[];

    if(nWin == 0)
    {
        for(int i = 0; i < N; i++)
        {
            s_y[i] = y[i];
            s_t[i] = t[i];
        }
    }

    __syncthreads();

    if(nWin < nWins)
    {   
        int currWinSize = winSizes[nWin];
        int Ns = N / currWinSize;
        double f = 0.0;
        
        for(int i = 0; i < Ns; i++)
        {   
            int startLim = i * currWinSize;
            double m = 0.0, q = 0.0;
            
            fit(currWinSize, s_t + startLim, s_y + startLim, &m, &q);
            
            for(int j = 0; j < currWinSize; j++)
            {   
                double var = s_y[startLim + j] - (q + m * s_t[startLim + j]);
                f += pow(var, 2.0);
            }
        }
        
        flucVec[nWin] = sqrt(f / (Ns * currWinSize));
    }
}

void cudaDFAInner(double *y, double *t, int N, int *winSizes, int nWins, double *flucVec)
{
    int nThreads = 512;
    dim3 threadsPerBlock(nThreads);
    dim3 blocksPerGrid((nWins + nThreads - 1) / nThreads);
    DFAKernelInner<<<blocksPerGrid, threadsPerBlock, N * sizeof(double)>>>(y, t, N, winSizes, nWins, flucVec);
    cudaDeviceSynchronize();
}
