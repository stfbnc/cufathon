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

void cudaDFA(double *y, double *t, int N, int *winSizes, int nWins, double *flucVec, double *I, double *H, int nThreads)
{
    cudaError_t cudaErr;

    dim3 threadsPerBlock(nThreads);
    dim3 blocksPerGrid((nWins + nThreads - 1) / nThreads);
    DFAKernel<<<blocksPerGrid, threadsPerBlock>>>(y, t, N, winSizes, nWins, flucVec);
    cudaDeviceSynchronize();

    double *logW, *logF;
    cudaErr = cudaMalloc(&logW, nWins * sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMalloc(&logF, nWins * sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    
    cudaStream_t stream_1, stream_2;
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);
    doubleToLog<<<blocksPerGrid, threadsPerBlock, 0, stream_1>>>(flucVec, logF, nWins);
    intToLog<<<blocksPerGrid, threadsPerBlock, 0, stream_2>>>(winSizes, logW, nWins);
    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);

    double *d_H, *d_I;
    cudaErr = cudaMalloc(&d_H, sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMalloc(&d_I, sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    hFit<<<1, 1>>>(nWins, logW, logF, d_H, d_I);
    cudaDeviceSynchronize();

    cudaErr = cudaMemcpy(I, d_I, sizeof(double), cudaMemcpyDeviceToHost);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMemcpy(H, d_H, sizeof(double), cudaMemcpyDeviceToHost);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaFree(d_H);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaFree(d_I);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    fprintf(stderr, "I = %lf, H = %lf\n", *I, *H);
}

__global__
void DFAKernelInner(const double * __restrict__ y, const double * __restrict__ t, int N,
                    const int * __restrict__ winSizes, int nWins, double * __restrict__ flucVec)
{
    int nWin = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double sh[];
    double *s_y = &sh[0];
    double *s_t = &sh[N];

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

void cudaDFAInner(double *y, double *t, int N, int *winSizes, int nWins, double *flucVec, int nThreads)
{
    dim3 threadsPerBlock(nThreads);
    dim3 blocksPerGrid((nWins + nThreads - 1) / nThreads);
    DFAKernelInner<<<blocksPerGrid, threadsPerBlock, 2 * N * sizeof(double)>>>(y, t, N, winSizes, nWins, flucVec);
    cudaDeviceSynchronize();
}
