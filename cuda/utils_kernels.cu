#include <stdio.h>
#include "utils_kernels.cuh"


__global__
void linRangeKernel(double * __restrict__ vec, int N, int start)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tx < N)
    {
        vec[tx] = tx + start;
    }
}

void linRange(double *vec, int N, int start)
{
    int nThreads = 512;
    dim3 threadsPerBlock(nThreads);
    dim3 blocksPerGrid((N + nThreads - 1) / nThreads);
    linRangeKernel<<<blocksPerGrid, threadsPerBlock>>>(vec, N, start);
    cudaDeviceSynchronize();
}

__global__
void doubleToLog(const double * __restrict__ vec, double * __restrict__ logVec, int N)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    if(tx < N)
    {
        logVec[tx] = log(vec[tx]);
    } 
}

__global__
void intToLog(const int * __restrict__ vec, double * __restrict__ logVec, int N)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    if(tx < N)
    {
        logVec[tx] = log(1.0 * vec[tx]);
    }
}

