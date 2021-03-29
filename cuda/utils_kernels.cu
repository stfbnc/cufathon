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
    int threadsPerBlock = 512;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    linRangeKernel<<<blocksPerGrid, threadsPerBlock>>>(vec, N, start);
    cudaDeviceSynchronize();
}

