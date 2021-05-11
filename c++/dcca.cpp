#include "dcca.h"
#include <iostream>


DCCA::DCCA(double *h_y1, double *h_y2, int yLen)
{
    // reset internal state
    cudaErr = cudaGetLastError();

    // assign local variables and reserve memory on device
    len = yLen;
  
    cudaErr = cudaMalloc(&d_y1, len * sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMalloc(&d_y2, len * sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMalloc(&d_t, len * sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    // fill device arrays
    cudaErr = cudaMemcpy(d_y1, h_y1, len * sizeof(double), cudaMemcpyHostToDevice);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMemcpy(d_y2, h_y2, len * sizeof(double), cudaMemcpyHostToDevice);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    linRange(d_t, len, 1);
    cudaErr = cudaGetLastError();
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
}

DCCA::~DCCA()
{
    cudaErr = cudaFree(d_y1);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    
    cudaErr = cudaFree(d_y2);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaFree(d_t);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
}

void DCCA::computeFlucVec(int *winSizes, int nWins, double *rho, int threads, bool revSeg)
{
    cudaDCCA(d_y1, d_y2, d_t, len, winSizes, nWins, revSeg, rho, threads);
    cudaErr = cudaGetLastError();
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    fprintf(stderr, "p[0]: %lf\n", rho[0]);
    fprintf(stderr, "p[%d]: %lf\n", nWins / 2, rho[nWins / 2]);
    fprintf(stderr, "p[%d]: %lf\n", nWins - 1, rho[nWins - 1]);
}
