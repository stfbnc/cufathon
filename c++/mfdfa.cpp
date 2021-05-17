#include "mfdfa.h"
#include <iostream>


MFDFA::MFDFA(float *h_y, int yLen)
{
    // reset internal state
    cudaErr = cudaGetLastError();

    // assign local variables and reserve memory on device
    len = yLen;
  
    cudaErr = cudaMalloc(&d_y, len * sizeof(float));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMalloc(&d_t, len * sizeof(float));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    // fill device arrays
    cudaErr = cudaMemcpy(d_y, h_y, len * sizeof(float), cudaMemcpyHostToDevice);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    linRange(d_t, len, 1);
    cudaErr = cudaGetLastError();
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
}

MFDFA::~MFDFA()
{
    cudaErr = cudaFree(d_y);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    
    cudaErr = cudaFree(d_t);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
}

void MFDFA::computeFlucVec(int *winSizes, int nWins, float *qVals, int nq, float *hq, int threads, bool revSeg)
{
    cudaMFDFA(d_y, d_t, len, winSizes, nWins, qVals, nq, revSeg, hq, threads);
    cudaErr = cudaGetLastError();
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    fprintf(stderr, "hq[0]: %lf\n", hq[0]);
    fprintf(stderr, "hq[%d]: %lf\n", nq / 2, hq[nq / 2]);
    fprintf(stderr, "hq[%d]: %lf\n", nq - 1, hq[nq - 1]);
}

void MFDFA::computeMultifractalSpectrum(int *winSizes, int nWins, float *qVals, int nq, float *a, float *fa, int threads, bool revSeg)
{
    cudaMultifractalSpectrum(d_y, d_t, len, winSizes, nWins, qVals, nq, revSeg, a, fa, threads);
    cudaErr = cudaGetLastError();
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    fprintf(stderr, "a[0]: %lf | fa[0] = %lf\n", a[0], fa[0]);
    fprintf(stderr, "a[%d]: %lf | fa[%d] = %lf\n", nq / 2, a[nq / 2], nq / 2, fa[nq / 2]);
    fprintf(stderr, "a[%d]: %lf | fa[%d] = %lf\n", nq - 2, a[nq - 2], nq - 2, fa[nq - 2]);
}

