#include "dfa.h"
#include <iostream>


DFA::DFA(double *h_y, int yLen)
{
    // reset internal state
    cudaErr = cudaGetLastError();

    // assign local variables and reserve memory on device
    len = yLen;
  
    cudaErr = cudaMalloc(&d_y, len * sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMalloc(&d_t, len * sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    // fill device arrays
    cudaErr = cudaMemcpy(d_y, h_y, len * sizeof(double), cudaMemcpyHostToDevice);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    linRange(d_t, len, 1);
    cudaErr = cudaGetLastError();
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
}

DFA::~DFA()
{
    cudaErr = cudaFree(d_y);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    
    cudaErr = cudaFree(d_t);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
}

void DFA::computeFlucVec(int *winSizes, int nWins, double *F, bool revSeg)
{
    double *flucVec = nullptr;
    cudaErr = cudaMalloc(&flucVec, nWins * sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    int *winSizesGpu = nullptr;
    cudaErr = cudaMalloc(&winSizesGpu, nWins * sizeof(int));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMemcpy(winSizesGpu, winSizes, nWins * sizeof(int), cudaMemcpyHostToDevice);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaDFA(d_y, d_t, len, winSizesGpu, nWins, flucVec, 64);
    cudaErr = cudaGetLastError();
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMemcpy(F, flucVec, nWins * sizeof(double), cudaMemcpyDeviceToHost);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaFree(flucVec);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaFree(winSizesGpu);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    fprintf(stderr, "F[0]: %lf\n", F[0]);
    fprintf(stderr, "F[%d]: %lf\n", nWins / 2, F[nWins / 2]);
    fprintf(stderr, "F[%d]: %lf\n", nWins - 1, F[nWins - 1]);
}

void DFA::computeFlucVecInner(int *winSizes, int nWins, double *F, bool revSeg)
{
    double *flucVec = nullptr;
    cudaErr = cudaMalloc(&flucVec, nWins * sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    int *winSizesGpu = nullptr;
    cudaErr = cudaMalloc(&winSizesGpu, nWins * sizeof(int));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMemcpy(winSizesGpu, winSizes, nWins * sizeof(int), cudaMemcpyHostToDevice);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaDFAInner(d_y, d_t, len, winSizesGpu, nWins, flucVec, 512);
    cudaErr = cudaGetLastError();
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMemcpy(F, flucVec, nWins * sizeof(double), cudaMemcpyDeviceToHost);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaFree(flucVec);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaFree(winSizesGpu);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    fprintf(stderr, "F[0]: %lf\n", F[0]);
    fprintf(stderr, "F[%d]: %lf\n", nWins / 2, F[nWins / 2]);
    fprintf(stderr, "F[%d]: %lf\n", nWins - 1, F[nWins - 1]);
}

