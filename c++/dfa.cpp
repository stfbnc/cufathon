#include "dfa.h"
#include <iostream>


DFA::DFA(float *h_y, int yLen)
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

DFA::~DFA()
{
    cudaErr = cudaFree(d_y);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    
    cudaErr = cudaFree(d_t);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
}

void DFA::computeFlucVec(int *winSizes, int nWins, float *F, float I, float H, int threads, bool revSeg)
{
    cudaDFA(d_y, d_t, len, winSizes, nWins, revSeg, F, &I, &H, threads);
    cudaErr = cudaGetLastError();
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    fprintf(stderr, "F[0]: %lf\n", F[0]);
    fprintf(stderr, "F[%d]: %lf\n", nWins / 2, F[nWins / 2]);
    fprintf(stderr, "F[%d]: %lf\n", nWins - 1, F[nWins - 1]);

    fprintf(stderr, "I = %lf, H = %lf\n", I, H);
}

