#include "mfdfa.h"
#include <iostream>


MFDFA::MFDFA(double *h_y, int yLen)
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

MFDFA::~MFDFA()
{
    cudaErr = cudaFree(d_y);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    
    cudaErr = cudaFree(d_t);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
}

void MFDFA::computeFlucVec(int *winSizes, int nWins, double *qVals, int nq, double *F, int threads, bool revSeg)
{
    double *d_F = nullptr;
    cudaErr = cudaMalloc(&d_F, nWins * sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    int *d_winSizes = nullptr;
    cudaErr = cudaMalloc(&d_winSizes, nWins * sizeof(int));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMemcpy(d_winSizes, winSizes, nWins * sizeof(int), cudaMemcpyHostToDevice);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    for(int iq = 0; iq < nq; iq++)
    {
        cudaMFDFA(d_y, d_t, len, d_winSizes, nWins, qVals[iq], d_F, threads);
        cudaErr = cudaGetLastError();
        if(cudaErr != cudaSuccess)
            fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

        cudaErr = cudaMemcpy(&F[iq * nWins], d_F, nWins * sizeof(double), cudaMemcpyDeviceToHost);
        if(cudaErr != cudaSuccess)
            fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    }

    cudaErr = cudaFree(d_F);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaFree(d_winSizes);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    fprintf(stderr, "F[0]: %lf\n", F[0]);
    fprintf(stderr, "F[%d]: %lf\n", nWins * nq / 2, F[nWins * nq / 2]);
    fprintf(stderr, "F[%d]: %lf\n", nWins * nq - 1, F[nWins * nq - 1]);
}

void MFDFA::computeFlucVec2D(int *winSizes, int nWins, double *qVals, int nq, double *F, int threads, bool revSeg)
{
    double *d_F = nullptr;
    cudaErr = cudaMalloc(&d_F, nWins * nq * sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    int *d_winSizes = nullptr;
    cudaErr = cudaMalloc(&d_winSizes, nWins * sizeof(int));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMemcpy(d_winSizes, winSizes, nWins * sizeof(int), cudaMemcpyHostToDevice);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    double *d_qVals = nullptr;
    cudaErr = cudaMalloc(&d_qVals, nq * sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMemcpy(d_qVals, qVals, nq * sizeof(double), cudaMemcpyHostToDevice);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaMFDFA2D(d_y, d_t, len, d_winSizes, nWins, d_qVals, nq, d_F, threads);
    cudaErr = cudaGetLastError();
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMemcpy(F, d_F, nWins * nq * sizeof(double), cudaMemcpyDeviceToHost);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaFree(d_F);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaFree(d_winSizes);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaFree(d_qVals);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    fprintf(stderr, "F[0]: %lf\n", F[0]);
    fprintf(stderr, "F[%d]: %lf\n", nWins * nq / 2, F[nWins * nq / 2]);
    fprintf(stderr, "F[%d]: %lf\n", nWins * nq - 1, F[nWins * nq - 1]);
}

