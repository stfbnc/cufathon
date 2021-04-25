#include "ht.h"
#include <iostream>


HT::HT(double *h_y, int yLen)
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

HT::~HT()
{
    cudaErr = cudaFree(d_y);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    
    cudaErr = cudaFree(d_t);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
}

void HT::computeFlucVec(int *scales, int nScales, double *F, int threads, int threads_mfdfa)
{
    int sLen = 0;
    for(int i = 0; i < nScales; i++)
    {
        sLen += (len - scales[i] + 1);
    }

    double *flucVec = nullptr;
    cudaErr = cudaMalloc(&flucVec, sLen * sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaHT(d_y, d_t, len, scales, nScales, flucVec, threads, threads_mfdfa);
    cudaErr = cudaGetLastError();
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMemcpy(F, flucVec, sLen * sizeof(double), cudaMemcpyDeviceToHost);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaFree(flucVec);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    fprintf(stderr, "F[0]: %lf\n", F[0]);
    fprintf(stderr, "F[%d]: %lf\n", sLen / 2, F[sLen / 2]);
    fprintf(stderr, "F[%d]: %lf\n", sLen - 1, F[sLen - 1]);
}
void HT::computeFlucVec_2(int *scales, int nScales, double *F, int threads, int threads_mfdfa)
{
    int sLen = 0;
    for(int i = 0; i < nScales; i++)
    {
        sLen += (len - scales[i] + 1);
    }

    double *flucVec = nullptr;
    cudaErr = cudaMalloc(&flucVec, sLen * sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaHT_2(d_y, d_t, len, scales, nScales, flucVec, threads, threads_mfdfa);
    cudaErr = cudaGetLastError();
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaMemcpy(F, flucVec, sLen * sizeof(double), cudaMemcpyDeviceToHost);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    cudaErr = cudaFree(flucVec);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    fprintf(stderr, "F[0]: %lf\n", F[0]);
    fprintf(stderr, "F[%d]: %lf\n", sLen / 2, F[sLen / 2]);
    fprintf(stderr, "F[%d]: %lf\n", sLen - 1, F[sLen - 1]);
}
