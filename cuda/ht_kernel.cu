#include <stdio.h>
#include "ht_kernel.cuh"


__global__
void MFDFAforHTKernel(const double * __restrict__ y, const double * __restrict__ t, int N,
                      const int * __restrict__ winSizes, int nWins,
                      double * __restrict__ flucVec_mfdfa)
{
    int iw = blockIdx.x * blockDim.x + threadIdx.x;

    if(iw < nWins)
    {
        int currWinSize = winSizes[iw];
        int Ns = N / currWinSize;
        double f = 0.0;

        for(int i = 0; i < Ns; i++)
        {
            double rms = 0.0;
            int startLim = i * currWinSize;
            double m = 0.0, q = 0.0;

            fit(currWinSize, t + startLim, y + startLim, &m, &q);

            for(int j = 0; j < currWinSize; j++)
            {
                double var = y[startLim + j] - (q + m * t[startLim + j]);
                rms += pow(var, 2.0);
            }

            f += log(rms / currWinSize);
        }

        flucVec_mfdfa[iw] = exp(f / (2.0 * Ns));
    }
}

__global__
void HTKernel_2(const double * __restrict__ y, const double * __restrict__ t, int N,
              const int * __restrict__ scales, const int * __restrict__ prevScales, int nScales,
              double * __restrict__ flucVec)
{
    int nScale = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(nScale < nScales)
    {
        int scale = scales[nScale];
        int prevScale = prevScales[nScale];
        int Ns = N - scale + 1;

        for(int i = 0; i < Ns; i++)
        {
            double f = 0.0;
            double m = 0.0, q = 0.0;

            fit(scale, t + i, y + i, &m, &q);

            for(int j = 0; j < scale; j++)
            {
                double var = y[i + j] - (q + m * t[i + j]);
                f += pow(var, 2.0);
            }

            flucVec[prevScale + i] = sqrt(f / scale);
        }
    }
}

__global__
void HTKernel(const double * __restrict__ y, const double * __restrict__ t, int N,
              int scale, int prevScale, int Ns, double * __restrict__ flucVec)
{   
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < Ns)
    {   
        double f = 0.0; 
        double m = 0.0, q = 0.0;
            
        fit(scale, t + i, y + i, &m, &q);
            
        for(int j = 0; j < scale; j++)
        {   
            double var = y[i + j] - (q + m * t[i + j]);
            f += pow(var, 2.0);
        }
            
        flucVec[prevScale + i] = sqrt(f / scale);
    }
}

__global__
void finalHTKernel(double * __restrict__ vecht, double Ns,
      int scale, int prevScale,
      double *H0, double *H0_intercept)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < Ns)
    {
        double dscale = static_cast<double>(scale);
        vecht[prevScale + i] = (*H0_intercept + *H0 * log(dscale) - log(vecht[prevScale + i])) / (log(Ns) - log(dscale)) + *H0;
    }
}

void cudaHT(double *y, double *t, int N, int *scales, int nScales, double *flucVec, int nThreads, int nThreads_mfdfa)
{
    cudaError_t cudaErr;
    
    // ht variables
    int *prevScales = new int [nScales];
    for(int i = 0; i < nScales; i++)
    {
        prevScales[i] = 0;
        for(int j = 0; j < i; j++)
        {
            prevScales[i] += (N - scales[j] + 1);
        }
    }

    // mfdfa variables
    int nWins = 20;
    int winSizes[nWins];
    int winStep = (N / 4 - 10 + 1) / nWins;
    for(int i = 0; i < nWins; i++)
    {
        winSizes[i] = 10 + i * winStep;
    }

    int *d_winSizes;
    cudaErr = cudaMalloc(&d_winSizes, nWins * sizeof(int));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMemcpy(d_winSizes, winSizes, nWins * sizeof(int), cudaMemcpyHostToDevice);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    double *flucVec_mfdfa;
    cudaErr = cudaMalloc(&flucVec_mfdfa, nWins * sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    // kernel parameters
    dim3 threadsPerBlock_mfdfa(nThreads_mfdfa);
    dim3 blocksPerGrid_mfdfa((nWins + nThreads_mfdfa - 1) / nThreads_mfdfa);
    dim3 threadsPerBlock(nThreads);

    cudaStream_t stream_1, stream_2;
    cudaErr = cudaStreamCreate(&stream_1);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaStreamCreate(&stream_2);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    MFDFAforHTKernel<<<blocksPerGrid_mfdfa, threadsPerBlock_mfdfa, 0, stream_1>>>(y, t, N, d_winSizes, nWins, flucVec_mfdfa);
    for(int i = 0; i < nScales; i++)
    {
        int Ns = N - scales[i] + 1;
        dim3 blocksPerGrid((Ns + nThreads - 1) / nThreads);
        HTKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_2>>>(y, t, N, scales[i], prevScales[i], Ns, flucVec);
    }
    cudaDeviceSynchronize();

    cudaErr = cudaStreamDestroy(stream_1);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaStreamDestroy(stream_2);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    double *d_H_mfdfa, *d_I_mfdfa;
    cudaErr = cudaMalloc(&d_H_mfdfa, sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMalloc(&d_I_mfdfa, sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    fit_intX<<<1, 1>>>(nWins, d_winSizes, flucVec_mfdfa, d_H_mfdfa, d_I_mfdfa);
    cudaDeviceSynchronize();

    for(int i = 0; i < nScales; i++)
    {
        double Ns = N - scales[i] + 1;
        dim3 blocksPerGrid((Ns + nThreads - 1) / nThreads);
        finalHTKernel<<<blocksPerGrid, threadsPerBlock>>>(flucVec, Ns, scales[i], prevScales[i], d_H_mfdfa, d_I_mfdfa);
    }
    cudaDeviceSynchronize();

    delete [] prevScales;

    cudaErr = cudaFree(d_winSizes);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaFree(flucVec_mfdfa);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
}

void cudaHT_2(double *y, double *t, int N, int *scales, int nScales, double *flucVec, int nThreads, int nThreads_mfdfa)
{
    cudaError_t cudaErr;

    // ht variables
    int *d_scales;
    cudaErr = cudaMalloc(&d_scales, nScales * sizeof(int));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMemcpy(d_scales, scales, nScales * sizeof(int), cudaMemcpyHostToDevice);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    int *prevScales = new int [nScales];
    for(int i = 0; i < nScales; i++)
    {   
        prevScales[i] = 0; 
        for(int j = 0; j < i; j++)
        {   
            prevScales[i] += (N - scales[j] + 1);
        }
    }

    int *d_prevScales;
    cudaErr = cudaMalloc(&d_prevScales, nScales * sizeof(int));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMemcpy(d_prevScales, prevScales, nScales * sizeof(int), cudaMemcpyHostToDevice);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    // mfdfa variables
    int nWins = 20;
    int winSizes[nWins];
    int winStep = (N / 4 - 10 + 1) / nWins;
    for(int i = 0; i < nWins; i++)
    {
        winSizes[i] = 10 + i * winStep;
    }

    int *d_winSizes;
    cudaErr = cudaMalloc(&d_winSizes, nWins * sizeof(int));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMemcpy(d_winSizes, winSizes, nWins * sizeof(int), cudaMemcpyHostToDevice);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    double *flucVec_mfdfa;
    cudaErr = cudaMalloc(&flucVec_mfdfa, nWins * sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    // kernel parameters
    dim3 threadsPerBlock_mfdfa(nThreads_mfdfa);
    dim3 blocksPerGrid_mfdfa((nWins + nThreads_mfdfa - 1) / nThreads_mfdfa);
    dim3 threadsPerBlock(nThreads);
    dim3 blocksPerGrid((nScales + nThreads - 1) / nThreads);

    cudaStream_t stream_1, stream_2;
    cudaErr = cudaStreamCreate(&stream_1);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaStreamCreate(&stream_2);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    MFDFAforHTKernel<<<blocksPerGrid_mfdfa, threadsPerBlock_mfdfa, 0, stream_1>>>(y, t, N, d_winSizes, nWins, flucVec_mfdfa);
    HTKernel_2<<<blocksPerGrid, threadsPerBlock, 0, stream_2>>>(y, t, N, d_scales, d_prevScales, nScales, flucVec);
    cudaDeviceSynchronize();

    cudaErr = cudaStreamDestroy(stream_1);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaStreamDestroy(stream_2);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));

    double *d_H_mfdfa, *d_I_mfdfa;
    cudaErr = cudaMalloc(&d_H_mfdfa, sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaMalloc(&d_I_mfdfa, sizeof(double));
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    fit_intX<<<1, 1>>>(nWins, d_winSizes, flucVec_mfdfa, d_H_mfdfa, d_I_mfdfa);
    cudaDeviceSynchronize();

    for(int i = 0; i < nScales; i++)
    {
        double Ns = N - scales[i] + 1;
        dim3 blocksPerGrid((Ns + nThreads - 1) / nThreads);
        finalHTKernel<<<blocksPerGrid, threadsPerBlock>>>(flucVec, Ns, scales[i], prevScales[i], d_H_mfdfa, d_I_mfdfa);
    }
    cudaDeviceSynchronize();

    delete [] prevScales;

    cudaErr = cudaFree(d_scales);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaFree(d_prevScales);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaFree(d_winSizes);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
    cudaErr = cudaFree(flucVec_mfdfa);
    if(cudaErr != cudaSuccess)
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaErr));
}

