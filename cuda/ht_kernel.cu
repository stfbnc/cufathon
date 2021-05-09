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
            double rms = 0.0, rms2 = 0.0;
            int startLim = i * currWinSize;
            double m = 0.0, q = 0.0;

            fit(currWinSize, t + startLim, y + startLim, &m, &q);

            for(int j = 0; j < currWinSize; j++)
            {
                double var = y[startLim + j] - (q + m * t[startLim + j]);
                rms += pow(var, 2.0);
            }

            startLim = i * currWinSize + (N - Ns * currWinSize);
            fit(currWinSize, t + startLim, y + startLim, &m, &q);

            for(int j = 0; j < currWinSize; j++)
            {
                double var = y[startLim + j] - (q + m * t[startLim + j]);
                rms2 += pow(var, 2.0);
            }

            f += log(rms / currWinSize) + log(rms2 / currWinSize);
        }

        flucVec_mfdfa[iw] = exp(f / (4.0 * Ns));
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

void cudaHT(double *y, double *t, int N, int *scales, int nScales, double *ht, int nThreads)
{
    // ht variables
    int *prevScales = new int [nScales];
    int sLen = 0;
    for(int i = 0; i < nScales; i++)
    {
        sLen += (N - scales[i] + 1);
        prevScales[i] = 0;
        for(int j = 0; j < i; j++)
        {
            prevScales[i] += (N - scales[j] + 1);
        }
    }

    double *d_ht;
    cudaMalloc(&d_ht, sLen * sizeof(double));

    // mfdfa variables
    int nWins = 20;
    int winSizes[nWins];
    int winStep = round((N / 4 - 10) / static_cast<double>(nWins));
    for(int i = 0; i < (nWins - 1); i++)
    {
        winSizes[i] = 10 + i * winStep;
    }
    winSizes[nWins - 1] = N / 4;

    int *d_winSizes;
    cudaMalloc(&d_winSizes, nWins * sizeof(int));
    cudaMemcpy(d_winSizes, winSizes, nWins * sizeof(int), cudaMemcpyHostToDevice);

    double *flucVec_mfdfa;
    cudaMalloc(&flucVec_mfdfa, nWins * sizeof(double));

    // kernel parameters
    dim3 threadsPerBlock_mfdfa(nThreads);
    dim3 blocksPerGrid_mfdfa((nWins + nThreads - 1) / nThreads);
    dim3 threadsPerBlock(nThreads);

    cudaStream_t stream_1, stream_2;
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);

    // kernels
    MFDFAforHTKernel<<<blocksPerGrid_mfdfa, threadsPerBlock_mfdfa, 0, stream_1>>>(y, t, N, d_winSizes, nWins, flucVec_mfdfa);
    for(int i = 0; i < nScales; i++)
    {
        int Ns = N - scales[i] + 1;
        dim3 blocksPerGrid((Ns + nThreads - 1) / nThreads);
        HTKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_2>>>(y, t, N, scales[i], prevScales[i], Ns, d_ht);
    }
    cudaDeviceSynchronize();

    // log variables for fit
    double *d_logW_mfdfa, *d_logF_mfdfa;
    cudaMalloc(&d_logW_mfdfa, nWins * sizeof(double));
    cudaMalloc(&d_logF_mfdfa, nWins * sizeof(double));

    doubleToLog<<<blocksPerGrid_mfdfa, threadsPerBlock_mfdfa, 0, stream_1>>>(flucVec_mfdfa, d_logF_mfdfa, nWins);
    intToLog<<<blocksPerGrid_mfdfa, threadsPerBlock_mfdfa, 0, stream_2>>>(d_winSizes, d_logW_mfdfa, nWins);

    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);

    // mfdfa fit
    double *d_H_mfdfa, *d_I_mfdfa;
    cudaMalloc(&d_H_mfdfa, sizeof(double));
    cudaMalloc(&d_I_mfdfa, sizeof(double));
    hFit<<<1, 1>>>(nWins, d_logW_mfdfa, d_logF_mfdfa, d_H_mfdfa, d_I_mfdfa);
    cudaDeviceSynchronize();

    // ht
    for(int i = 0; i < nScales; i++)
    {
        double Ns = N - scales[i] + 1;
        dim3 blocksPerGrid((Ns + nThreads - 1) / nThreads);
        finalHTKernel<<<blocksPerGrid, threadsPerBlock>>>(d_ht, Ns, scales[i], prevScales[i], d_H_mfdfa, d_I_mfdfa);
    }

    // copy to host
    cudaMemcpy(ht, d_ht, sLen * sizeof(double), cudaMemcpyDeviceToHost);

    // free memory
    delete [] prevScales;

    cudaFree(d_winSizes);
    cudaFree(flucVec_mfdfa);
    cudaFree(d_logW_mfdfa);
    cudaFree(d_logF_mfdfa);
    cudaFree(d_H_mfdfa);
    cudaFree(d_I_mfdfa);
}

