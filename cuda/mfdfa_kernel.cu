#include <stdio.h>
#include "mfdfa_kernel.cuh"


__global__
void MFDFAKernel(const double * __restrict__ y, const double * __restrict__ t, int N,
                 const int * __restrict__ winSizes, int nWins,
                 const double * __restrict__ qVals, int nq,
                 double * __restrict__ flucVec)
{
    int iq = blockIdx.x * blockDim.x + threadIdx.x;
    int iw = blockIdx.y * blockDim.y + threadIdx.y;

    if((iq < nq) && (iw < nWins))
    {
        double qVal = qVals[iq];
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

            if(qVal == 0.0)
            {
                f += log(rms / currWinSize);
            }
            else
            {
                f += pow(rms / currWinSize, 0.5 * qVal);
            }
        }
        
        if(qVal == 0.0)
        {
            flucVec[iq * nWins + iw] = exp(f / (2.0 * Ns));
        }
        else
        {
            flucVec[iq * nWins + iw] = pow(f / Ns, 1.0 / qVal);
        }
    }
}

__global__
void MFDFAKernelBackwards(const double * __restrict__ y, const double * __restrict__ t, int N,
                          const int * __restrict__ winSizes, int nWins,
                          const double * __restrict__ qVals, int nq,
                          double * __restrict__ flucVec)
{
    int iq = blockIdx.x * blockDim.x + threadIdx.x;
    int iw = blockIdx.y * blockDim.y + threadIdx.y;

    if((iq < nq) && (iw < nWins))
    {
        double qVal = qVals[iq];
        int currWinSize = winSizes[iw];
        int Ns = N / currWinSize;
        double f = 0.0;

        for(int i = 0; i < Ns; i++)
        {
            double rms1 = 0.0, rms2 = 0.0;
            int startLim = i * currWinSize;
            double m = 0.0, q = 0.0;

            fit(currWinSize, t + startLim, y + startLim, &m, &q);

            for(int j = 0; j < currWinSize; j++)
            {   
                double var = y[startLim + j] - (q + m * t[startLim + j]);
                rms1 += pow(var, 2.0);
            }

            startLim = i * currWinSize + (N - Ns * currWinSize);
            fit(currWinSize, t + startLim, y + startLim, &m, &q);
 
            for(int j = 0; j < currWinSize; j++)
            {
                double var = y[startLim + j] - (q + m * t[startLim + j]);
                rms2 += pow(var, 2.0);
            }

            if(qVal == 0.0)
            {   
                f += log(rms1 / currWinSize) + log(rms2 / currWinSize);
            }
            else
            {   
                f += pow(rms1 / currWinSize, 0.5 * qVal) + pow(rms2 / currWinSize, 0.5 * qVal);
            }
        }

        if(qVal == 0.0)
        {   
            flucVec[iq * nWins + iw] = exp(f / (4.0 * Ns));
        }
        else
        {   
            flucVec[iq * nWins + iw] = pow(f / (2.0 * Ns), 1.0 / qVal);
        }
    }
}

__global__
void hqKernel(const double * __restrict__ y, const double * __restrict__ x, int n,
              double * __restrict__ hq, int nq)
{
    int iq = blockIdx.x * blockDim.x + threadIdx.x;

    if(iq < nq)
    {
        double dummy = 0.0;
        fit(n, x, y + n * iq, &hq[iq], &dummy);
    }
}

void cudaMFDFA(double *y, double *t, int N, int *winSizes, int nWins, double *qVals, int nq, bool revSeg, double *hq, int nThreads)
{
    // device variables
    double *d_F;
    cudaMalloc(&d_F, nWins * nq * sizeof(double));

    int *d_winSizes;
    cudaMalloc(&d_winSizes, nWins * sizeof(int));

    double *d_qVals;
    cudaMalloc(&d_qVals, nq * sizeof(double));

    double *d_hq;
    cudaMalloc(&d_hq, nq * sizeof(double));

    // copy to device
    cudaMemcpy(d_winSizes, winSizes, nWins * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qVals, qVals, nq * sizeof(double), cudaMemcpyHostToDevice);

    // mfdfa kernel
    dim3 threadsPerBlock(nThreads, nThreads);
    dim3 blocksPerGrid((nq + nThreads - 1) / nThreads, (nWins + nThreads - 1) / nThreads);
    if(revSeg)
    {
        MFDFAKernelBackwards<<<blocksPerGrid, threadsPerBlock>>>(y, t, N, d_winSizes, nWins, d_qVals, nq, d_F);
    }
    else
    {
        MFDFAKernel<<<blocksPerGrid, threadsPerBlock>>>(y, t, N, d_winSizes, nWins, d_qVals, nq, d_F);
    }

    // device variables
    double *d_logW, *d_logF;
    cudaMalloc(&d_logW, nWins * sizeof(double));
    cudaMalloc(&d_logF, nWins * nq * sizeof(double));

    // log transforms
    dim3 threadsPerBlock_log(nThreads * nThreads / 2);
    dim3 blocksPerGrid_logF((nq * nWins + nThreads * nThreads / 2 - 1) / (nThreads * nThreads / 2));
    dim3 blocksPerGrid_logW((nWins + nThreads * nThreads / 2 - 1) / (nThreads * nThreads / 2));

    cudaStream_t stream_1, stream_2;
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);

    doubleToLog<<<blocksPerGrid_logF, threadsPerBlock_log, 0, stream_1>>>(d_F, d_logF, nWins * nq);
    intToLog<<<blocksPerGrid_logW, threadsPerBlock_log, 0, stream_2>>>(d_winSizes, d_logW, nWins);

    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);

    // hq
    dim3 threadsPerBlock_hq(nThreads * nThreads / 2);
    dim3 blocksPerGrid_hq((nq + nThreads * nThreads / 2 - 1) / (nThreads * nThreads / 2));
    hqKernel<<<blocksPerGrid_hq, threadsPerBlock_hq>>>(d_logF, d_logW, nWins, d_hq, nq);

    // copy to host
    cudaMemcpy(hq, d_hq, nq * sizeof(double), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_F);
    cudaFree(d_winSizes);
    cudaFree(d_qVals);
    cudaFree(d_hq);
    cudaFree(d_logW);
    cudaFree(d_logF);
}
