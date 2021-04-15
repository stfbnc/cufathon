#include <stdio.h>
#include "mfdfa_kernel.cuh"


__global__
void MFDFAKernel(const double * __restrict__ y, const double * __restrict__ t, int N,
                 const int * __restrict__ winSizes, int nWins,
                 const double * __restrict__ qVals, int iq,
                 double * __restrict__ flucVec)
{
    int nWin = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(nWin < nWins)
    {
        double qVal = qVals[iq];
        int currWinSize = winSizes[nWin];
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
            flucVec[iq * nWins + nWin] = exp(f / (2.0 * Ns));
        }
        else
        {
            flucVec[iq * nWins + nWin] = pow(f / Ns, 1.0 / qVal);
        }
    }
}

void cudaMFDFA(double *y, double *t, int N, int *winSizes, int nWins, double *qVals, int nq, double *flucVec, int nThreads)
{
    dim3 threadsPerBlock(nThreads);
    dim3 blocksPerGrid((nWins + nThreads - 1) / nThreads);
    for(int iq = 0; iq < nq; iq++)
    {
        MFDFAKernel<<<blocksPerGrid, threadsPerBlock>>>(y, t, N, winSizes, nWins, qVals, iq, flucVec);
    }
    cudaDeviceSynchronize();
}


__global__
void MFDFAKernel2D(const double * __restrict__ y, const double * __restrict__ t, int N,
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

void cudaMFDFA2D(double *y, double *t, int N, int *winSizes, int nWins, double *qVals, int nq, double *flucVec, int nThreads)
{
    dim3 threadsPerBlock(nThreads, nThreads);
    dim3 blocksPerGrid((nq + nThreads - 1) / nThreads, (nWins + nThreads - 1) / nThreads);
    MFDFAKernel2D<<<blocksPerGrid, threadsPerBlock>>>(y, t, N, winSizes, nWins, qVals, nq, flucVec);
    cudaDeviceSynchronize();
}
