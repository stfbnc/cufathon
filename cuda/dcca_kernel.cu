#include "dcca_kernel.cuh"


__global__
void DCCAKernel(const double * __restrict__ y1, const double * __restrict__ y2,
                const double * __restrict__ t, int N,
                const int * __restrict__ winSizes, int nWins, double * __restrict__ flucVec)
{
    int nWin = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(nWin < nWins)
    {
        int currWinSize = winSizes[nWin];
        int Ns = N / currWinSize;
        double f = 0.0;
        
        for(int i = 0; i < Ns; i++)
        {
            int startLim = i * currWinSize;
            double m1 = 0.0, q1 = 0.0;
            double m2 = 0.0, q2 = 0.0;

            fit(currWinSize, t + startLim, y1 + startLim, &m1, &q1);
            fit(currWinSize, t + startLim, y2 + startLim, &m2, &q2);

            for(int j = 0; j < currWinSize; j++)
            {
                double var1 = y1[startLim + j] - (q1 + m1 * t[startLim + j]);
                double var2 = y2[startLim + j] - (q2 + m2 * t[startLim + j]);
                f += var1 * var2;
            }
        }

        flucVec[nWin] = f / (Ns * currWinSize);
    }
}

__global__
void DCCAKernelBackwards(const double * __restrict__ y1, const double * __restrict__ y2,
                         const double * __restrict__ t, int N,
                         const int * __restrict__ winSizes, int nWins, double * __restrict__ flucVec)
{
    int nWin = blockIdx.x * blockDim.x + threadIdx.x;

    if(nWin < nWins)
    {
        int currWinSize = winSizes[nWin];
        int Ns = N / currWinSize;
        double f = 0.0;

        for(int i = 0; i < Ns; i++)
        {
            int startLim = i * currWinSize;
            double m1 = 0.0, q1 = 0.0;
            double m2 = 0.0, q2 = 0.0;

            fit(currWinSize, t + startLim, y1 + startLim, &m1, &q1);
            fit(currWinSize, t + startLim, y2 + startLim, &m2, &q2);

            for(int j = 0; j < currWinSize; j++)
            {
                double var1 = y1[startLim + j] - (q1 + m1 * t[startLim + j]);
                double var2 = y2[startLim + j] - (q2 + m2 * t[startLim + j]);
                f += var1 * var2;
            }

            startLim = i * currWinSize + (N - Ns * currWinSize);
            fit(currWinSize, t + startLim, y1 + startLim, &m1, &q1);
            fit(currWinSize, t + startLim, y2 + startLim, &m2, &q2);

            for(int j = 0; j < currWinSize; j++)
            {
                double var1 = y1[startLim + j] - (q1 + m1 * t[startLim + j]);
                double var2 = y2[startLim + j] - (q2 + m2 * t[startLim + j]);
                f += var1 * var2;
            }
        }

        flucVec[nWin] = f / (2.0 * Ns * currWinSize);
    }
}

// Aggiungere un kernel per dfa con alla fine sqrt per fxx e fyy
__global__
void rhoKernel(const double * __restrict__ fxx, const double * __restrict__ fyy,
               const double * __restrict__ fxy, int n, double * __restrict__ p)
{
    int nWin = blockIdx.x * blockDim.x + threadIdx.x;

    if(nWin < n)
    {
        p[nWin] = fxy[nWin] / (fxx[nWin] * fyy[nWin]);
    }
}

void cudaDCCA(double *y1, double *y2, double *t, int N, int *winSizes, int nWins, bool revSeg, double *rho, int nThreads)
{
    // device variables
    double *d_Fxx;
    cudaMalloc(&d_Fxx, nWins * sizeof(double));

    double *d_Fyy;
    cudaMalloc(&d_Fyy, nWins * sizeof(double));

    double *d_Fxy;
    cudaMalloc(&d_Fxy, nWins * sizeof(double));

    int *d_winSizes;
    cudaMalloc(&d_winSizes, nWins * sizeof(int));

    double *d_rho;
    cudaMalloc(&d_rho, nWins * sizeof(double));

    // copy to device
    cudaMemcpy(d_winSizes, winSizes, nWins * sizeof(int), cudaMemcpyHostToDevice);

    // dcca kernel
    cudaStream_t stream_1, stream_2, stream_3;
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);
    cudaStreamCreate(&stream_3);

    dim3 threadsPerBlock(nThreads);
    dim3 blocksPerGrid((nWins + nThreads - 1) / nThreads);
    if(revSeg)
    {
        DCCAKernelBackwards<<<blocksPerGrid, threadsPerBlock, 0, stream_1>>>(y1, y1, t, N, d_winSizes, nWins, d_Fxx);
        DCCAKernelBackwards<<<blocksPerGrid, threadsPerBlock, 0, stream_2>>>(y2, y2, t, N, d_winSizes, nWins, d_Fyy);
        DCCAKernelBackwards<<<blocksPerGrid, threadsPerBlock, 0, stream_3>>>(y1, y2, t, N, d_winSizes, nWins, d_Fxy);
    }
    else
    {
        DCCAKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_1>>>(y1, y1, t, N, d_winSizes, nWins, d_Fxx);
        DCCAKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_2>>>(y2, y2, t, N, d_winSizes, nWins, d_Fyy);
        DCCAKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_3>>>(y1, y2, t, N, d_winSizes, nWins, d_Fxy);
    }

    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);
    cudaStreamDestroy(stream_3);

    // rho kernel
    rhoKernel<<<blocksPerGrid, threadsPerBlock>>>(d_Fxx, d_Fyy, d_Fxy, nWins, d_rho);

    // copy to host
    cudaMemcpy(rho, d_rho, nWins * sizeof(double), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_Fxx);
    cudaFree(d_Fyy);
    cudaFree(d_Fxy);
    cudaFree(d_winSizes);
    cudaFree(d_rho);
}

