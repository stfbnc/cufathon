#include "dfa_kernel.cuh"


__global__
void DFAKernel(const double * __restrict__ y, const double * __restrict__ t, int N,
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
            double m = 0.0, q = 0.0;

            fit(currWinSize, t + startLim, y + startLim, &m, &q);

            for(int j = 0; j < currWinSize; j++)
            {
                double var = y[startLim + j] - (q + m * t[startLim + j]);
                f += pow(var, 2.0);
            }
        }

        flucVec[nWin] = sqrt(f / (Ns * currWinSize));
    }
}

__global__
void DFAKernelBackwards(const double * __restrict__ y, const double * __restrict__ t, int N,
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
            double m = 0.0, q = 0.0;

            fit(currWinSize, t + startLim, y + startLim, &m, &q);

            for(int j = 0; j < currWinSize; j++)
            {
                double var = y[startLim + j] - (q + m * t[startLim + j]);
                f += pow(var, 2.0);
            }

            startLim = i * currWinSize + (N - Ns * currWinSize);
            fit(currWinSize, t + startLim, y + startLim, &m, &q);

            for(int j = 0; j < currWinSize; j++)
            {
                double var = y[startLim + j] - (q + m * t[startLim + j]);
                f += pow(var, 2.0);
            }
        }

        flucVec[nWin] = sqrt(f / (2.0 * Ns * currWinSize));
    }
}

void cudaDFA(double *y, double *t, int N, int *winSizes, int nWins, bool revSeg, double *flucVec, double *I, double *H, int nThreads)
{
    // device variables
    double *d_flucVec;
    cudaMalloc(&d_flucVec, nWins * sizeof(double));

    int *d_winSizes;
    cudaMalloc(&d_winSizes, nWins * sizeof(int));

    // copy to device
    cudaMemcpy(d_winSizes, winSizes, nWins * sizeof(int), cudaMemcpyHostToDevice);

    // dfa kernel
    dim3 threadsPerBlock(nThreads);
    dim3 blocksPerGrid((nWins + nThreads - 1) / nThreads);
    if(revSeg)
    {
        DFAKernelBackwards<<<blocksPerGrid, threadsPerBlock>>>(y, t, N, d_winSizes, nWins, d_flucVec);
    }
    else
    {
        DFAKernel<<<blocksPerGrid, threadsPerBlock>>>(y, t, N, d_winSizes, nWins, d_flucVec);
    }

    // device variables
    double *d_logW, *d_logF;
    cudaMalloc(&d_logW, nWins * sizeof(double));
    cudaMalloc(&d_logF, nWins * sizeof(double));
    
    // log transforms
    cudaStream_t stream_1, stream_2;
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);
   
    doubleToLog<<<blocksPerGrid, threadsPerBlock, 0, stream_1>>>(d_flucVec, d_logF, nWins);
    intToLog<<<blocksPerGrid, threadsPerBlock, 0, stream_2>>>(d_winSizes, d_logW, nWins);
   
    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);

    // device variables
    double *d_H, *d_I;
    cudaMalloc(&d_H, sizeof(double));
    cudaMalloc(&d_I, sizeof(double));
   
    // fit kernel
    hFit<<<1, 1>>>(nWins, d_logW, d_logF, d_H, d_I);

    // copy to host
    cudaMemcpy(flucVec, d_flucVec, nWins * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(I, d_I, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(H, d_H, sizeof(double), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_flucVec);
    cudaFree(d_winSizes);
    cudaFree(d_logW);
    cudaFree(d_logF);
    cudaFree(d_H);
    cudaFree(d_I);
}

__global__
void DFAKernelInner(const double * __restrict__ y, const double * __restrict__ t, int N,
                    const int * __restrict__ winSizes, int nWins, double * __restrict__ flucVec)
{
    int nWin = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double sh[];
    double *s_y = &sh[0];
    double *s_t = &sh[N];

    if(nWin == 0)
    {
        for(int i = 0; i < N; i++)
        {
            s_y[i] = y[i];
            s_t[i] = t[i];
        }
    }

    __syncthreads();

    if(nWin < nWins)
    {   
        int currWinSize = winSizes[nWin];
        int Ns = N / currWinSize;
        double f = 0.0;
        
        for(int i = 0; i < Ns; i++)
        {   
            int startLim = i * currWinSize;
            double m = 0.0, q = 0.0;
            
            fit(currWinSize, s_t + startLim, s_y + startLim, &m, &q);
            
            for(int j = 0; j < currWinSize; j++)
            {   
                double var = s_y[startLim + j] - (q + m * s_t[startLim + j]);
                f += pow(var, 2.0);
            }
        }
        
        flucVec[nWin] = sqrt(f / (Ns * currWinSize));
    }
}

