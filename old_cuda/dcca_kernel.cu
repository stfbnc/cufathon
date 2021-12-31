#include "dcca_kernel.cuh"
#include <iostream>


__global__
void DCCAKernel(const float * __restrict__ y1, const float * __restrict__ y2,
                const float * __restrict__ t, int N,
                const int * __restrict__ winSizes, int nWins, float * __restrict__ flucVec)
{
    int nWin = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(nWin < nWins)
    {
        int currWinSize = winSizes[nWin];
        int Ns = N / currWinSize;
        float f = 0.0;
        
        for(int i = 0; i < Ns; i++)
        {
            int startLim = i * currWinSize;
            float m1 = 0.0, q1 = 0.0;
            float m2 = 0.0, q2 = 0.0;

            fit(currWinSize, t + startLim, y1 + startLim, &m1, &q1);
            fit(currWinSize, t + startLim, y2 + startLim, &m2, &q2);

            for(int j = 0; j < currWinSize; j++)
            {
                float var1 = y1[startLim + j] - (q1 + m1 * t[startLim + j]);
                float var2 = y2[startLim + j] - (q2 + m2 * t[startLim + j]);
                f += var1 * var2;
            }
        }

        flucVec[nWin] = f / (Ns * currWinSize);
    }
}

__global__
void DCCAabsKernel(const float * __restrict__ y1, const float * __restrict__ y2,
                const float * __restrict__ t, int N,
                const int * __restrict__ winSizes, int nWins, float * __restrict__ flucVec)
{
    int nWin = blockIdx.x * blockDim.x + threadIdx.x;

    if(nWin < nWins)
    {
        int currWinSize = winSizes[nWin];
        int Ns = N / currWinSize;
        float f = 0.0;

        for(int i = 0; i < Ns; i++)
        {
            int startLim = i * currWinSize;
            float m1 = 0.0, q1 = 0.0;
            float m2 = 0.0, q2 = 0.0;

            fit(currWinSize, t + startLim, y1 + startLim, &m1, &q1);
            fit(currWinSize, t + startLim, y2 + startLim, &m2, &q2);

            for(int j = 0; j < currWinSize; j++)
            {
                float var1 = y1[startLim + j] - (q1 + m1 * t[startLim + j]);
                float var2 = y2[startLim + j] - (q2 + m2 * t[startLim + j]);
                f += fabs(var1 * var2);
            }
        }

        flucVec[nWin] = sqrt(f / (Ns * currWinSize));
    }
}

__global__
void DCCAKernelBackwards(const float * __restrict__ y1, const float * __restrict__ y2,
                         const float * __restrict__ t, int N,
                         const int * __restrict__ winSizes, int nWins, float * __restrict__ flucVec)
{
    int nWin = blockIdx.x * blockDim.x + threadIdx.x;

    if(nWin < nWins)
    {
        int currWinSize = winSizes[nWin];
        int Ns = N / currWinSize;
        float f = 0.0;

        for(int i = 0; i < Ns; i++)
        {
            int startLim = i * currWinSize;
            float m1 = 0.0, q1 = 0.0;
            float m2 = 0.0, q2 = 0.0;

            fit(currWinSize, t + startLim, y1 + startLim, &m1, &q1);
            fit(currWinSize, t + startLim, y2 + startLim, &m2, &q2);

            for(int j = 0; j < currWinSize; j++)
            {
                float var1 = y1[startLim + j] - (q1 + m1 * t[startLim + j]);
                float var2 = y2[startLim + j] - (q2 + m2 * t[startLim + j]);
                f += var1 * var2;
            }

            startLim = i * currWinSize + (N - Ns * currWinSize);
            fit(currWinSize, t + startLim, y1 + startLim, &m1, &q1);
            fit(currWinSize, t + startLim, y2 + startLim, &m2, &q2);

            for(int j = 0; j < currWinSize; j++)
            {
                float var1 = y1[startLim + j] - (q1 + m1 * t[startLim + j]);
                float var2 = y2[startLim + j] - (q2 + m2 * t[startLim + j]);
                f += var1 * var2;
            }
        }

        flucVec[nWin] = f / (2.0f * Ns * currWinSize);
    }
}

__global__
void DCCAabsKernelBackwards(const float * __restrict__ y1, const float * __restrict__ y2,
                         const float * __restrict__ t, int N,
                         const int * __restrict__ winSizes, int nWins, float * __restrict__ flucVec)
{
    int nWin = blockIdx.x * blockDim.x + threadIdx.x;

    if(nWin < nWins)
    {
        int currWinSize = winSizes[nWin];
        int Ns = N / currWinSize;
        float f = 0.0;

        for(int i = 0; i < Ns; i++)
        {
            int startLim = i * currWinSize;
            float m1 = 0.0, q1 = 0.0;
            float m2 = 0.0, q2 = 0.0;

            fit(currWinSize, t + startLim, y1 + startLim, &m1, &q1);
            fit(currWinSize, t + startLim, y2 + startLim, &m2, &q2);

            for(int j = 0; j < currWinSize; j++)
            {
                float var1 = y1[startLim + j] - (q1 + m1 * t[startLim + j]);
                float var2 = y2[startLim + j] - (q2 + m2 * t[startLim + j]);
                f += fabs(var1 * var2);
            }

            startLim = i * currWinSize + (N - Ns * currWinSize);
            fit(currWinSize, t + startLim, y1 + startLim, &m1, &q1);
            fit(currWinSize, t + startLim, y2 + startLim, &m2, &q2);

            for(int j = 0; j < currWinSize; j++)
            {
                float var1 = y1[startLim + j] - (q1 + m1 * t[startLim + j]);
                float var2 = y2[startLim + j] - (q2 + m2 * t[startLim + j]);
                f += fabs(var1 * var2);
            }
        }

        flucVec[nWin] = sqrt(f / (2.0f * Ns * currWinSize));
    }
}

__global__
void rhoKernel(const float * __restrict__ fxx, const float * __restrict__ fyy,
               const float * __restrict__ fxy, int n, float * __restrict__ p)
{
    int nWin = blockIdx.x * blockDim.x + threadIdx.x;

    if(nWin < n)
    {
        p[nWin] = fxy[nWin] / (fxx[nWin] * fyy[nWin]);
    }
}

__host__
int index_of_percentile(int N, float percentile)
{
    int i = static_cast<int>(round(N * percentile));
    if(i == N)
    {
        i = N - 1;
    }

    return i;
}

void cudaDCCA(float *y1, float *y2, float *t, int N, int *winSizes, int nWins, bool revSeg, float *rho, int nThreads)
{
    // device variables
    float *d_Fxx;
    cudaMalloc(&d_Fxx, nWins * sizeof(float));

    float *d_Fyy;
    cudaMalloc(&d_Fyy, nWins * sizeof(float));

    float *d_Fxy;
    cudaMalloc(&d_Fxy, nWins * sizeof(float));

    int *d_winSizes;
    cudaMalloc(&d_winSizes, nWins * sizeof(int));

    float *d_rho;
    cudaMalloc(&d_rho, nWins * sizeof(float));

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
        DCCAabsKernelBackwards<<<blocksPerGrid, threadsPerBlock, 0, stream_1>>>(y1, y1, t, N, d_winSizes, nWins, d_Fxx);
        DCCAabsKernelBackwards<<<blocksPerGrid, threadsPerBlock, 0, stream_2>>>(y2, y2, t, N, d_winSizes, nWins, d_Fyy);
        DCCAKernelBackwards<<<blocksPerGrid, threadsPerBlock, 0, stream_3>>>(y1, y2, t, N, d_winSizes, nWins, d_Fxy);
    }
    else
    {
        DCCAabsKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_1>>>(y1, y1, t, N, d_winSizes, nWins, d_Fxx);
        DCCAabsKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_2>>>(y2, y2, t, N, d_winSizes, nWins, d_Fyy);
        DCCAKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_3>>>(y1, y2, t, N, d_winSizes, nWins, d_Fxy);
    }

    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);
    cudaStreamDestroy(stream_3);

    // rho kernel
    rhoKernel<<<blocksPerGrid, threadsPerBlock>>>(d_Fxx, d_Fyy, d_Fxy, nWins, d_rho);

    // copy to host
    cudaMemcpy(rho, d_rho, nWins * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_Fxx);
    cudaFree(d_Fyy);
    cudaFree(d_Fxy);
    cudaFree(d_winSizes);
    cudaFree(d_rho);
}

void cudaDCCAConfInt(int *winSizes, int nWins, int N, int nSim, float confLevel, float *confUp, float *confDown, int nThreads)
{
    // random numbers generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // device variables
    float *d_t;
    cudaMalloc(&d_t, N * sizeof(float));
    linRange(d_t, N, 1);
    float *d_rand;
    cudaMalloc(&d_rand, 2 * N * sizeof(float));

    int *d_winSizes;
    cudaMalloc(&d_winSizes, nWins * sizeof(int));
    cudaMemcpy(d_winSizes, winSizes, nWins * sizeof(int), cudaMemcpyHostToDevice);

    float *d_fxx, *d_fyy, *d_fxy;
    cudaMalloc(&d_fxx, nWins * sizeof(float));
    cudaMalloc(&d_fyy, nWins * sizeof(float));
    cudaMalloc(&d_fxy, nWins * sizeof(float));

    float *d_rho;
    cudaMalloc(&d_rho, nWins * nSim * sizeof(float));

    cudaStream_t stream_1, stream_2, stream_3;
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);
    cudaStreamCreate(&stream_3);

    dim3 threadsPerBlock(nThreads);
    dim3 blocksPerGrid((nWins + nThreads - 1) / nThreads);

    for(int i = 0; i < nSim; i++)
    {
        // generate random sequences
        curandGenerateNormal(gen, d_rand, 2 * N, 0.0f, 1.0f);

        DCCAabsKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_1>>>(d_rand, d_rand, d_t, N, d_winSizes, nWins, d_fxx);
        DCCAabsKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_2>>>(&d_rand[N], &d_rand[N], d_t, N, d_winSizes, nWins, d_fyy);
        DCCAKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_3>>>(d_rand, &d_rand[N], d_t, N, d_winSizes, nWins, d_fxy);

        cudaDeviceSynchronize();

        rhoKernel<<<blocksPerGrid, threadsPerBlock>>>(d_fxx, d_fyy, d_fxy, nWins, &d_rho[i * nWins]);
    }

    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);
    cudaStreamDestroy(stream_3);

    // copy to host
    float *rho = new float [nWins * nSim];
    cudaMemcpy(rho, d_rho, nWins * nSim * sizeof(float), cudaMemcpyDeviceToHost);

    float *by_win = new float [nSim];
    for(int i = 0; i < nWins; i++)
    {
        for(int j = 0; j < nSim; j++)
        {
            by_win[j] = rho[i + j * nWins];
        }
        std::sort(by_win, by_win + nSim);
        confUp[i] = by_win[index_of_percentile(nSim, confLevel)];
        confDown[i] = by_win[index_of_percentile(nSim, 1 - confLevel)];
    }

    // free memory
    curandDestroyGenerator(gen);
    cudaFree(d_t);
    cudaFree(d_rand);
    cudaFree(d_winSizes);
    cudaFree(d_fxx);
    cudaFree(d_fyy);
    cudaFree(d_fxy);
    cudaFree(d_rho);

    delete [] rho;
    delete [] by_win;
}

