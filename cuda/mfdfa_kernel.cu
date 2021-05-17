#include <stdio.h>
#include "mfdfa_kernel.cuh"


__global__
void MFDFAKernel(const float * __restrict__ y, const float * __restrict__ t, int N,
                 const int * __restrict__ winSizes, int nWins,
                 const float * __restrict__ qVals, int nq,
                 float * __restrict__ flucVec)
{
    int iq = blockIdx.x * blockDim.x + threadIdx.x;
    int iw = blockIdx.y * blockDim.y + threadIdx.y;

    if((iq < nq) && (iw < nWins))
    {
        float qVal = qVals[iq];
        int currWinSize = winSizes[iw];
        int Ns = N / currWinSize;
        float f = 0.0;
        
        for(int i = 0; i < Ns; i++)
        {
            float rms = 0.0;
            int startLim = i * currWinSize;
            float m = 0.0, q = 0.0;
            
            fit(currWinSize, t + startLim, y + startLim, &m, &q);
            
            for(int j = 0; j < currWinSize; j++)
            {   
                float var = y[startLim + j] - (q + m * t[startLim + j]);
                rms += pow(var, 2.0f);
            }

            if(qVal == 0.0f)
            {
                f += log(rms / currWinSize);
            }
            else
            {
                f += pow(rms / currWinSize, 0.5f * qVal);
            }
        }
        
        if(qVal == 0.0f)
        {
            flucVec[iq * nWins + iw] = exp(f / (2.0f * Ns));
        }
        else
        {
            flucVec[iq * nWins + iw] = pow(f / Ns, 1.0f / qVal);
        }
    }
}

__global__
void MFDFAKernelBackwards(const float * __restrict__ y, const float * __restrict__ t, int N,
                          const int * __restrict__ winSizes, int nWins,
                          const float * __restrict__ qVals, int nq,
                          float * __restrict__ flucVec)
{
    int iq = blockIdx.x * blockDim.x + threadIdx.x;
    int iw = blockIdx.y * blockDim.y + threadIdx.y;

    if((iq < nq) && (iw < nWins))
    {
        float qVal = qVals[iq];
        int currWinSize = winSizes[iw];
        int Ns = N / currWinSize;
        float f = 0.0;

        for(int i = 0; i < Ns; i++)
        {
            float rms1 = 0.0, rms2 = 0.0;
            int startLim = i * currWinSize;
            float m = 0.0, q = 0.0;

            fit(currWinSize, t + startLim, y + startLim, &m, &q);

            for(int j = 0; j < currWinSize; j++)
            {   
                float var = y[startLim + j] - (q + m * t[startLim + j]);
                rms1 += pow(var, 2.0f);
            }

            startLim = i * currWinSize + (N - Ns * currWinSize);
            fit(currWinSize, t + startLim, y + startLim, &m, &q);
 
            for(int j = 0; j < currWinSize; j++)
            {
                float var = y[startLim + j] - (q + m * t[startLim + j]);
                rms2 += pow(var, 2.0f);
            }

            if(qVal == 0.0f)
            {   
                f += log(rms1 / currWinSize) + log(rms2 / currWinSize);
            }
            else
            {   
                f += pow(rms1 / currWinSize, 0.5f * qVal) + pow(rms2 / currWinSize, 0.5f * qVal);
            }
        }

        if(qVal == 0.0f)
        {   
            flucVec[iq * nWins + iw] = exp(f / (4.0f * Ns));
        }
        else
        {   
            flucVec[iq * nWins + iw] = pow(f / (2.0f * Ns), 1.0f / qVal);
        }
    }
}

__global__
void hqKernel(const float * __restrict__ y, const float * __restrict__ x, int n,
              float * __restrict__ hq, int nq)
{
    int iq = blockIdx.x * blockDim.x + threadIdx.x;

    if(iq < nq)
    {
        float dummy = 0.0;
        fit(n, x, y + n * iq, &hq[iq], &dummy);
    }
}

__global__
void faKernel(const float * __restrict__ q, const float * __restrict__ h, int nq,
              float * __restrict__ a, float * __restrict__ fa)
{
    int iq = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(iq < (nq - 1))
    {
        float dq = q[1] - q[0];
        a[iq] = (h[iq + 1] * q[iq + 1] - h[iq] * q[iq]) / dq;
        fa[iq] = q[iq] * (a[iq] - h[iq]) + 1.0f;
    }
}

void cudaMFDFA(float *y, float *t, int N, int *winSizes, int nWins, float *qVals, int nq, bool revSeg, float *hq, int nThreads)
{
    // device variables
    float *d_F;
    cudaMalloc(&d_F, nWins * nq * sizeof(float));

    int *d_winSizes;
    cudaMalloc(&d_winSizes, nWins * sizeof(int));

    float *d_qVals;
    cudaMalloc(&d_qVals, nq * sizeof(float));

    float *d_hq;
    cudaMalloc(&d_hq, nq * sizeof(float));

    // copy to device
    cudaMemcpy(d_winSizes, winSizes, nWins * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qVals, qVals, nq * sizeof(float), cudaMemcpyHostToDevice);

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
    float *d_logW, *d_logF;
    cudaMalloc(&d_logW, nWins * sizeof(float));
    cudaMalloc(&d_logF, nWins * nq * sizeof(float));

    // log transforms
    dim3 threadsPerBlock_log(nThreads * nThreads / 2);
    dim3 blocksPerGrid_logF((nq * nWins + nThreads * nThreads / 2 - 1) / (nThreads * nThreads / 2));
    dim3 blocksPerGrid_logW((nWins + nThreads * nThreads / 2 - 1) / (nThreads * nThreads / 2));

    cudaStream_t stream_1, stream_2;
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);

    floatToLog<<<blocksPerGrid_logF, threadsPerBlock_log, 0, stream_1>>>(d_F, d_logF, nWins * nq);
    intToLog<<<blocksPerGrid_logW, threadsPerBlock_log, 0, stream_2>>>(d_winSizes, d_logW, nWins);

    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);

    // hq
    dim3 threadsPerBlock_hq(nThreads * nThreads / 2);
    dim3 blocksPerGrid_hq((nq + nThreads * nThreads / 2 - 1) / (nThreads * nThreads / 2));
    hqKernel<<<blocksPerGrid_hq, threadsPerBlock_hq>>>(d_logF, d_logW, nWins, d_hq, nq);

    // copy to host
    cudaMemcpy(hq, d_hq, nq * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_F);
    cudaFree(d_winSizes);
    cudaFree(d_qVals);
    cudaFree(d_hq);
    cudaFree(d_logW);
    cudaFree(d_logF);
}

void cudaMultifractalSpectrum(float *y, float *t, int N, int *winSizes, int nWins, float *qVals, int nq, bool revSeg, float *a, float *fa, int nThreads)
{
    // device variables
    float *d_F;
    cudaMalloc(&d_F, nWins * nq * sizeof(float));

    int *d_winSizes;
    cudaMalloc(&d_winSizes, nWins * sizeof(int));

    float *d_qVals;
    cudaMalloc(&d_qVals, nq * sizeof(float));

    float *d_hq;
    cudaMalloc(&d_hq, nq * sizeof(float));

    // copy to device
    cudaMemcpy(d_winSizes, winSizes, nWins * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qVals, qVals, nq * sizeof(float), cudaMemcpyHostToDevice);

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
    float *d_logW, *d_logF;
    cudaMalloc(&d_logW, nWins * sizeof(float));
    cudaMalloc(&d_logF, nWins * nq * sizeof(float));

    // log transforms
    dim3 threadsPerBlock_log(nThreads * nThreads / 2);
    dim3 blocksPerGrid_logF((nq * nWins + nThreads * nThreads / 2 - 1) / (nThreads * nThreads / 2));
    dim3 blocksPerGrid_logW((nWins + nThreads * nThreads / 2 - 1) / (nThreads * nThreads / 2));

    cudaStream_t stream_1, stream_2;
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);

    floatToLog<<<blocksPerGrid_logF, threadsPerBlock_log, 0, stream_1>>>(d_F, d_logF, nWins * nq);
    intToLog<<<blocksPerGrid_logW, threadsPerBlock_log, 0, stream_2>>>(d_winSizes, d_logW, nWins);

    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);

    // hq
    dim3 threadsPerBlock_hq(nThreads * nThreads / 2);
    dim3 blocksPerGrid_hq((nq + nThreads * nThreads / 2 - 1) / (nThreads * nThreads / 2));
    hqKernel<<<blocksPerGrid_hq, threadsPerBlock_hq>>>(d_logF, d_logW, nWins, d_hq, nq);

    // device variables
    float *d_a, *d_fa;
    cudaMalloc(&d_a, (nq - 1) * sizeof(float));
    cudaMalloc(&d_fa, (nq - 1) * sizeof(float));

    // multifractal spectrum
    dim3 threadsPerBlock_fa(nThreads * nThreads / 2);
    dim3 blocksPerGrid_fa((nq - 1 + nThreads * nThreads / 2 - 1) / (nThreads * nThreads / 2));
    faKernel<<<blocksPerGrid_fa, threadsPerBlock_fa>>>(d_qVals, d_hq, nq, d_a, d_fa);

    // copy to host
    cudaMemcpy(a, d_a, (nq - 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fa, d_fa, (nq - 1) * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_F);
    cudaFree(d_winSizes);
    cudaFree(d_qVals);
    cudaFree(d_hq);
    cudaFree(d_logW);
    cudaFree(d_logF);
    cudaFree(d_a);
    cudaFree(d_fa);
}

