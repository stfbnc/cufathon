#include "ht_kernel.cuh"

__global__
void MFDFAforHTKernel(const float * __restrict__ y, int n,
                      const int * __restrict__ win_sizes, int n_wins,
                      float * __restrict__ fluc_vec_mfdfa)
{
    int iw = blockIdx.x * blockDim.x + threadIdx.x;

    if(iw < n_wins)
    {
        int curr_win_size = win_sizes[iw];
        int n_s = n / curr_win_size;
        float f = 0.0f;

        for(int i = 0; i < n_s; i++)
        {
            float rms = 0.0f, rms2 = 0.0f;
            int start_lim = i * curr_win_size;
            float m = 0.0f, q = 0.0f;

            fit(curr_win_size, start_lim + 1, y + start_lim, &m, &q);

            for(int j = 0; j < curr_win_size; j++)
            {
                float var = y[start_lim + j] - (q + m * (start_lim + 1 + j));
                rms += pow(var, 2.0f);
            }

            start_lim = i * curr_win_size + (n - n_s * curr_win_size);
            fit(curr_win_size, start_lim + 1, y + start_lim, &m, &q);

            for(int j = 0; j < curr_win_size; j++)
            {
                float var = y[start_lim + j] - (q + m * (start_lim + 1 + j));
                rms2 += pow(var, 2.0f);
            }

            f += log(rms / curr_win_size) + log(rms2 / curr_win_size);
        }

        fluc_vec_mfdfa[iw] = exp(f / (4.0f * n_s));
    }
}

__global__
void HTKernel(const float * __restrict__ y,
              int scale, int prev_scale, int n_s,
              float * __restrict__ fluc_vec)
{   
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < n_s)
    {   
        float f = 0.0f; 
        float m = 0.0f, q = 0.0f;
            
        fit(scale, i + 1, y + i, &m, &q);
            
        for(int j = 0; j < scale; j++)
        {   
            float var = y[i + j] - (q + m * (i + 1 + j));
            f += pow(var, 2.0f);
        }
            
        fluc_vec[prev_scale + i] = sqrt(f / scale);
    }
}

__global__
void finalHTKernel(float * __restrict__ vecht, float n_s,
                   int scale, int prev_scale,
                   float *h0, float *h0_intercept)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_s)
    {
        float dscale = static_cast<float>(scale);
        vecht[prev_scale + i] = (*h0_intercept + *h0 * log(dscale) - log(vecht[prev_scale + i])) / (log(n_s) - log(dscale)) + *h0;
    }
}

void cudaHT(float *y, int n, int *scales, int n_scales, float *ht, int n_threads)
{
    // ht variables
    int *prev_scales = new int [n_scales];
    int s_len = 0;
    for(int i = 0; i < n_scales; i++)
    {
        s_len += (n - scales[i] + 1);
        prev_scales[i] = 0;
        for(int j = 0; j < i; j++)
        {
            prev_scales[i] += (n - scales[j] + 1);
        }
    }

    float *d_y;
    cudaMalloc(&d_y, n * sizeof(float));
    float *d_ht;
    cudaMalloc(&d_ht, s_len * sizeof(float));

    // mfdfa variables
    int n_wins = 20;
    int win_sizes[n_wins];
    int win_step = round((n / 4 - 10) / static_cast<float>(n_wins));
    for(int i = 0; i < (n_wins - 1); i++)
    {
        win_sizes[i] = 10 + i * win_step;
    }
    win_sizes[n_wins - 1] = n / 4;

    int *d_win_sizes;
    cudaMalloc(&d_win_sizes, n_wins * sizeof(int));
    cudaMemcpy(d_win_sizes, win_sizes, n_wins * sizeof(int), cudaMemcpyHostToDevice);

    float *fluc_vec_mfdfa;
    cudaMalloc(&fluc_vec_mfdfa, n_wins * sizeof(float));

    // kernel parameters
    dim3 threadsPerBlock_mfdfa(n_wins);
    dim3 blocksPerGrid_mfdfa(1);
    dim3 threadsPerBlock(n_threads);

    cudaStream_t stream_1, stream_2;
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);

    // kernels
    MFDFAforHTKernel<<<blocksPerGrid_mfdfa, threadsPerBlock_mfdfa, 0, stream_1>>>(y, n, d_win_sizes, n_wins, fluc_vec_mfdfa);
    for(int i = 0; i < n_scales; i++)
    {
        int n_s = n - scales[i] + 1;
        dim3 blocksPerGrid((n_s + n_threads - 1) / n_threads);
        HTKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_2>>>(y, scales[i], prev_scales[i], n_s, d_ht);
    }

    // log variables for fit
    /*float *d_logW_mfdfa, *d_logF_mfdfa;
    cudaMalloc(&d_logW_mfdfa, nWins * sizeof(float));
    cudaMalloc(&d_logF_mfdfa, nWins * sizeof(float));

    floatToLog<<<blocksPerGrid_mfdfa, threadsPerBlock_mfdfa, 0, stream_1>>>(flucVec_mfdfa, d_logF_mfdfa, nWins);
    intToLog<<<blocksPerGrid_mfdfa, threadsPerBlock_mfdfa, 0, stream_2>>>(d_winSizes, d_logW_mfdfa, nWins);

    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);*/

    // mfdfa fit
    /*float *d_H_mfdfa, *d_I_mfdfa;
    cudaMalloc(&d_H_mfdfa, sizeof(float));
    cudaMalloc(&d_I_mfdfa, sizeof(float));
    hFit<<<1, 1>>>(nWins, d_logW_mfdfa, d_logF_mfdfa, d_H_mfdfa, d_I_mfdfa);
    cudaDeviceSynchronize();*/

    // ht
    /*for(int i = 0; i < nScales; i++)
    {
        float Ns = N - scales[i] + 1;
        dim3 blocksPerGrid((Ns + nThreads - 1) / nThreads);
        finalHTKernel<<<blocksPerGrid, threadsPerBlock>>>(d_ht, Ns, scales[i], prevScales[i], d_H_mfdfa, d_I_mfdfa);
    }*/

    // copy to host
    //cudaMemcpy(ht, d_ht, sLen * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory
    delete [] prev_scales;

    cudaFree(d_y);
    cudaFree(d_win_sizes);
    cudaFree(fluc_vec_mfdfa);
    //cudaFree(d_logW_mfdfa);
    //cudaFree(d_logF_mfdfa);
    //cudaFree(d_H_mfdfa);
    //cudaFree(d_I_mfdfa);
    cudaFree(d_ht);
}

