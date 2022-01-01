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
void H0Kernel(float * __restrict__ logw,
              float * __restrict__ logf,
              const int * __restrict__ win_sizes,
              const float * __restrict__ fluc_vec,
              int n, float *h0, float *h0_intercept)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n)
    {
        logf[i] = log(fluc_vec[i]);
        logw[i] = log(static_cast<float>(win_sizes[i]));
    }

    __syncthreads();

    if(i == 0)
    {
        h_fit(n, logw, logf, h0, h0_intercept);
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
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float dscale;
    dscale = static_cast<float>(scale);
    __shared__ float h;
    h = *h0;
    __shared__ float i;
    i = *h0_intercept;

    __syncthreads();

    if(tx < n_s)
    {
        vecht[prev_scale + tx] = (i + h * log(dscale) - log(vecht[prev_scale + tx])) / (log(n_s) - log(dscale)) + h;
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
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
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

    float *d_fluc_vec_mfdfa;
    cudaMalloc(&d_fluc_vec_mfdfa, n_wins * sizeof(float));

    float *d_log_w, *d_log_f, *d_h, *d_i;
    cudaMalloc(&d_log_w, n_wins * sizeof(float));
    cudaMalloc(&d_log_f, n_wins * sizeof(float));
    cudaMalloc(&d_h, sizeof(float));
    cudaMalloc(&d_i, sizeof(float));

    // kernel parameters
    dim3 threadsPerBlock_mfdfa(n_wins);
    dim3 blocksPerGrid_mfdfa(1);
    dim3 threadsPerBlock(n_threads);

    int n_streams = 3;
    cudaStream_t streams[n_streams];
    for(int i = 0; i < n_streams; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    // kernels
    MFDFAforHTKernel<<<blocksPerGrid_mfdfa, threadsPerBlock_mfdfa, 0, streams[0]>>>(d_y, n, d_win_sizes, n_wins, d_fluc_vec_mfdfa);
    H0Kernel<<<blocksPerGrid_mfdfa, threadsPerBlock_mfdfa, 0, streams[0]>>>(d_log_w, d_log_f, d_win_sizes, d_fluc_vec_mfdfa, n_wins, d_h, d_i);
    for(int i = 0; i < n_scales; i++)
    {
        int n_s = n - scales[i] + 1;
        dim3 blocksPerGrid((n_s + n_threads - 1) / n_threads);
        HTKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[1]>>>(d_y, scales[i], prev_scales[i], n_s, d_ht);
    }    

    cudaDeviceSynchronize();

    // ht
    for(int i = 0; i < n_scales; i++)
    {
        float n_s = n - scales[i] + 1;
        dim3 blocksPerGrid((n_s + n_threads - 1) / n_threads);
        finalHTKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i % n_streams]>>>(d_ht, n_s, scales[i], prev_scales[i], d_h, d_i);
    }

    for(int i = 0; i < n_streams; i++)
    {
        cudaStreamDestroy(streams[i]);
    }

    // copy to host
    cudaMemcpy(ht, d_ht, s_len * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory
    delete [] prev_scales;

    cudaFree(d_y);
    cudaFree(d_win_sizes);
    cudaFree(d_fluc_vec_mfdfa);
    cudaFree(d_log_w);
    cudaFree(d_log_f);
    cudaFree(d_h);
    cudaFree(d_i);
    cudaFree(d_ht);
}

