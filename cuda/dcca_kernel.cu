#include "dcca_kernel.cuh"

__global__
void DCCAKernel(const float * __restrict__ y1, const float * __restrict__ y2, int n,
                const int * __restrict__ win_sizes, int n_wins, float * __restrict__ fluc_vec)
{
    int n_win = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(n_win < n_wins)
    {
        int curr_win_size = win_sizes[n_win];
        int n_s = n / curr_win_size;
        float f = 0.0f;
        
        for(int i = 0; i < n_s; i++)
        {
            int start_lim = i * curr_win_size;
            float m1 = 0.0f, q1 = 0.0f;
            float m2 = 0.0f, q2 = 0.0f;

            fit(curr_win_size, start_lim + 1, y1 + start_lim, &m1, &q1);
            fit(curr_win_size, start_lim + 1, y2 + start_lim, &m2, &q2);

            for(int j = 0; j < curr_win_size; j++)
            {
                float var1 = y1[start_lim + j] - (q1 + m1 * (start_lim + 1 + j));
                float var2 = y2[start_lim + j] - (q2 + m2 * (start_lim + 1 + j));
                f += var1 * var2;
            }
        }

        fluc_vec[n_win] = f / (n_s * curr_win_size);
    }
}

__global__
void DCCAabsKernel(const float * __restrict__ y1, const float * __restrict__ y2, int n,
                   const int * __restrict__ win_sizes, int n_wins, float * __restrict__ fluc_vec)
{
    int n_win = blockIdx.x * blockDim.x + threadIdx.x;

    if(n_win < n_wins)
    {
        int curr_win_size = win_sizes[n_win];
        int n_s = n / curr_win_size;
        float f = 0.0f;

        for(int i = 0; i < n_s; i++)
        {
            int start_lim = i * curr_win_size;
            float m1 = 0.0f, q1 = 0.0f;
            float m2 = 0.0f, q2 = 0.0f;

            fit(curr_win_size, start_lim + 1, y1 + start_lim, &m1, &q1);
            fit(curr_win_size, start_lim + 1, y2 + start_lim, &m2, &q2);

            for(int j = 0; j < curr_win_size; j++)
            {
                float var1 = y1[start_lim + j] - (q1 + m1 * (start_lim + 1 + j));
                float var2 = y2[start_lim + j] - (q2 + m2 * (start_lim + 1 + j));
                f += fabs(var1 * var2);
            }
        }

        fluc_vec[n_win] = sqrt(f / (n_s * curr_win_size));
    }
}

__global__
void DCCAKernelBackwards(const float * __restrict__ y1, const float * __restrict__ y2, int n,
                         const int * __restrict__ win_sizes, int n_wins, float * __restrict__ fluc_vec)
{
    int n_win = blockIdx.x * blockDim.x + threadIdx.x;

    if(n_win < n_wins)
    {
        int curr_win_size = win_sizes[n_win];
        int n_s = n / curr_win_size;
        float f = 0.0f;

        for(int i = 0; i < n_s; i++)
        {
            int start_lim = i * curr_win_size;
            float m1 = 0.0f, q1 = 0.0f;
            float m2 = 0.0f, q2 = 0.0f;

            fit(curr_win_size, start_lim + 1, y1 + start_lim, &m1, &q1);
            fit(curr_win_size, start_lim + 1, y2 + start_lim, &m2, &q2);

            for(int j = 0; j < curr_win_size; j++)
            {
                float var1 = y1[start_lim + j] - (q1 + m1 * (start_lim + 1 + j));
                float var2 = y2[start_lim + j] - (q2 + m2 * (start_lim + 1 + j));
                f += var1 * var2;
            }

            start_lim = i * curr_win_size + (n - n_s * curr_win_size);
            fit(curr_win_size, start_lim + 1, y1 + start_lim, &m1, &q1);
            fit(curr_win_size, start_lim + 1, y2 + start_lim, &m2, &q2);

            for(int j = 0; j < curr_win_size; j++)
            {
                float var1 = y1[start_lim + j] - (q1 + m1 * (start_lim + 1 + j));
                float var2 = y2[start_lim + j] - (q2 + m2 * (start_lim + 1 + j));
                f += var1 * var2;
            }
        }

        fluc_vec[n_win] = f / (2.0f * n_s * curr_win_size);
    }
}

__global__
void DCCAabsKernelBackwards(const float * __restrict__ y1, const float * __restrict__ y2, int n,
                            const int * __restrict__ win_sizes, int n_wins, float * __restrict__ fluc_vec)
{
    int n_win = blockIdx.x * blockDim.x + threadIdx.x;

    if(n_win < n_wins)
    {
        int curr_win_size = win_sizes[n_win];
        int n_s = n / curr_win_size;
        float f = 0.0f;

        for(int i = 0; i < n_s; i++)
        {
            int start_lim = i * curr_win_size;
            float m1 = 0.0f, q1 = 0.0f;
            float m2 = 0.0f, q2 = 0.0f;

            fit(curr_win_size, start_lim + 1, y1 + start_lim, &m1, &q1);
            fit(curr_win_size, start_lim + 1, y2 + start_lim, &m2, &q2);

            for(int j = 0; j < curr_win_size; j++)
            {
                float var1 = y1[start_lim + j] - (q1 + m1 * (start_lim + 1 + j));
                float var2 = y2[start_lim + j] - (q2 + m2 * (start_lim + 1 + j));
                f += fabs(var1 * var2);
            }

            start_lim = i * curr_win_size + (n - n_s * curr_win_size);
            fit(curr_win_size, start_lim + 1, y1 + start_lim, &m1, &q1);
            fit(curr_win_size, start_lim + 1, y2 + start_lim, &m2, &q2);

            for(int j = 0; j < curr_win_size; j++)
            {
                float var1 = y1[start_lim + j] - (q1 + m1 * (start_lim + 1 + j));
                float var2 = y2[start_lim + j] - (q2 + m2 * (start_lim + 1 + j));
                f += fabs(var1 * var2);
            }
        }

        fluc_vec[n_win] = sqrt(f / (2.0f * n_s * curr_win_size));
    }
}

__global__
void rhoKernel(const float * __restrict__ fxx, const float * __restrict__ fyy,
               const float * __restrict__ fxy, int n, float * __restrict__ p)
{
    int n_win = blockIdx.x * blockDim.x + threadIdx.x;

    if(n_win < n)
    {
        p[n_win] = fxy[n_win] / (fxx[n_win] * fyy[n_win]);
    }
}

__host__
int index_of_percentile(int n, float percentile)
{
    int i = static_cast<int>(round(n * percentile));
    if(i == n)
    {
        i = n - 1;
    }

    return i;
}

void cudaDCCA(float *y1, float *y2, int n, int *win_sizes, int n_wins, bool rev_seg, float *fxx, float *fyy, float *fxy, int n_threads)
{
    // device variables
    float *d_y1, *d_y2;
    cudaMalloc(&d_y1, n * sizeof(float));
    cudaMalloc(&d_y2, n * sizeof(float));
    int *d_win_sizes;
    cudaMalloc(&d_win_sizes, n_wins * sizeof(int));
    float *d_fxx, *d_fyy, *d_fxy;
    cudaMalloc(&d_fxx, n_wins * sizeof(float));
    cudaMalloc(&d_fyy, n_wins * sizeof(float));
    cudaMalloc(&d_fxy, n_wins * sizeof(float));

    // copy to device
    cudaMemcpy(d_y1, y1, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y2, y2, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_win_sizes, win_sizes, n_wins * sizeof(int), cudaMemcpyHostToDevice);

    // dcca kernel
    cudaStream_t stream_1, stream_2, stream_3;
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);
    cudaStreamCreate(&stream_3);

    dim3 threadsPerBlock(n_threads);
    dim3 blocksPerGrid((n_wins + n_threads - 1) / n_threads);
    if(rev_seg)
    {
        DCCAabsKernelBackwards<<<blocksPerGrid, threadsPerBlock, 0, stream_1>>>(d_y1, d_y1, n, d_win_sizes, n_wins, d_fxx);
        DCCAabsKernelBackwards<<<blocksPerGrid, threadsPerBlock, 0, stream_2>>>(d_y2, d_y2, n, d_win_sizes, n_wins, d_fyy);
        DCCAKernelBackwards<<<blocksPerGrid, threadsPerBlock, 0, stream_3>>>(d_y1, d_y2, n, d_win_sizes, n_wins, d_fxy);
    }
    else
    {
        DCCAabsKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_1>>>(d_y1, d_y1, n, d_win_sizes, n_wins, d_fxx);
        DCCAabsKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_2>>>(d_y2, d_y2, n, d_win_sizes, n_wins, d_fyy);
        DCCAKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_3>>>(d_y1, d_y2, n, d_win_sizes, n_wins, d_fxy);
    }

    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);
    cudaStreamDestroy(stream_3);

    // copy to host
    cudaMemcpy(fxx, d_fxx, n_wins * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fyy, d_fyy, n_wins * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fxy, d_fxy, n_wins * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_y1);
    cudaFree(d_y2);
    cudaFree(d_win_sizes);
    cudaFree(d_fxx);
    cudaFree(d_fyy);
    cudaFree(d_fxy);
}

void cudaDCCAConfInt(int *win_sizes, int n_wins, int n, int n_sim, float conf_level, float *conf_up, float *conf_down, int n_threads)
{
    // random numbers generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // device variables
    float *d_rand;
    cudaMalloc(&d_rand, 2 * n * sizeof(float));

    int *d_win_sizes;
    cudaMalloc(&d_win_sizes, n_wins * sizeof(int));
    cudaMemcpy(d_win_sizes, win_sizes, n_wins * sizeof(int), cudaMemcpyHostToDevice);

    float *d_fxx, *d_fyy, *d_fxy;
    cudaMalloc(&d_fxx, n_wins * sizeof(float));
    cudaMalloc(&d_fyy, n_wins * sizeof(float));
    cudaMalloc(&d_fxy, n_wins * sizeof(float));

    float *d_rho;
    cudaMalloc(&d_rho, n_wins * n_sim * sizeof(float));

    cudaStream_t stream_1, stream_2, stream_3;
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);
    cudaStreamCreate(&stream_3);

    dim3 threadsPerBlock(n_threads);
    dim3 blocksPerGrid((n_wins + n_threads - 1) / n_threads);

    for(int i = 0; i < n_sim; i++)
    {
        // generate random sequences
        curandGenerateNormal(gen, d_rand, 2 * n, 0.0f, 1.0f);

        DCCAabsKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_1>>>(d_rand, d_rand, n, d_win_sizes, n_wins, d_fxx);
        DCCAabsKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_2>>>(&d_rand[n], &d_rand[n], n, d_win_sizes, n_wins, d_fyy);
        DCCAKernel<<<blocksPerGrid, threadsPerBlock, 0, stream_3>>>(d_rand, &d_rand[n], n, d_win_sizes, n_wins, d_fxy);

        cudaDeviceSynchronize();

        rhoKernel<<<blocksPerGrid, threadsPerBlock>>>(d_fxx, d_fyy, d_fxy, n_wins, &d_rho[i * n_wins]);
    }

    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);
    cudaStreamDestroy(stream_3);

    // copy to host
    float *rho = new float [n_wins * n_sim];
    cudaMemcpy(rho, d_rho, n_wins * n_sim * sizeof(float), cudaMemcpyDeviceToHost);

    float *by_win = new float [n_sim];
    for(int i = 0; i < n_wins; i++)
    {
        for(int j = 0; j < n_sim; j++)
        {
            by_win[j] = rho[i + j * n_wins];
        }
        std::sort(by_win, by_win + n_sim);
        conf_up[i] = by_win[index_of_percentile(n_sim, conf_level)];
        conf_down[i] = by_win[index_of_percentile(n_sim, 1 - conf_level)];
    }

    // free memory
    curandDestroyGenerator(gen);
    cudaFree(d_rand);
    cudaFree(d_win_sizes);
    cudaFree(d_fxx);
    cudaFree(d_fyy);
    cudaFree(d_fxy);
    cudaFree(d_rho);

    delete [] rho;
    delete [] by_win;
}

