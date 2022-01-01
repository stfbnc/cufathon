#include "dfa_kernel.cuh"

__global__
void DFAKernel(const float * __restrict__ y, int n,
               const int * __restrict__ win_sizes, int n_wins,
               float * __restrict__ fluc_vec)
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
            float m = 0.0f, q = 0.0f;

            fit(curr_win_size, start_lim + 1, y + start_lim, &m, &q);

            for(int j = 0; j < curr_win_size; j++)
            {
                float var = y[start_lim + j] - (q + m * (start_lim + 1 + j));
                f += pow(var, 2.0f);
            }
        }

        fluc_vec[n_win] = sqrt(f / (n_s * curr_win_size));
    }
}

__global__
void DFAKernelBackwards(const float * __restrict__ y, int n,
                        const int * __restrict__ win_sizes, int n_wins,
                        float * __restrict__ fluc_vec)
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
            float m = 0.0f, q = 0.0f;

            fit(curr_win_size, start_lim + 1, y + start_lim, &m, &q);

            for(int j = 0; j < curr_win_size; j++)
            {
                float var = y[start_lim + j] - (q + m * (start_lim + 1 + j));
                f += pow(var, 2.0f);
            }

            start_lim = i * curr_win_size + (n - n_s * curr_win_size);
            fit(curr_win_size, start_lim + 1, y + start_lim, &m, &q);

            for(int j = 0; j < curr_win_size; j++)
            {
                float var = y[start_lim + j] - (q + m * (start_lim + 1 + j));
                f += pow(var, 2.0f);
            }
        }

        fluc_vec[n_win] = sqrt(f / (2.0f * n_s * curr_win_size));
    }
}

void cudaDFA(float *y, int n, int *win_sizes, int n_wins, bool rev_seg, float *fluc_vec, int n_threads)
{
    // device variables
    float *d_y;
    cudaMalloc(&d_y, n * sizeof(float));
    int *d_win_sizes;
    cudaMalloc(&d_win_sizes, n_wins * sizeof(int));
    float *d_fluc_vec;
    cudaMalloc(&d_fluc_vec, n_wins * sizeof(float));

    // copy to device
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_win_sizes, win_sizes, n_wins * sizeof(int), cudaMemcpyHostToDevice);

    // dfa kernel
    dim3 threadsPerBlock(n_threads);
    dim3 blocksPerGrid((n_wins + n_threads - 1) / n_threads);
    if(rev_seg)
    {
        DFAKernelBackwards<<<blocksPerGrid, threadsPerBlock>>>(d_y, n, d_win_sizes, n_wins, d_fluc_vec);
    }
    else
    {
        DFAKernel<<<blocksPerGrid, threadsPerBlock>>>(d_y, n, d_win_sizes, n_wins, d_fluc_vec);
    }

    // copy to host
    cudaMemcpy(fluc_vec, d_fluc_vec, n_wins * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_y);
    cudaFree(d_win_sizes);
    cudaFree(d_fluc_vec);
}
