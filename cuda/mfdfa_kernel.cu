#include "mfdfa_kernel.cuh"

__global__
void MFDFAKernel(const float * __restrict__ y, int n,
                 const int * __restrict__ win_sizes, int n_wins,
                 const float * __restrict__ q_vals, int nq,
                 float * __restrict__ fluc_vec)
{
    int iq = blockIdx.x * blockDim.x + threadIdx.x;
    int iw = blockIdx.y * blockDim.y + threadIdx.y;

    if((iq < nq) && (iw < n_wins))
    {
        float q_val = q_vals[iq];
        int curr_win_size = win_sizes[iw];
        int n_s = n / curr_win_size;
        float f = 0.0f;
        
        for(int i = 0; i < n_s; i++)
        {
            float rms = 0.0f;
            int start_lim = i * curr_win_size;
            float m = 0.0f, q = 0.0f;
            
            fit(curr_win_size, start_lim + 1, y + start_lim, &m, &q);
            
            for(int j = 0; j < curr_win_size; j++)
            {   
                float var = y[start_lim + j] - (q + m * (start_lim + 1 + j));
                rms += pow(var, 2.0f);
            }

            if(q_val == 0.0f)
            {
                f += log(rms / curr_win_size);
            }
            else
            {
                f += pow(rms / curr_win_size, 0.5f * q_val);
            }
        }
        
        if(q_val == 0.0f)
        {
            fluc_vec[iq * n_wins + iw] = exp(f / (2.0f * n_s));
        }
        else
        {
            fluc_vec[iq * n_wins + iw] = pow(f / n_s, 1.0f / q_val);
        }
    }
}

__global__
void MFDFAKernelBackwards(const float * __restrict__ y, int n,
                          const int * __restrict__ win_sizes, int n_wins,
                          const float * __restrict__ q_vals, int nq,
                          float * __restrict__ fluc_vec)
{
    int iq = blockIdx.x * blockDim.x + threadIdx.x;
    int iw = blockIdx.y * blockDim.y + threadIdx.y;

    if((iq < nq) && (iw < n_wins))
    {
        float q_val = q_vals[iq];
        int curr_win_size = win_sizes[iw];
        int n_s = n / curr_win_size;
        float f = 0.0f;

        for(int i = 0; i < n_s; i++)
        {
            float rms1 = 0.0f, rms2 = 0.0f;
            int start_lim = i * curr_win_size;
            float m = 0.0f, q = 0.0f;

            fit(curr_win_size, start_lim + 1, y + start_lim, &m, &q);

            for(int j = 0; j < curr_win_size; j++)
            {   
                float var = y[start_lim + j] - (q + m * (start_lim + 1 + j));
                rms1 += pow(var, 2.0f);
            }

            start_lim = i * curr_win_size + (n - n_s * curr_win_size);
            fit(curr_win_size, start_lim + 1, y + start_lim, &m, &q);
 
            for(int j = 0; j < curr_win_size; j++)
            {
                float var = y[start_lim + j] - (q + m * (start_lim + 1 + j));
                rms2 += pow(var, 2.0f);
            }

            if(q_val == 0.0f)
            {   
                f += log(rms1 / curr_win_size) + log(rms2 / curr_win_size);
            }
            else
            {   
                f += pow(rms1 / curr_win_size, 0.5f * q_val) + pow(rms2 / curr_win_size, 0.5f * q_val);
            }
        }

        if(q_val == 0.0f)
        {   
            fluc_vec[iq * n_wins + iw] = exp(f / (4.0f * n_s));
        }
        else
        {   
            fluc_vec[iq * n_wins + iw] = pow(f / (2.0f * n_s), 1.0f / q_val);
        }
    }
}

void cudaMFDFA(float *y, int n, int *win_sizes, int n_wins, float *q_vals, int nq, bool rev_seg, float *fluc_vec, int n_threads)
{
    // device variables
    float *d_y;
    cudaMalloc(&d_y, n * sizeof(float));
    int *d_win_sizes;
    cudaMalloc(&d_win_sizes, n_wins * sizeof(int));
    float *d_q_vals;
    cudaMalloc(&d_q_vals, nq * sizeof(float));
    float *d_fluc_vec;
    cudaMalloc(&d_fluc_vec, n_wins * nq * sizeof(float));

    // copy to device
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_win_sizes, win_sizes, n_wins * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_vals, q_vals, nq * sizeof(float), cudaMemcpyHostToDevice);

    // mfdfa kernel
    dim3 threadsPerBlock(n_threads, n_threads);
    dim3 blocksPerGrid((nq + n_threads - 1) / n_threads, (n_wins + n_threads - 1) / n_threads);
    if(rev_seg)
    {
        MFDFAKernelBackwards<<<blocksPerGrid, threadsPerBlock>>>(d_y, n, d_win_sizes, n_wins, d_q_vals, nq, d_fluc_vec);
    }
    else
    {
        MFDFAKernel<<<blocksPerGrid, threadsPerBlock>>>(d_y, n, d_win_sizes, n_wins, d_q_vals, nq, d_fluc_vec);
    }

    // copy to host
    cudaMemcpy(fluc_vec, d_fluc_vec, n_wins * nq * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_y);
    cudaFree(d_win_sizes);
    cudaFree(d_q_vals);
    cudaFree(d_fluc_vec);
}
