import cupy as cp
import numpy as np
import fathon
import sys

from cupy import prof
from timeit import default_timer as timer


HTCompute_src = r"""
extern "C"
{
    const int MAX_SCALE = 1000;

    __device__ void polynomialFit(
    	    int scale,
    	    int ord,
    	    double * __restrict__ fitCoeffs)
    {
    	for(int i = 0; i < (ord + 1); i++)
    	{
    	    fitCoeffs[i] = 1.0;
    	}
    }

    __global__ void cuda_HTCompute(
	    const double * __restrict__ y,
	    const double * __restrict__ t,
	    int scale,
	    int N,
	    int polOrd,
	    double * __restrict__ vecHt)
    {
        const int tx { static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
        const int stride { static_cast<int>(blockDim.x * gridDim.x) };

	const int loopRange = N - scale + 1;
	double fitCoeffs[2];

        for(int tid = tx; tid < loopRange; tid += stride)
        {
    	    double f = 0.0;

            polynomialFit(scale, polOrd + 1, fitCoeffs);

    	    for(int j = 0; j < scale; j++)
    	    {
        	double var = y[tid + j];
        	for(int k = 0; k < (polOrd + 1); k++)
       		{
            	    var -= fitCoeffs[k] * pow(t[tid + j], static_cast<double>(k));
        	}
        	f += pow(var, 2.0);
    	    }

    	    vecHt[tid] = sqrt(f / static_cast<double>(scale));
        }
    }
}
"""


def HTKernel(y, t, scale, N, pol_ord, ht):
    device_id = cp.cuda.Device()
    num_sm = device_id.attributes["MultiProcessorCount"]
    threads_per_block = (128, )
    blocks_per_grid = (num_sm * 20, )
    
    module = cp.RawModule(code=HTCompute_src, options=("-std=c++11", ))
    kernel = module.get_function("cuda_HTCompute")

    kernel_args = (y, t, cp.int32(scale), cp.int32(N), cp.int32(pol_ord), ht)

    kernel(blocks_per_grid, threads_per_block, kernel_args)

    cp.cuda.runtime.deviceSynchronize()


def gpu_ht(y, t, scale, pol_ord):
    N = len(y)
    ht = cp.empty((N - scale + 1, ), dtype=y.dtype)

    HTKernel(y, t, scale, N, pol_ord, ht)

    return ht


def cpu_ht(y, t, scale, pol_ord):
    N = len(y)
    ht = np.empty((N - scale + 1, ), dtype=y.dtype)
    for tid in range(N - scale + 1): 
        f = 0.0
        t_fit = t[tid:tid+scale]
        y_fit = y[tid:tid+scale]

        fit_coeffs = np.ones((pol_ord + 1, ), dtype=float)

        for j in range(scale):
            var = y_fit[j]
            for k in range(pol_ord + 1):
                var -= fit_coeffs[k] * (t_fit[j] ** k)
 
            f += (var ** 2.0)

        ht[tid] = np.sqrt(f / float(scale))

    return ht


if __name__ == "__main__":
    max_scale = 1000
    sz = int(sys.argv[1])
    scale = int(sys.argv[2])
    if scale > max_scale:
        raise ValueError("Maximum supported scale is {:d}".format(max_scale))
    pol_ord = 1

    y = np.random.randn(sz)
    t = np.arange(1, sz + 1, dtype=float)
    gpu_y = cp.array(y)
    gpu_t = cp.array(t)

    #f_ht = fathon.HT(y)
    start = timer()
    #ht1 = f_ht.computeHt(scale)
    ht1 = cpu_ht(y, t, scale, pol_ord)
    end = timer()
    print("CPU time: {:f}".format(end - start))

    start = timer()
    ht2 = gpu_ht(gpu_y, gpu_t, scale, pol_ord)
    end = timer()
    print("GPU time: {:f}".format(end - start))

    ht2 = cp.asnumpy(ht2)

    print(np.allclose(ht1, ht2))
