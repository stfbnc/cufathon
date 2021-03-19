import cupy as cp
import numpy as np
import fathon
from fathon import fathonUtils as fu
import sys

from cupy import prof
from timeit import default_timer as timer


def gpu_dfa(y, t, curr_win_size):
    device_id = cp.cuda.Device()
    max_threads = device_id.attributes["MaxThreadsPerBlock"]
    max_blocks_x = device_id.attributes["MaxGridDimX"]
    max_blocks_y = device_id.attributes["MaxGridDimY"]
    block_size_x = max_threads #256
    #block_size_y = max_threads // block_size_x
    threads_per_block = (block_size_x, )#block_size_y)

    module = cp.RawModule(path="./fatbins/dfa.fatbin")
    kernel = module.get_function("cuda_DFACompute")

    f_vec = np.zeros((len(win_sizes), ), dtype=float)
    len_y = len(y)
    #N_s_max = len_y // np.min(win_sizes)
    #f = cp.zeros((N_s_max * np.max(win_sizes), ), dtype=y.dtype)
    for i, curr_win_size in enumerate(win_sizes):
        N_s = len_y // curr_win_size
        
        blocks_x = (N_s + block_size_x - 1) // block_size_x
        if blocks_x > max_blocks_x:
            blocks_x = max_blocks_x
        #blocks_y = (curr_win_size + block_size_y - 1) // block_size_y
        #if blocks_y > max_blocks_y:
        #    blocks_y = max_blocks_y
        
        f = cp.zeros((N_s * curr_win_size, ), dtype=y.dtype)
        #f = cp.zeros((N_s, ), dtype=y.dtype)

        blocks_per_grid = (blocks_x, )#blocks_y)
        kernel_args = (y, t, cp.int32(curr_win_size), cp.int32(N_s), f)
        kernel(blocks_per_grid, threads_per_block, kernel_args)
        cp.cuda.runtime.deviceSynchronize()

        f_cpu = cp.asnumpy(f)
        f_sum = np.sum(f_cpu)
        f_vec[i] = np.sqrt(f_sum / (N_s * curr_win_size))
        #break

    return f_vec


def cpu_dfa(y, t, win_sizes):
    f_vec = np.zeros((len(win_sizes), ), dtype=float)
    pol_ord = 1

    for i, curr_win_size in enumerate(win_sizes):
        N_s = len(y) // curr_win_size
        f = 0.0
        for tid in range(N_s):
            start_lim = tid * curr_win_size
            fit_coeffs = np.polyfit(t[start_lim:start_lim+curr_win_size], y[start_lim:start_lim+curr_win_size], pol_ord)
            for j in range(curr_win_size):
                var = y[start_lim + j]
                for k in range(pol_ord + 1):
                    var -= fit_coeffs[pol_ord - k] * (t[start_lim + j] ** k)
                f += (var ** 2.0)
        f_vec[i] = np.sqrt(f / (N_s * curr_win_size))

    return f_vec


if __name__ == "__main__":
    sz = int(sys.argv[1])
    #win_sizes = fu.linRangeByStep(10, 11)
    #win_sizes = fu.linRangeByStep(sz//4, sz//4 + 1)
    win_sizes = fu.linRangeByStep(sz // 4 - 1000, sz // 4)

    y = np.random.randn(sz)
    t = np.arange(1, sz + 1, dtype=float)
    gpu_y = cp.array(y)
    gpu_t = cp.array(t)

    f_dfa = fathon.DFA(y)
    start = timer()
    n, dfa1 = f_dfa.computeFlucVec(win_sizes, polOrd=1, revSeg=False)
    end = timer()
    print("CPU fathon time: {:f}".format(end - start))

#    start = timer()
#    dfa2 = cpu_dfa(y, t, win_sizes)
#    end = timer()
#    print("CPU time: {:f}".format(end - start))

    start = timer()
#    with prof.time_range("dfa_gpu", 0):
    dfa3 = gpu_dfa(gpu_y, gpu_t, win_sizes)
    end = timer()
    print("GPU time: {:f}".format(end - start))

    print(dfa1)
    # print(dfa2)
    print(dfa3)

#    print("fathon vs CPU: {}".format(np.allclose(dfa1, dfa2)))
    print("fathon vs GPU: {}".format(np.allclose(dfa1, dfa3)))
#    print("CPU vs GPU: {}".format(np.allclose(dfa2, dfa3)))
