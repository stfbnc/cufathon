#cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool

cdef extern from "../cuda/dfa_kernel.cuh" nogil:
    void cudaDFA(float *y, int n, int *win_sizes, int n_wins, bool rev_seg, float *fluc_vec, int n_threads)
cdef extern from "../cuda/mfdfa_kernel.cuh" nogil:
    void cudaMFDFA(float *y, int n, int *win_sizes, int n_wins, float *q_vals, int nq, bool rev_seg, float *fluc_vec, int n_threads)
cdef extern from "../cuda/dcca_kernel.cuh" nogil:
    void cudaDCCA(float *y1, float *y2, int n, int *win_sizes, int n_wins, bool rev_seg, float *fxx, float *fyy, float *fxy, int n_threads)
    void cudaDCCAConfInt(int *win_sizes, int n_wins, int n, int n_sim, float conf_level, float *conf_up, float *conf_down, int n_threads)
cdef extern from "../cuda/ht_kernel.cuh" nogil:
    void cudaHT(float *y, int n, int *scales, int n_scales, float *ht, int n_threads)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef dfa(np.ndarray[float, ndim=1, mode='c'] y, int n,
          np.ndarray[int, ndim=1, mode='c'] win_sizes, int n_wins,
          bint rev_seg, int n_threads):
    cdef np.ndarray[float, ndim=1, mode='c'] fluc_vec
    fluc_vec = np.zeros((n_wins, ), dtype=np.float32)
    cudaDFA(&y[0], n, &win_sizes[0], n_wins, rev_seg, &fluc_vec[0], n_threads)

    return fluc_vec

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef mfdfa(np.ndarray[float, ndim=1, mode='c'] y, int n,
            np.ndarray[int, ndim=1, mode='c'] win_sizes, int n_wins,
            np.ndarray[float, ndim=1, mode='c'] q_vals, int nq,
            bint rev_seg, int n_threads):
    cdef np.ndarray[float, ndim=1, mode='c'] fluc_vec
    fluc_vec = np.zeros((n_wins * nq, ), dtype=np.float32)
    cudaMFDFA(&y[0], n, &win_sizes[0], n_wins, &q_vals[0], nq, rev_seg, &fluc_vec[0], n_threads)

    return fluc_vec.reshape((nq, n_wins))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef dcca(np.ndarray[float, ndim=1, mode='c'] y1,
           np.ndarray[float, ndim=1, mode='c'] y2, int n,
           np.ndarray[int, ndim=1, mode='c'] win_sizes, int n_wins,
           bint rev_seg, int n_threads):
    cdef np.ndarray[float, ndim=1, mode='c'] fxx, fyy, fxy
    fxx = np.zeros((n_wins, ), dtype=np.float32)
    fyy = np.zeros((n_wins, ), dtype=np.float32)
    fxy = np.zeros((n_wins, ), dtype=np.float32)
    cudaDCCA(&y1[0], &y2[0], n, &win_sizes[0], n_wins, rev_seg, &fxx[0], &fyy[0], &fxy[0], n_threads)

    return fxx, fyy, fxy

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef dcca_thresholds(np.ndarray[int, ndim=1, mode='c'] win_sizes, int n_wins,
                      int n, int n_sim, float conf_level, int n_threads):
    cdef np.ndarray[float, ndim=1, mode='c'] conf_up, conf_down
    conf_up = np.zeros((n_wins, ), dtype=np.float32)
    conf_down = np.zeros((n_wins, ), dtype=np.float32)
    cudaDCCAConfInt(&win_sizes[0], n_wins, n, n_sim, conf_level, &conf_up[0], &conf_down[0], n_threads)

    return conf_up, conf_down

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef ht(np.ndarray[float, ndim=1, mode='c'] y, int n,
         np.ndarray[int, ndim=1, mode='c'] scales, int n_scales,
         int n_threads):
    cdef np.ndarray[float, ndim=1, mode='c'] ht
    cdef int s_len = 0
    cdef int ht_len

    for s in scales:
        s_len += s
    ht_len = n_scales * (n + 1) - s_len
    ht = np.zeros((ht_len, ), dtype=np.float32)
    cudaHT(&y[0], n, &scales[0], n_scales, &ht[0], n_threads)

    return ht
