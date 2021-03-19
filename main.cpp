#include <vector>
#include <time.h>
#include "c++/dfa.h"


int main()
{
    //cudaEvent_t start, stop;
    //float elapsedTime;

    //cudaEventCreate(&start);
    //cudaEventRecord(start, 0);

    //patch_preprocessing(toNorm, norm, W, H, B, G, R);
    //cudaDeviceSynchronize();

    //cudaEventCreate(&stop);
    //cudaEventRecord(stop, 0);
    //cudaEventSynchronize(stop);

    //cudaEventElapsedTime(&elapsedTime, start, stop);
    //fprintf(stderr, "GPU Time : %f ms\n", elapsedTime);

    //struct timespec start_cpu, end_cpu;
    //clock_gettime(CLOCK_MONOTONIC_RAW, &start_cpu);
    //clock_gettime(CLOCK_MONOTONIC_RAW, &end_cpu);

    //uint64_t delta_us = (end_cpu.tv_sec - start_cpu.tv_sec) * 1000000 + (end_cpu.tv_nsec - start_cpu.tv_nsec) / 1000;
    //fprintf(stderr, "CPU Time : %lu ms\n", delta_us / 1000);

    std::vector<double> ivec = std::vector<double>();
    DFA dfa(ivec);
    dfa.computeFlucVec();

    return 0;
}
