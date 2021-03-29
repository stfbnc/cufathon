#include <time.h>
#include <iostream>
#include <cstdlib>
#include <random>
#include "c++/dfa.h"


int main(int argc, char **argv)
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

    int N = atoi(argv[1]);
    double *in = new double [N];
    double *in_cs = new double [N];

    std::random_device rd;
    std::mt19937 gen(42);  //gen(rd());
    std::normal_distribution<> gdist(0, 1);
    for(int i = 0; i < N; i++)
        in[i] = gdist(gen);
    in_cs[0] = in[0];
    for(int i = 1; i < N; i++)
        in_cs[i] = in_cs[i - 1] + in[i];

    int nWins = 200;
    int *wins = new int [nWins];
    double *fVec = new double [nWins];
    for(int i = 0; i < nWins; i++)
    {
       wins[i] = i + 10;
       fVec[i] = 0.0;
    }

    DFA dfa(in_cs, N);
    dfa.computeFlucVec(wins, nWins, fVec);

    //for(int i = 0; i < nWins; i++)
    //   std::cout << in_cs[i] << std::endl;

    delete [] in;
    delete [] in_cs;
    delete [] wins;
    delete [] fVec;    

    return 0;
}
