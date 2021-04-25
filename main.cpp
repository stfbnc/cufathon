#include <time.h>
#include <iostream>
#include <cstdlib>
#include <random>
#include "c++/dfa.h"
#include "c++/mfdfa.h"
#include "c++/ht.h"
#include "c++/utils.h"

#include "cudaProfiler.h"


int main(int argc, char **argv)
{
    int gpu = 0;
    gpuUtils::getGpuInfo(gpu);

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

    int minWin = 10;
    int minq = -4;
    int nWins = N / 4 - minWin;
    int nq = 10;
    int nScales = 10;
    int *wins = new int [nWins];
    double *qs = new double [nq];
    int *scales = new int [nScales];
    //double *fVec = new double [nWins * nq];
    for(int i = 0; i < nWins; i++)
    {
        wins[i] = i + minWin;
        //fVec[i] = 0.0;
    }
    for(int i = 0; i < nq; i++)
    {
        qs[i] = i + minq;
    }
    int sLen = 0;
    for(int i = 0; i < nScales; i++)
    {
        scales[i] = 10 * (i + 1);
        sLen += scales[i];
    }
    
    int hfLen = nScales * (N + 1) - sLen;
    double *fVec = new double [hfLen];
    //for(int i = 0; i < (nWins * nq); i++)
    for(int i = 0; i < hfLen; i++)
    {
        fVec[i] = 0.0;
    }

    fprintf(stderr, "Input vector length: %d\n", N);
    fprintf(stderr, "win[0] = %d | win[-1] = %d\n", wins[0], wins[nWins - 1]);

    int th = atoi(argv[2]);
    int th2D = atoi(argv[3]);
    //DFA dfa(in_cs, N);
    //MFDFA mfdfa(in_cs, N);
    HT ht(in_cs, N);

    cudaEvent_t start_o, stop_o, start_i, stop_i;
    float elapsedTime_o, elapsedTime_i;

    cudaEventCreate(&start_o);
    cudaEventRecord(start_o, 0);

    //dfa.computeFlucVec(wins, nWins, fVec, th);
    //mfdfa.computeFlucVec(wins, nWins, qs, nq, fVec, th);
    ht.computeFlucVec(scales, nScales, fVec, th, th2D);

    cudaEventCreate(&stop_o);
    cudaEventRecord(stop_o, 0);
    cudaEventSynchronize(stop_o);

    cudaEventElapsedTime(&elapsedTime_o, start_o, stop_o);
    fprintf(stderr, "1D -> GPU Time (threads = %d) : %f ms\n", th, elapsedTime_o);

    cudaEventCreate(&start_i);
    cudaEventRecord(start_i, 0);

    //dfa.computeFlucVecInner(wins, nWins, fVec);
    //mfdfa.computeFlucVec2D(wins, nWins, qs, nq, fVec, th2D);
    ht.computeFlucVec_2(scales, nScales, fVec, th, th2D);

    cudaEventCreate(&stop_i);
    cudaEventRecord(stop_i, 0);
    cudaEventSynchronize(stop_i);

    cudaEventElapsedTime(&elapsedTime_i, start_i, stop_i);
    fprintf(stderr, "2D -> GPU Time (threads = %d) : %f ms\n", th, elapsedTime_i);

    delete [] in;
    delete [] in_cs;
    delete [] wins;
    delete [] qs;
    delete [] fVec;    

    return 0;
}
