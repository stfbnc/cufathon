#include <time.h>
#include <iostream>
#include <cstdlib>
#include <random>
#include "c++/dfa.h"
#include "c++/mfdfa.h"
#include "c++/ht.h"
#include "c++/dcca.h"
#include "c++/utils.h"

#include "cudaProfiler.h"

#define THRESHOLDS

int main(int argc, char **argv)
{
    int gpu = 0;
    gpuUtils::getGpuInfo(gpu);

    int N = atoi(argv[1]);
    float *in = new float [N];
    float *in_cs = new float [N];
    float *in_2 = new float [N];
    float *in_cs_2 = new float [N];

    std::random_device rd;
    std::mt19937 gen(42);  //gen(rd());
    std::normal_distribution<> gdist(0, 1);
    for(int i = 0; i < N; i++)
        in[i] = gdist(gen);
    in_cs[0] = in[0];
    for(int i = 1; i < N; i++)
        in_cs[i] = in_cs[i - 1] + in[i];
    for(int i = 0; i < N; i++)
        in_2[i] = gdist(gen);
    in_cs_2[0] = in_2[0];
    for(int i = 1; i < N; i++)
        in_cs_2[i] = in_cs_2[i - 1] + in_2[i];

    /*char file_name[64];
    sprintf(file_name, "test_file_2.txt");
    FILE *out_file = fopen(file_name, "w");
    fprintf(out_file, "in,in_cs\n");
    for(int i = 0; i < N; i++)
    {
	fprintf(out_file, "%.6f,%.6f\n", in_2[i], in_cs_2[i]);
    }
    fclose(out_file);*/

#ifdef DFA_MAIN
    int minWin = 10;
    int nWins = N / 4 - minWin;
    int *wins = new int [nWins];
    float *fVec = new float [nWins];
    
    for(int i = 0; i < nWins; i++)
    {
        wins[i] = i + minWin;
    }
    
    fprintf(stderr, "Input vector length: %d\n", N);
    fprintf(stderr, "win[0] = %d | win[-1] = %d\n", wins[0], wins[nWins - 1]);

    int th = atoi(argv[2]);
    
    DFA dfa(in_cs, N);
    
    cudaEvent_t start_o, stop_o, start_i, stop_i;
    float elapsedTime_o, elapsedTime_i;

    cudaEventCreate(&start_o);
    cudaEventRecord(start_o, 0);

    float I = 0.0, H = 0.0;
    dfa.computeFlucVec(wins, nWins, fVec, I, H, th);

    cudaEventCreate(&stop_o);
    cudaEventRecord(stop_o, 0);
    cudaEventSynchronize(stop_o);

    cudaEventElapsedTime(&elapsedTime_o, start_o, stop_o);
    fprintf(stderr, "DFA FW -> GPU Time (threads = %d) : %f ms\n", th, elapsedTime_o);

    cudaEventCreate(&start_i);
    cudaEventRecord(start_i, 0);

    dfa.computeFlucVec(wins, nWins, fVec, I, H, th, true);

    cudaEventCreate(&stop_i);
    cudaEventRecord(stop_i, 0);
    cudaEventSynchronize(stop_i);

    cudaEventElapsedTime(&elapsedTime_i, start_i, stop_i);
    fprintf(stderr, "DFA BW -> GPU Time (threads = %d) : %f ms\n", th, elapsedTime_i);

    delete [] wins;
    delete [] fVec;
#endif

#ifdef MFDFA_MAIN
    int minWin = 10;
    int minq = -4;
    int nWins = N / 4 - minWin;
    int nq = 10;
    int *wins = new int [nWins];
    float *qs = new float [nq];
    float *hq = new float [nq];
    float *a = new float [nq - 1];
    float *fa = new float [nq - 1];

    for(int i = 0; i < nWins; i++)
    {
        wins[i] = i + minWin;
    }
    for(int i = 0; i < nq; i++)
    {
        qs[i] = i + minq;
    }

    fprintf(stderr, "Input vector length: %d\n", N);
    fprintf(stderr, "win[0] = %d | win[-1] = %d\n", wins[0], wins[nWins - 1]);

    int th = atoi(argv[2]);
    MFDFA mfdfa(in_cs, N);

    cudaEvent_t start_o, stop_o, start_i, stop_i;
    float elapsedTime_o, elapsedTime_i;

    cudaEventCreate(&start_o);
    cudaEventRecord(start_o, 0);

    //mfdfa.computeFlucVec(wins, nWins, qs, nq, hq, th);
    mfdfa.computeMultifractalSpectrum(wins, nWins, qs, nq, a, fa, th);

    cudaEventCreate(&stop_o);
    cudaEventRecord(stop_o, 0);
    cudaEventSynchronize(stop_o);

    cudaEventElapsedTime(&elapsedTime_o, start_o, stop_o);
    fprintf(stderr, "FW MFDFA -> GPU Time (threads = %d) : %f ms\n", th, elapsedTime_o);

    cudaEventCreate(&start_i);
    cudaEventRecord(start_i, 0);

    //mfdfa.computeFlucVec(wins, nWins, qs, nq, hq, th, true);
    mfdfa.computeMultifractalSpectrum(wins, nWins, qs, nq, a, fa, th, true);

    cudaEventCreate(&stop_i);
    cudaEventRecord(stop_i, 0);
    cudaEventSynchronize(stop_i);

    cudaEventElapsedTime(&elapsedTime_i, start_i, stop_i);
    fprintf(stderr, "BW MFDFA -> GPU Time (threads = %d) : %f ms\n", th, elapsedTime_i);

    delete [] wins;
    delete [] qs;
    delete [] hq;
    delete [] a;
    delete [] fa;
#endif

#ifdef HT_MAIN
    int nScales = 10;
    int *scales = new int [nScales];
    
    int sLen = 0;
    for(int i = 0; i < nScales; i++)
    {
        scales[i] = 10 * (i + 1);
        sLen += scales[i];
    }
    
    int hfLen = nScales * (N + 1) - sLen;
    float *fVec = new float [hfLen];
    /*for(int i = 0; i < hfLen; i++)
    {
        fVec[i] = 0.0;
    }*/

    fprintf(stderr, "Input vector length: %d\n", N);

    int th = atoi(argv[2]);
    HT ht(in_cs, N);

    cudaEvent_t start_o, stop_o;
    float elapsedTime_o;

    cudaEventCreate(&start_o);
    cudaEventRecord(start_o, 0);

    ht.computeFlucVec(scales, nScales, fVec, th);

    cudaEventCreate(&stop_o);
    cudaEventRecord(stop_o, 0);
    cudaEventSynchronize(stop_o);

    cudaEventElapsedTime(&elapsedTime_o, start_o, stop_o);
    fprintf(stderr, "HT -> GPU Time (threads = %d) : %f ms\n", th, elapsedTime_o);

    delete [] fVec;
#endif

#ifdef DCCA_MAIN
    int minWin = 10;
    int nWins = N / 4 - minWin;
    int *wins = new int [nWins];
    float *rho = new float [nWins];

    for(int i = 0; i < nWins; i++)
    {
        wins[i] = i + minWin;
    }

    fprintf(stderr, "Input vector length: %d\n", N);
    fprintf(stderr, "win[0] = %d | win[-1] = %d\n", wins[0], wins[nWins - 1]);

    int th = atoi(argv[2]);

    DCCA dcca(in_cs, in_cs_2, N);

    cudaEvent_t start_o, stop_o, start_i, stop_i;
    float elapsedTime_o, elapsedTime_i;

    cudaEventCreate(&start_o);
    cudaEventRecord(start_o, 0);

    dcca.computeFlucVec(wins, nWins, rho, th);

    cudaEventCreate(&stop_o);
    cudaEventRecord(stop_o, 0);
    cudaEventSynchronize(stop_o);

    cudaEventElapsedTime(&elapsedTime_o, start_o, stop_o);
    fprintf(stderr, "DCCA FW -> GPU Time (threads = %d) : %f ms\n", th, elapsedTime_o);

    cudaEventCreate(&start_i);
    cudaEventRecord(start_i, 0);

    dcca.computeFlucVec(wins, nWins, rho, th, true);

    cudaEventCreate(&stop_i);
    cudaEventRecord(stop_i, 0);
    cudaEventSynchronize(stop_i);

    cudaEventElapsedTime(&elapsedTime_i, start_i, stop_i);
    fprintf(stderr, "DCCA BW -> GPU Time (threads = %d) : %f ms\n", th, elapsedTime_i);

    delete [] wins;
    delete [] rho;
#endif

#ifdef THRESHOLDS
    int minWin = 10;
    int nWins = N / 4 - minWin;
    int *wins = new int [nWins];

    for(int i = 0; i < nWins; i++)
    {
        wins[i] = i + minWin;
    }

    fprintf(stderr, "Input vector length: %d\n", N);
    fprintf(stderr, "win[0] = %d | win[-1] = %d\n", wins[0], wins[nWins - 1]);

    int th = atoi(argv[2]);
    int nSim = atof(argv[3]);
    float confLevel = 0.95f;
    float *confUp = new float [nWins];
    float *confDown = new float [nWins];

    DCCA dcca(in_cs, in_cs_2, N);

    cudaEvent_t start_o, stop_o;
    float elapsedTime_o;

    cudaEventCreate(&start_o);
    cudaEventRecord(start_o, 0);

    dcca.computeThresholds(wins, nWins, nSim, confLevel, confUp, confDown, th);

    cudaEventCreate(&stop_o);
    cudaEventRecord(stop_o, 0);
    cudaEventSynchronize(stop_o);

    cudaEventElapsedTime(&elapsedTime_o, start_o, stop_o);
    fprintf(stderr, "THRESHOLDS -> GPU Time (threads = %d) : %f ms\n", th, elapsedTime_o);

    delete [] wins;
    delete [] confUp;
    delete [] confDown;
#endif

    delete [] in;
    delete [] in_cs;
    delete [] in_2;
    delete [] in_cs_2;

    return 0;
}
