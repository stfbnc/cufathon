#include <time.h>
#include <iostream>
#include <cstdlib>
#include <random>
#include "c++/dfa.h"
#include "c++/utils.h"


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
    int nWins = N / 4 - minWin;
    int *wins = new int [nWins];
    double *fVec = new double [nWins];
    for(int i = 0; i < nWins; i++)
    {
       wins[i] = i + minWin;
       fVec[i] = 0.0;
    }

    fprintf(stderr, "Input vector length: %d\n", N);
    fprintf(stderr, "win[0] = %d | win[-1] = %d\n", wins[0], wins[nWins - 1]);

/*    DFA dfa(in_cs, N);

    cudaEvent_t start_o, stop_o, start_i, stop_i;
    float elapsedTime_o, elapsedTime_i;

    cudaEventCreate(&start_o);
    cudaEventRecord(start_o, 0);

    dfa.computeFlucVec(wins, nWins, fVec);

    cudaEventCreate(&stop_o);
    cudaEventRecord(stop_o, 0);
    cudaEventSynchronize(stop_o);

    cudaEventElapsedTime(&elapsedTime_o, start_o, stop_o);
    fprintf(stderr, "GPU Time (outer) : %f ms\n", elapsedTime_o);

    //for(int i = 0; i < 3; i++)
    //    std::cout << fVec[i] << std::endl;

    cudaEventCreate(&start_i);
    cudaEventRecord(start_i, 0);

    dfa.computeFlucVecInner(wins, nWins, fVec);

    cudaEventCreate(&stop_i);
    cudaEventRecord(stop_i, 0);
    cudaEventSynchronize(stop_i);

    cudaEventElapsedTime(&elapsedTime_i, start_i, stop_i);
    fprintf(stderr, "GPU Time (inner) : %f ms\n", elapsedTime_i);
*/
    //for(int i = 0; i < 3; i++)
    //    std::cout << fVec[i] << std::endl;

    delete [] in;
    delete [] in_cs;
    delete [] wins;
    delete [] fVec;    

    return 0;
}
