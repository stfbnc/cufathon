#include "dfa.h"
#include <iostream>


DFA::DFA(double *y, int yLen)
{
    this->yLen = yLen;
    cudaMalloc((void**)&(this->y), this->yLen * sizeof(double));
    cudaMemcpy(this->y, y, this->yLen * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&(this->t), this->yLen * sizeof(double));
    linRange(this->t, this->yLen, 1);
}

DFA::~DFA()
{
    cudaFree(this->y);
    cudaFree(this->t);
}

void DFA::computeFlucVec(int *winSizes, int nWins, double *F, bool revSeg)
{
    //for(int nWin = 0; nWin < nWins; nWin++)
    //{
    //    int currWinSize = winSizes[nWin];
    //    int Ns = this->yLen / currWinSize;

        double *flucVec = nullptr;
        //cudaMalloc((void**)&f, Ns * sizeof(double));
        cudaMalloc((void**)&flucVec, nWins * sizeof(double));
        int *winSizesGpu = nullptr;
        cudaMalloc((void**)&winSizesGpu, nWins * sizeof(int));
        cudaMemcpy(winSizesGpu, winSizes, nWins * sizeof(int), cudaMemcpyHostToDevice);

        //cudaDFA(this->y, this->t, currWinSize, Ns, f);
        cudaDFA(this->y, this->t, this->yLen, winSizesGpu, nWins, flucVec);

        cudaMemcpy(F, flucVec, nWins * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(flucVec);
        cudaFree(winSizesGpu);
        //double *fcpu = new double [Ns];
        //cudaMemcpy(fcpu, f, Ns * sizeof(double), cudaMemcpyDeviceToHost);
        //double sum = 0.0;
        //for(int i = 0; i < Ns; i++)
        //    sum += fcpu[i];
        //delete [] fcpu;
        //cudaFree(f);

    for(int nWin = 0; nWin < nWins; nWin++)
    {
        fprintf(stderr, "F[%d]: %lf\n", nWin, F[nWin]);
    }
}

