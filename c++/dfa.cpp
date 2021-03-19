#include "dfa.h"


DFA::DFA(std::vector<double> ts)
{
    this->yLen = ts.size();
    this->y = new double [this->yLen];
    this->fromStdVectorToCArray(ts);
}

DFA::~DFA()
{
    delete [] this->y;
}

void DFA::computeFlucVec(){}

void DFA::fromStdVectorToCArray(std::vector<double> vec)
{
    for(int i = 0; i < this->yLen; i++)
        this->y[i] = vec.at(i);
}

