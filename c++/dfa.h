#ifndef DFA_H
#define DFA_H

#include <iostream>
#include <vector>
#include "../cuda/dfa_kernel.h"


class DFA
{
public:
    explicit DFA(std::vector<double> ts);
    ~DFA();
    void computeFlucVec();
private:
    void fromStdVectorToCArray(std::vector<double> vec);

    double *y = nullptr;
    int yLen = 0;
};

#endif
