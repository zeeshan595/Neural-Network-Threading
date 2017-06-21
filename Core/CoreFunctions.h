#ifndef _CORE_FUNCTIONS
#define _CORE_FUNCTIONS

namespace NN
{
    std::vector<uint32_t> Shuffle(
        std::vector<uint32_t> sequence
    );
    
    void PrintDataSet(
        std::vector<std::vector<double> > dataset,
        uint32_t tab
    );

    void Normalize(
        std::vector<std::vector<double> > &dataset,
        std::vector<uint32_t> cols
    );

    void Randomize(
        std::vector<std::vector<double> > &dataset
    );
};

#include "CoreFunctions.cpp"
#endif