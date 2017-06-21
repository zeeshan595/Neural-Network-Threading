#ifndef _BASE
#define _BASE

namespace NN
{
    class Base
    {
    public:
        virtual std::vector<double> Compute(
            std::vector<double> inputs
        ){  return {};  }

        virtual double GetMeanSquaredError(
            std::vector<std::vector<double> > dataset
        ){  return 0;   }

        virtual std::vector<double> GetWeights()
        {   return {};  }

        virtual void SetWeights(
            std::vector<double> weights
        ){}
    };
};

#endif