#ifndef _ACTIVATION
#define _ACTIVATION

namespace Activation
{
    enum ActivationType
    {
        NONE,
        LOGISTIC_SIGMOID,
        HYPERBOLIC_TANGENT,
        HEAVISIDE_STEP
        /*
            NO SOFTMAX AT THE MOMENT
        */
    };

    double  ApplyActivation     (double v, ActivationType type);

    //Activation Functions
    double  LogisticSigmoid     (double v);
    double  HyperbolicTangent   (double v);
    double  HeavisideStep       (double v);

    //Derivative Functions
    double  LogisticSigmoidDerivative       (double v);
    double  HyperbolicTangentDerivative     (double v);
};


#include "Activation.cpp"
#endif