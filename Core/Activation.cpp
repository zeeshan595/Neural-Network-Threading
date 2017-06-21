double  Activation::ApplyActivation (double v, ActivationType type)
{
    switch(type)
    {
        case LOGISTIC_SIGMOID:
            return LogisticSigmoid(v);
        case HYPERBOLIC_TANGENT:
            return HyperbolicTangent(v);
        case HEAVISIDE_STEP:
            return HeavisideStep(v);
        default:
            return 0;
    }
}

//Activation Functions
double  Activation::LogisticSigmoid (double v)
{
    return ( 1.0 / ( 1.0 + exp( -v ) ) );
}
double  Activation::HyperbolicTangent (double v)
{
    return tanh( v );
}
double  Activation::HeavisideStep (double v)
{
    if (v < 0)
        return 0;
    else
        return 1;
}

//Derivative Functions
double  Activation::LogisticSigmoidDerivative (double v)
{
    return ( ( 1 - v ) * v );
}
double  Activation::HyperbolicTangentDerivative (double v)
{
    return  ( ( 1 - v ) * ( 1 + v ) );
}