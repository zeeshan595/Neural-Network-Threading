#ifndef _NEURON
#define _NEURON

namespace NN
{
    struct Neuron
    {
        double                      neuron_value        = 0.0;
        double                      bias_value          = 0.0;
        Activation::ActivationType  activation_type     = Activation::ActivationType::NONE;
        std::vector<Synapse*>       synapsis_in;
        std::vector<Synapse*>       synapsis_out;
    };
};

#endif