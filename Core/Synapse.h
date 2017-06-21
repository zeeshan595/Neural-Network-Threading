#ifndef _SYNAPSE
#define _SYNAPSE

namespace NN
{
    struct Neuron;
    struct Synapse
    {
        double      synapse_weight              = 0.0;
        Neuron*     connected_from_neuron       = NULL;
        Neuron*     connected_to_neuron         = NULL;
    };
};

#endif