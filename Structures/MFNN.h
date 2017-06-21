#ifndef _MFNN
#define _MFNN

namespace NN
{
    struct MFNNThreadStructure
    {
        Neuron*     neuron      = NULL; 
    };

    class MFNN : public Base
    {
    public:
        struct Layer
        {
            //list of neurons in a layer
            std::vector<Neuron*>    neurons;
            //list of synapsis in a layer
            std::vector<Synapse*>   synapsis;
        };

        MFNN(
            std::vector<uint32_t> neurons_per_layer,
            std::vector<Activation::ActivationType> layer_activations
        );
        ~MFNN();
        
        std::vector<double> Compute(
            std::vector<double> inputs
        );

        double GetMeanSquaredError(
            std::vector<std::vector<double> > dataset
        );

        void SetRandomWeights();
        std::vector<double> GetWeights();
        void SetWeights(
            std::vector<double> weights
        );

        Layer* GetLayer(uint32_t layer_number);
        std::vector<double> GetOutput();

        std::vector<Layer*>     layers;
    };
};

#include "MFNN.cpp"
#endif