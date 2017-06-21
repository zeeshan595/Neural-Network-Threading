#include "Common.h"

int main()
{
    NN::Normalize(IrisData::dataset, { 0, 1, 2, 3 });
    NN::Randomize(IrisData::dataset);

    //Actual Neural Network
    NN::MFNN my_network = NN::MFNN({4, 7, 3}, {
        Activation::ActivationType::NONE,   //First must be none due to input layer
        Activation::ActivationType::HYPERBOLIC_TANGENT,
        Activation::ActivationType::LOGISTIC_SIGMOID
    });

    std::cout << "PSO Threading Test" << std::endl;

    {
        //Create a seperate network for each particle
        uint32_t particle_count = 50;
        std::vector<NN::Base*> particles(particle_count);
        for (uint32_t i = 0; i < particle_count; i++)
        {
            particles[i] = new NN::MFNN({4, 7, 3}, {
                Activation::ActivationType::NONE,   //First must be none due to input layer
                Activation::ActivationType::HYPERBOLIC_TANGENT,
                Activation::ActivationType::LOGISTIC_SIGMOID
            });
        }
        //Train the network and set weights 
        my_network.SetWeights(NN::PSO(IrisData::dataset, 10000, particles));
        for (uint32_t i = 0; i < particle_count; i++)
            delete particles[i];
    }

    return 0;
}