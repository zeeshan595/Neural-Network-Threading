#include "NeuralNetworkCommon.h"

NN::Base* CreateNetwork()
{
    return new NN::MFNN({4, 7, 3}, {
        Activation::ActivationType::NONE,   //First must be none due to input layer
        Activation::ActivationType::LOGISTIC_SIGMOID,
        Activation::ActivationType::LOGISTIC_SIGMOID
    });
}

void TrainingCallBack(double progress, double msr, std::vector<double> weights)
{
    std::cout << "Progress: " << progress << " MSR: " << msr << std::endl;
}

int main()
{
    NN::Normalize(IrisData::dataset, { 0, 1, 2, 3 });
    NN::Randomize(IrisData::dataset);

    //Actual Neural Network
    NN::MFNN* my_network = (NN::MFNN*)CreateNetwork();

    std::cout << "PSO Threading Test" << std::endl;
    std::vector<double> good_weights = NN::PSO(IrisData::dataset, 20, 1000, &CreateNetwork, &TrainingCallBack);
    my_network->SetWeights(good_weights);
    std::cout << "MSR: " << my_network->GetMeanSquaredError(IrisData::dataset) << std::endl;

    std::cout << "End Of Program" << std::endl;
    std::cin.get();

    //Clean up
    delete my_network;
    return 0;
}