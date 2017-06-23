#include "NeuralNetworkCommon.h"

NN::Base* CreateNetwork()
{
    return new NN::MFNN({21, 10, 4}, {
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
    NN::Normalize(CarEvaluationData::dataset, { 0, 1, 2, 3 });
    NN::Randomize(CarEvaluationData::dataset);

    //Actual Neural Network
    NN::MFNN* my_network = (NN::MFNN*)CreateNetwork();

    std::cout << "PSO Threading Test" << std::endl;
    std::vector<double> good_weights = NN::PSO(CarEvaluationData::dataset, 20, 1000, &CreateNetwork, &TrainingCallBack);
    my_network->SetWeights(good_weights);
    std::cout << "MSR: " << my_network->GetMeanSquaredError(CarEvaluationData::dataset) << std::endl;

    std::cout << "End Of Program" << std::endl;
    std::cin.get();

    //Clean up
    delete my_network;
    return 0;
}