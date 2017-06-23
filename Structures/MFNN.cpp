NN::MFNN::MFNN(
    std::vector<uint32_t>                       neurons_per_layer,
    std::vector<Activation::ActivationType>     layer_activations
){
    if (neurons_per_layer.size() < 2)
    {
        throw std::runtime_error("ERROR [MFNN]: There must be atleast 2 layers in the MFNN.");
    }
    if (layer_activations.size() != neurons_per_layer.size())
    {
        throw std::runtime_error("ERROR [MFNN]: Layer activation size and neurons per layer size does not match.");
    }

    layers.resize(neurons_per_layer.size());
    
    for (uint32_t i = 0; i < neurons_per_layer.size(); i++)
    {
        if (neurons_per_layer[i] <= 0)
        {
            throw std::runtime_error("ERROR [MFNN]: A layer must contain atleast 1 neuron.");
        }

        //Setup Layers
        layers[i] = new Layer();

        //Setup Neurons
        layers[i]->neurons.resize(neurons_per_layer[i]);
        for (uint32_t j = 0; j < neurons_per_layer[i]; j++)
        {
            layers[i]->neurons[j] = new Neuron();
            layers[i]->neurons[j]->activation_type = layer_activations[i];
        }
        //Setup Synapsis
        if (i + 1 < neurons_per_layer.size())
        {
            layers[i]->synapsis.resize(neurons_per_layer[i] * neurons_per_layer[i + 1]);
            for (uint32_t j = 0; j < neurons_per_layer[i + 1]; j++)
            {
                for (uint32_t k = 0; k < neurons_per_layer[i]; k++)
                {
                    uint32_t convert_id = (j * neurons_per_layer[i]) + k;
                    layers[i]->synapsis[convert_id] = new Synapse();
                }
            }
        }
    }

    //Connect Neurons & Synapsis
    for (uint32_t i = 0; i < neurons_per_layer.size() - 1; i++)
    {
        for (uint32_t j = 0; j < neurons_per_layer[i + 1]; j++)
        {
            for (uint32_t k = 0; k < neurons_per_layer[i]; k++)
            {
                //This is a formula used to order the synapsis so a specfic synapse can be
                //extracted later on in the process.
                uint32_t convert_id = (j * neurons_per_layer[i]) + k;
                //Set synapsis pointers and neuron pointers to correct values
                //making sure they can access each others values and are connected.
                layers[i]->synapsis[convert_id]->connected_to_neuron    = layers[i+1]->neurons[j];
                layers[i]->synapsis[convert_id]->connected_from_neuron  = layers[i]->neurons[k];
                
                layers[i]->neurons[k]->synapsis_out.push_back(layers[i]->synapsis[convert_id]);
                layers[i+1]->neurons[j]->synapsis_in.push_back(layers[i]->synapsis[convert_id]);
            }
        }
    }

    SetRandomWeights();
}
NN::MFNN::~MFNN()
{
    for (uint32_t i = 0; i < layers.size(); i++)
    {
        for (uint32_t j = 0; j < layers[i]->neurons.size(); j++)
            delete layers[i]->neurons[j];

        for (uint32_t j = 0; j < layers[i]->synapsis.size(); j++)
            delete layers[i]->synapsis[j];

        delete layers[i];
    }
}

std::vector<double> NN::MFNN::Compute(
    std::vector<double> inputs
)
{
    //Error Checking
    if (layers[0]->neurons.size() != inputs.size())
    {
        throw std::runtime_error("ERROR [Compute]: inputs size does not match the network.");
    }
    //Setup Input Layer
    for (uint32_t i = 0; i < layers[0]->neurons.size(); i++)
    {
        layers[0]->neurons[i]->neuron_value = inputs[i];
    }

    for (uint32_t i = 1; i < layers.size(); i++)
    {
        for (uint32_t j = 0; j < layers[i]->neurons.size(); j++)
        {
            double sum = 0.0;
            for (uint32_t k = 0; k < layers[i]->neurons[j]->synapsis_in.size(); k++)
            {
                sum += layers[i]->neurons[j]->synapsis_in[k]->synapse_weight * layers[i]->neurons[j]->synapsis_in[k]->connected_from_neuron->neuron_value;
            }
            sum += layers[i]->neurons[j]->bias_value;
            sum = Activation::ApplyActivation(sum, layers[i]->neurons[j]->activation_type);

            layers[i]->neurons[j]->neuron_value = sum;
        }
    }

    return GetOutput();
}

double NN::MFNN::GetMeanSquaredError(
    std::vector<std::vector<double> > dataset
){
    //Error Checking
    if (dataset.size() <= 0)
        throw std::runtime_error("ERROR [GetMeanSquaredError]: Could not locate dataset.");

    //Get the size of input & output layer to ensure dataset matches it.
    uint32_t    input_layer_size    = layers[0]->neurons.size();
    uint32_t    output_layer_size   = layers[layers.size() - 1]->neurons.size();
    if (dataset[0].size() != input_layer_size + output_layer_size)
        throw std::runtime_error("ERROR [GetMeanSquaredError]: dataset does not match neural network");

    //Setup input and desired output variables.
    std::vector<double> xValues(input_layer_size); // Inputs
	std::vector<double> tValues(output_layer_size); //Outputs

    //Go through each training data sent from the "dataset" parameter
    //Set sum squared error to 0
	double sum_squared_error = 0.0;
	for (uint32_t i = 0; i < dataset.size(); ++i)
	{
		//Extract data from "dataset" parameter and store it in the xValues/tValues variables.
		std::copy(dataset[i].begin(), dataset[i].begin() + input_layer_size, xValues.begin());
		std::copy(dataset[i].begin() + input_layer_size, dataset[i].begin() + input_layer_size + output_layer_size, tValues.begin());

        //Get the output value computed by the neural network
		std::vector<double> yValues = Compute(xValues);
        //Go through each node and and add the (computed value - desired value)^2
        //to the sum squared error
		for (uint32_t j = 0; j < yValues.size(); ++j)
			sum_squared_error += ((yValues[j] - tValues[j]) * (yValues[j] - tValues[j]));
	}

    //Return the sum squared error
	return sum_squared_error;
}

void NN::MFNN::SetRandomWeights()
{
    std::srand(std::time(NULL));
    std::vector<double> new_weights;
    double high = +0.1;
    double low  = -0.1;
    for (uint32_t i = 0; i < layers.size(); i++)
    {
        for (uint32_t j = 0; j < layers[i]->neurons.size(); j++)
        {
            double r = (high - low) * ((double)std::rand() / (double)RAND_MAX) + low;
            new_weights.push_back(r);
        }
        for (uint32_t j = 0; j < layers[i]->synapsis.size(); j++)
        {
            double r = (high - low) * ((double)std::rand() / (double)RAND_MAX) + low;
            new_weights.push_back(r);
        }
    }
    SetWeights(new_weights);
}
std::vector<double> NN::MFNN::GetWeights()
{
    std::vector<double> result;
    for (uint32_t i = 0; i < layers.size(); i++)
    {
        for (uint32_t j = 0; j < layers[i]->neurons.size(); j++)
        {
            result.push_back( layers[i]->neurons[j]->bias_value );
        }
        for (uint32_t j = 0; j < layers[i]->synapsis.size(); j++)
        {
            result.push_back( layers[i]->synapsis[j]->synapse_weight );
        }
    }

    return result;
}
void NN::MFNN::SetWeights(
    std::vector<double> weights
){
    uint32_t k = 0;
    for (uint32_t i = 0; i < layers.size(); i++)
    {
        for (uint32_t j = 0; j < layers[i]->neurons.size(); j++)
        {
            layers[i]->neurons[j]->bias_value       = weights[k];
            k++;
        }
        for (uint32_t j = 0; j < layers[i]->synapsis.size(); j++)
        {
            layers[i]->synapsis[j]->synapse_weight  = weights[k];
            k++;
        }
    }
}

NN::MFNN::Layer* NN::MFNN::GetLayer(uint32_t layer_number)
{
    return layers[layer_number];
}
std::vector<double> NN::MFNN::GetOutput()
{
    std::vector<double> result;
    uint32_t output_layer       = layers.size() - 1;
    uint32_t output_layer_size  = layers[output_layer]->neurons.size();
    result.resize(output_layer_size);
    for (uint32_t i = 0; i < output_layer_size; i++)
    {
        result[i] = layers[output_layer]->neurons[i]->neuron_value;
    }

    return result;
}