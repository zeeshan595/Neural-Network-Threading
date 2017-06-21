std::vector<uint32_t> NN::Shuffle(
    std::vector<uint32_t> sequence
){
    std::vector<uint32_t> result = sequence;
	std::srand(std::time(0));
	for (unsigned int i = 0; i < result.size(); ++i)
	{
		int r = std::rand() % result.size();
		int tmp = result[r];
		result[r] = result[i];
		result[i] = tmp;
	}
	return result;
}

void NN::PrintDataSet(
    std::vector<std::vector<double> > dataset,
    uint32_t tab
){
    for (uint32_t i = 0; i < dataset.size(); i++)
    {
        std::cout << "{ ";
        for (uint32_t j = 0; j < dataset[i].size(); j++)
        {
            std::cout << std::setw(tab) << dataset[i][j] << ", ";
        }
        std::cout << "}," << std::endl;
    }
}

void NN::Normalize(
    std::vector<std::vector<double> > &dataset,
    std::vector<uint32_t> cols
){
    for (uint32_t col = 0; col < cols.size(); col++)
    {
        uint32_t i = cols[col];
        double sum = 0.0;
        for (int j = 0; j < dataset.size(); j++)
            sum += dataset[j][i];

        double mean = sum / dataset.size();
        sum = 0.0;

        for (int j = 0; j < dataset.size(); j++)
            sum += (dataset[j][i] - mean) * (dataset[j][i] - mean);

        double sd = sqrt(sum / (dataset.size() - 1));

        for (int j = 0; j < dataset.size(); j++)
            dataset[j][i] = (dataset[j][i] - mean) / sd;
    }
}

void NN::Randomize(
    std::vector<std::vector<double> > &dataset
){
    //Use shuffle function to create a randomly ordered index for
    //the training data.
    std::vector<uint32_t> sequence(dataset.size());
    for (uint32_t i = 0; i < dataset.size(); i++)
        sequence[i] = i;
    sequence = Shuffle(sequence);

    //Use the randomly created index to store randomly ordered train data
    //into new_dataset variable
    std::vector<std::vector<double> > new_dataset(dataset.size());
    for (uint32_t i = 0; i < dataset.size(); i++)
    {
        new_dataset[i] = dataset[sequence[i]];
    }

    //Update the training data.
    dataset = new_dataset;
}