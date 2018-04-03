#pragma once

#include "nn_utils.hh"

#include <vector>

class CoordinatesDataset {
private:
	size_t batch_size;
	size_t number_of_batches;

	std::vector<nn_utils::Tensor3D> batches;
	std::vector<nn_utils::Tensor3D> targets;

public:

	CoordinatesDataset(size_t batch_size, size_t number_of_batches);

	int getNumOfBatches();
	std::vector<nn_utils::Tensor3D>& getBatches();
	std::vector<nn_utils::Tensor3D>& getTargets();

};
