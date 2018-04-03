#include "coordinates_dataset.hh"

CoordinatesDataset::CoordinatesDataset(size_t batch_size, size_t number_of_batches) :
	batch_size(batch_size), number_of_batches(number_of_batches)
{
	for (int i = 0; i < number_of_batches; i++) {
		batches.push_back(Matrix(Shape(batch_size, 2)));
		targets.push_back(Matrix(Shape(batch_size, 1)));

		batches[i].allocateCudaMemory();
		batches[i].allocateHostMemory();

		targets[i].allocateCudaMemory();
		targets[i].allocateHostMemory();

		for (int k = 0; k < batch_size; k++) {
			batches[i][k] = static_cast<float>(rand()) / RAND_MAX - 0.5;
			batches[i][batches[i].shape.x + k] = static_cast<float>(rand()) / RAND_MAX - 0.5;;
			targets[i][k] = batches[i][k] > 0 ? 1 : 0;
		}

		batches[i].copyHostToDevice();
		targets[i].copyHostToDevice();
	}
}

int CoordinatesDataset::getNumOfBatches() {
	return number_of_batches;
}

std::vector<Matrix>& CoordinatesDataset::getBatches() {
	return batches;
}

std::vector<Matrix>& CoordinatesDataset::getTargets() {
	return targets;
}
