#include "neural_network.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/nn_utils.hh"

#include <iostream>

NeuralNetwork::NeuralNetwork()
{ }

NeuralNetwork::~NeuralNetwork() {
	for (std::vector<NNLayer*>::iterator it = this->layers.begin(); it != this->layers.end(); it++) {
		delete *it;
	}

	dY.freeCudaAndHostMemory();
}

void NeuralNetwork::addLayer(NNLayer* layer) {
	this->layers.push_back(layer);
}

Matrix NeuralNetwork::forward(Matrix X) {
	Matrix Z = X;

	for (std::vector<NNLayer*>::iterator it = this->layers.begin(); it != this->layers.end(); it++) {
		Z = (*it)->forward(Z);
	}

	Y = Z;
	return Y;
}

void NeuralNetwork::backprop(Matrix predictions, Matrix target) {

	dY.allocateIfNotAllocated(predictions.shape);

	Matrix err = nn_utils::dBinaryCrossEntropyCost(predictions, target, dY);

	for (std::vector<NNLayer*>::reverse_iterator it = this->layers.rbegin(); it != this->layers.rend(); it++) {
		err = (*it)->backprop(err, 0.01);
	}
}

std::vector<NNLayer*> NeuralNetwork::getLayers() const {
	return layers;
}
