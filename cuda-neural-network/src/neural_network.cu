#include "neural_network.hh"

NeuralNetwork::NeuralNetwork()
{ }

NeuralNetwork::~NeuralNetwork() {
	Y.freeCudaMemory();
	for (std::vector<NNLayer*>::iterator it = this->layers.begin(); it != this->layers.end(); it++) {
		delete *it;
	}
}

void NeuralNetwork::addLayer(NNLayer* layer) {
	this->layers.push_back(layer);
}

nn_utils::Tensor3D NeuralNetwork::forward(nn_utils::Tensor3D X) {
	nn_utils::Tensor3D Z = X;

	for (std::vector<NNLayer*>::iterator it = this->layers.begin(); it != this->layers.end(); it++) {
		Z = (*it)->forward(Z);
	}

	Y = Z;
	return Y;
}

std::vector<NNLayer*> NeuralNetwork::getLayers() const {
	return layers;
}
