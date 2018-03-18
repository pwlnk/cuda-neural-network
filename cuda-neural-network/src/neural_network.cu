#include "neural_network.hh"

void NeuralNetwork::addLayer(NNLayer* layer) {
	this->layers.push_back(layer);
}

float* NeuralNetwork::forward(float* X) {

	for (std::vector<NNLayer*>::iterator it = this->layers.begin(); it != this->layers.end(); it++) {
		(*it)->forward(nullptr, 0, 0);
	}

	return nullptr;
}
