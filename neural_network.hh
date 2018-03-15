#pragma once

#include <vector>
#include "nn_layer.hh"

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;

public:
	void addLayer(NNLayer *layer);
	float* forward(float* X);

};
