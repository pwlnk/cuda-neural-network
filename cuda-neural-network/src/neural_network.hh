#pragma once

#include <vector>
#include "nn_layer.hh"

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;
	nn_utils::Tensor3D Y;

public:
	NeuralNetwork();
	~NeuralNetwork();

	void addLayer(NNLayer *layer);
	nn_utils::Tensor3D forward(nn_utils::Tensor3D X);
	std::vector<NNLayer*> getLayers() const;

};
