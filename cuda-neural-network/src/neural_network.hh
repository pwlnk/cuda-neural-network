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

	nn_utils::Tensor3D forward(nn_utils::Tensor3D X);
	float binaryCrossEntropyCost(nn_utils::Tensor3D predictions, nn_utils::Tensor3D target);
	float dBinaryCrossEntropyCost(nn_utils::Tensor3D predictions, nn_utils::Tensor3D target);

	void addLayer(NNLayer *layer);
	std::vector<NNLayer*> getLayers() const;

};
