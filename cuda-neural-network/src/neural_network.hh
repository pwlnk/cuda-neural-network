#pragma once

#include <vector>
#include "nn_layer.hh"
#include "nn_utils/bce_cost.hh"

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;
	BCECost bce_cost;

	Matrix Y;
	Matrix dY;

public:
	NeuralNetwork();
	~NeuralNetwork();

	Matrix forward(Matrix X);
	void backprop(Matrix predictions, Matrix target);

	float binaryCrossEntropyCost(Matrix predictions, Matrix target);
	float dBinaryCrossEntropyCost(Matrix predictions, Matrix target);

	void addLayer(NNLayer *layer);
	std::vector<NNLayer*> getLayers() const;

};
