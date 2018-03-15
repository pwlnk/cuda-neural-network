#pragma once

#include "nn_layer.hh"

class LinearLayer : public NNLayer {
public:
	LinearLayer(std::string name);
	~LinearLayer();

	float* forward(float* A);
};
