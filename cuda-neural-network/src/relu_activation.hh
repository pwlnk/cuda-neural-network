#pragma once

#include "nn_layer.hh"

class ReLUActivation : public NNLayer {
public:
	ReLUActivation(std::string name);
	~ReLUActivation();

	float* forward(float* A);
};
