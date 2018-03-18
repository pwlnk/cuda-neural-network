#pragma once

#include "nn_layer.hh"

class ReLUActivation : public NNLayer {
private:
	float* Z;

public:
	ReLUActivation(std::string name);
	~ReLUActivation();

	float* forward(float* A, int A_x_dim, int A_y_dim);
};
