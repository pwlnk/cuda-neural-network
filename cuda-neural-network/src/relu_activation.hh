#pragma once

#include "nn_layer.hh"

class ReLUActivation : public NNLayer {
private:
	nn_utils::Tensor3D A;

	nn_utils::Tensor3D Z;
	nn_utils::Tensor3D dZ;

public:
	ReLUActivation(std::string name);
	~ReLUActivation();

	nn_utils::Tensor3D forward(nn_utils::Tensor3D Z);
	nn_utils::Tensor3D backprop(nn_utils::Tensor3D dA, float learning_rate = 0.01);
};
