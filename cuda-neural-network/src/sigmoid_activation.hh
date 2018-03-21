#pragma once

#include "nn_layer.hh"

class SigmoidActivation : public NNLayer {
private:
	nn_utils::Tensor3D A;

	nn_utils::Tensor3D Z;
	nn_utils::Tensor3D dZ;

public:
	SigmoidActivation(std::string name);
	~SigmoidActivation();

	nn_utils::Tensor3D forward(nn_utils::Tensor3D Z);
	nn_utils::Tensor3D backprop(nn_utils::Tensor3D dA);
};
