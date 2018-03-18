#pragma once

#include "nn_layer.hh"

class ReLUActivation : public NNLayer {
private:
	nn_utils::Tensor3D Z;

public:
	ReLUActivation(std::string name);
	~ReLUActivation();

	nn_utils::Tensor3D forward(nn_utils::Tensor3D A);
};
