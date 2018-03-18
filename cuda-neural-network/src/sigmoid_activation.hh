#pragma once

#include "nn_layer.hh"

class SigmoidActivation : public NNLayer {
private:
	nn_utils::Tensor3D Z;

public:
	SigmoidActivation(std::string name);
	~SigmoidActivation();

	nn_utils::Tensor3D forward(nn_utils::Tensor3D A);
};
