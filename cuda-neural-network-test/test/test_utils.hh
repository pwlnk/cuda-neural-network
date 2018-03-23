#pragma once

#include "nn_utils.hh"
#include <vector>

namespace testutils {

	void initializeTensorWithValue(nn_utils::Tensor3D M, float value);
	void initializeTensorRandomlyInRange(nn_utils::Tensor3D M, float min, float max);
	void initializeEachTensorRowWithValue(nn_utils::Tensor3D M, std::vector<float> values);
	void initializeEachTensorColWithValue(nn_utils::Tensor3D M, std::vector<float> values);

	float sigmoid(float x);

}
