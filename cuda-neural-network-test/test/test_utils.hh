#pragma once

#include "nn_utils/matrix.hh"
#include <vector>

namespace testutils {

	void initializeTensorWithValue(Matrix M, float value);
	void initializeTensorRandomlyInRange(Matrix M, float min, float max);
	void initializeEachTensorRowWithValue(Matrix M, std::vector<float> values);
	void initializeEachTensorColWithValue(Matrix M, std::vector<float> values);

	float sigmoid(float x);

}
