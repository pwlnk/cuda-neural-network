#pragma once

#include "nn_utils.hh"

namespace testutils {

	void initializeTensorWithValue(nn_utils::Tensor3D M, float value);
	void initializeTensorRandomlyInRange(nn_utils::Tensor3D M, float min, float max);

}
