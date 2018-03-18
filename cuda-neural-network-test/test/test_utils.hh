#pragma once

namespace testutils {

	void initializeMatrixWithValue(float* M, int x_dim, int y_dim, float value);
	void initializeMatrixRandomlyInRange(float* M, int x_dim, int y_dim, float min, float max);

}
