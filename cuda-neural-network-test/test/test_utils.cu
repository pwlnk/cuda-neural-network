#include <iostream>
#include <time.h>

#include "test_utils.hh"

namespace testutils {

	void initializeTensorWithValue(nn_utils::Tensor3D M, float value) {
		for (int x = 0; x < M.shape.x; x++) {
			for (int y = 0; y < M.shape.y; y++) {
				M.data[y * M.shape.x + x] = value;
			}
		}
	}

	void initializeTensorRandomlyInRange(nn_utils::Tensor3D M, float min, float max) {
		srand( time(NULL) );
		for (int x = 0; x < M.shape.x; x++) {
			for (int y = 0; y < M.shape.y; y++) {
				M.data[y * M.shape.x + x] = (static_cast<float>(rand()) / RAND_MAX) * (max - min) + min;
			}
		}
	}

}
