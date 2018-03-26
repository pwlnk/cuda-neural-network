#include <iostream>
#include <time.h>
#include <math.h>
#include <assert.h>

#include "test_utils.hh"

namespace testutils {

	void initializeTensorWithValue(nn_utils::Tensor3D M, float value) {
		for (int x = 0; x < M.shape.x; x++) {
			for (int y = 0; y < M.shape.y; y++) {
				M[y * M.shape.x + x] = value;
			}
		}
	}

	void initializeTensorRandomlyInRange(nn_utils::Tensor3D M, float min, float max) {
		srand( time(NULL) );
		for (int x = 0; x < M.shape.x; x++) {
			for (int y = 0; y < M.shape.y; y++) {
				M[y * M.shape.x + x] = (static_cast<float>(rand()) / RAND_MAX) * (max - min) + min;
			}
		}
	}

	void initializeEachTensorRowWithValue(nn_utils::Tensor3D M, std::vector<float> values) {
		assert(M.shape.y == values.size());

		for (int x = 0; x < M.shape.x; x++) {
			for (int y = 0; y < M.shape.y; y++) {
				M[y * M.shape.x + x] = values[y];
			}
		}
	}

	void initializeEachTensorColWithValue(nn_utils::Tensor3D M, std::vector<float> values) {
		assert(M.shape.x == values.size());

		for (int x = 0; x < M.shape.x; x++) {
			for (int y = 0; y < M.shape.y; y++) {
				M[y * M.shape.x + x] = values[x];
			}
		}
	}

	float sigmoid(float x) {
		return exp(x) / (1 + exp(x));
	}

}
