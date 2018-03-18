#include <iostream>
#include <time.h>

#include "test_utils.hh"

namespace testutils {

	void initializeMatrixWithValue(float* M, int x_dim, int y_dim, float value) {
		for (int x = 0; x < x_dim; x++) {
			for (int y = 0; y < y_dim; y++) {
				M[y * x_dim + x] = value;
			}
		}
	}

	void initializeMatrixRandomlyInRange(float* M, int x_dim, int y_dim, float min, float max) {
		srand( time(NULL) );
		for (int x = 0; x < x_dim; x++) {
			for (int y = 0; y < y_dim; y++) {
				M[y * x_dim + x] = (static_cast<float>(rand()) / RAND_MAX) * (max - min) + min;
			}
		}
	}

}
