#pragma once

#include "matrix.hh"

namespace nn_utils {

	float binaryCrossEntropyCost(Matrix predictions, Matrix target);
	Matrix dBinaryCrossEntropyCost(Matrix predictions, Matrix target, Matrix dY);
}
