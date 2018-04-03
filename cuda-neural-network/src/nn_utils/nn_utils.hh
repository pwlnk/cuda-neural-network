#pragma once

#include "matrix.hh"

namespace nn_utils {

	void throwIfDeviceErrorsOccurred(const char* exception_message);

	float binaryCrossEntropyCost(Matrix predictions, Matrix target);
	Matrix dBinaryCrossEntropyCost(Matrix predictions, Matrix target, Matrix dY);
}
