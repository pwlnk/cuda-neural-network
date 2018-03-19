#include "nn_utils.hh"
#include "nn_exception.hh"

#include <math.h>

namespace nn_utils {

	void throwIfDeviceErrorsOccurred(const char* exception_message) {
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			throw NNException(exception_message);
		}
	}

	float binaryCrossEntropyLoss(Tensor3D predictions, Tensor3D target) {
		if (predictions.shape.x != target.shape.x) {
			throw NNException("Predictions and target shapes don't match.");
		}

		float cost = 0.0;
		for (int i = 0; i < predictions.shape.x; i++) {
			cost += target.data[i] * log(predictions.data[i]) + (1 - target.data[i]) * log(1 - predictions.data[i]);
		}

		return -cost / predictions.shape.x;
	}

	Shape::Shape(size_t x, size_t y, size_t z) :
		x(x), y(y), z(z)
	{ }

	Tensor3D::Tensor3D(size_t x_dim, size_t y_dim, size_t z_dim) :
		shape(x_dim, y_dim, z_dim), data(nullptr)
	{ }

	Tensor3D::Tensor3D(Shape shape) :
		shape(shape), data(nullptr)
	{ }

	void Tensor3D::allocateCudaMemory() {
		cudaMallocManaged(&data, shape.x * shape.y * shape.z * sizeof(float));
		throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
	}

	void Tensor3D::freeCudaMemory() {
		cudaFree(data);
		data = nullptr;
	}

}
