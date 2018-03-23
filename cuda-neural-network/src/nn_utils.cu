#include "nn_utils.hh"
#include "nn_exception.hh"

#include <math.h>
#include <iostream>
#include <assert.h>

namespace nn_utils {

	void throwIfDeviceErrorsOccurred(const char* exception_message) {
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			throw NNException(exception_message);
		}
	}

	Shape::Shape(size_t x, size_t y, size_t z) :
		x(x), y(y), z(z)
	{ }

	Tensor3D::Tensor3D(size_t x_dim, size_t y_dim, size_t z_dim) :
		shape(x_dim, y_dim, z_dim), data(nullptr), memory_allocated(false)
	{ }

	Tensor3D::Tensor3D(Shape shape) :
		shape(shape), data(nullptr), memory_allocated(false)
	{ }

	void Tensor3D::allocateCudaMemory() {
		cudaMallocManaged(&data, shape.x * shape.y * shape.z * sizeof(float));
		throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
		memory_allocated = true;
	}

	void Tensor3D::allocateIfNotAllocated(nn_utils::Shape shape) {
		if (!memory_allocated) {
			this->shape = shape;
			allocateCudaMemory();
		}
	}

	void Tensor3D::freeCudaMemory() {
		if (memory_allocated) {
			cudaFree(data);
		}
		data = nullptr;
		memory_allocated = false;
	}

	float& Tensor3D::operator[](const int index) {
		return data[index];
	}

	const float& Tensor3D::operator[](const int index) const {
		return data[index];
	}

	float binaryCrossEntropyCost(nn_utils::Tensor3D predictions, nn_utils::Tensor3D target) {
		assert(predictions.shape.x == target.shape.x);

		float cost = 0.0;
		for (int i = 0; i < predictions.shape.x; i++) {
			cost += target.data[i] * log(predictions.data[i]) + (1 - target.data[i]) * log(1 - predictions.data[i]);
		}

		return -cost / predictions.shape.x;
	}

	// TODO: move operation to CUDA
	Tensor3D dBinaryCrossEntropyCost(Tensor3D predictions, Tensor3D target, Tensor3D dY) {
		assert(predictions.shape.x == target.shape.x);

		dY.allocateIfNotAllocated(predictions.shape);

		for (int i = 0; i < predictions.shape.x; i++) {
			// TODO: what sign should be here + or - ?
			dY.data[i] =  (predictions.data[i] - target.data[i]) / (static_cast<double>(1 - predictions.data[i]) * predictions.data[i]);
		}

		return dY;
	}

}
