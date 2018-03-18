#include "nn_utils.hh"
#include "nn_exception.hh"

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
		shape(x_dim, y_dim, z_dim), data(nullptr)
	{ }

	Tensor3D::Tensor3D(Shape shape) :
		shape(shape)
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
