#include "nn_utils.hh"
#include "nn_exception.hh"

#include <math.h>
#include <iostream>
#include <assert.h>

namespace nn_utils {

	void throwIfDeviceErrorsOccurred(const char* exception_message) {
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			std::cerr << error << ": " << exception_message;
			throw NNException(exception_message);
		}
	}

	Shape::Shape(size_t x, size_t y, size_t z) :
		x(x), y(y), z(z)
	{ }

	Tensor3D::Tensor3D(size_t x_dim, size_t y_dim, size_t z_dim) :
		shape(x_dim, y_dim, z_dim), data_device(nullptr), data_host(nullptr),
		device_allocated(false), host_allocated(false)
	{ }

	Tensor3D::Tensor3D(Shape shape) :
		Tensor3D(shape.x, shape.y, shape.z)
	{ }

	void Tensor3D::allocateCudaMemory() {
		if (!device_allocated) {
			cudaMalloc(&data_device, shape.x * shape.y * shape.z * sizeof(float));
			throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
			device_allocated = true;
		}
	}

	void Tensor3D::allocateHostMemory() {
		if (!host_allocated) {
			data_host = new float[shape.x * shape.y];
			host_allocated = true;
		}
	}

	void Tensor3D::allocateIfNotAllocated(nn_utils::Shape shape) {
		if (!device_allocated) {
			this->shape = shape;
			allocateCudaMemory();
		}
	}

	void Tensor3D::freeCudaMemory() {
		if (device_allocated) {
			cudaFree(data_device);
			throwIfDeviceErrorsOccurred("Cannot free cuda memory.");
		}
		data_device = nullptr;
		device_allocated = false;
	}

	void Tensor3D::freeHostMemory() {
		if (host_allocated) {
			delete [] data_host;
		}
		data_host = nullptr;
		host_allocated = false;
	}

	void Tensor3D::freeCudaAndHostMemory() {
		freeCudaMemory();
		freeHostMemory();
	}

	void Tensor3D::copyHostToDevice() {
		if (device_allocated && host_allocated) {
			cudaMemcpy(data_device, data_host, shape.x * shape.y * shape.z * sizeof(float), cudaMemcpyHostToDevice);
			throwIfDeviceErrorsOccurred("Cannot copy host data to CUDA device.");
		}
		else {
			throw NNException("Cannot copy host data to not allocated memory on device.");
		}
	}

	void Tensor3D::copyDeviceToHost() {
		if (device_allocated && host_allocated) {
			cudaMemcpy(data_host, data_device, shape.x * shape.y * shape.z * sizeof(float), cudaMemcpyDeviceToHost);
			throwIfDeviceErrorsOccurred("Cannot copy device data to host.");
		}
		else {
			throw NNException("Cannot copy device data to not allocated memory on host.");
		}
	}

	float& Tensor3D::operator[](const int index) {
		return data_host[index];
	}

	const float& Tensor3D::operator[](const int index) const {
		return data_host[index];
	}

	float binaryCrossEntropyCost(nn_utils::Tensor3D predictions, nn_utils::Tensor3D target) {
		assert(predictions.shape.x == target.shape.x);

		predictions.allocateHostMemory();
		predictions.copyDeviceToHost();
		target.allocateHostMemory();
		target.copyDeviceToHost();

		float cost = 0.0;
		for (int i = 0; i < predictions.shape.x; i++) {
			cost += target[i] * log(predictions[i]) + (1 - target[i]) * log(1 - predictions[i]);
		}

		return -cost / predictions.shape.x;
	}

	// TODO: move operation to CUDA
	Tensor3D dBinaryCrossEntropyCost(Tensor3D predictions, Tensor3D target, Tensor3D dY) {
		assert(predictions.shape.x == target.shape.x);

		dY.allocateIfNotAllocated(predictions.shape);
		dY.allocateHostMemory();


		predictions.allocateHostMemory();
		predictions.copyDeviceToHost();
		target.allocateHostMemory();
		target.copyDeviceToHost();

		for (int i = 0; i < predictions.shape.x; i++) {
			// TODO: what sign should be here + or - ?
			dY[i] =  (predictions[i] - target[i]) / (static_cast<double>(1 - predictions[i]) * predictions[i]);
		}

		dY.copyHostToDevice();
		return dY;
	}

}
