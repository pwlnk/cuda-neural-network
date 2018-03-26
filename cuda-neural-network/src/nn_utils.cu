#include "nn_utils.hh"
#include "nn_exception.hh"

#include <math.h>
#include <iostream>
#include <assert.h>

__global__ void cross_entropy_cost(float* predictions, float* target,
										   int size, float* cost) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		float partial_cost = target[index] * logf(predictions[index])
				+ (1.0f - target[index]) * logf(1.0f - predictions[index]);
		atomicAdd(cost, - partial_cost / size);
	}
}

__global__ void d_cross_entropy_cost(float* predictions, float* target, float* dY,
								     int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		dY[index] =  (predictions[index] - target[index])
				/ (static_cast<double>(1.0f - predictions[index]) * predictions[index]);
	}
}

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

		float* cost;
		cudaMallocManaged(&cost, sizeof(float));
		*cost = 0.0f;

		dim3 block_size(256);
		dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
		cross_entropy_cost<<<num_of_blocks, block_size>>>(predictions.data_device, target.data_device,
															      predictions.shape.x, cost);
		cudaDeviceSynchronize();
		nn_utils::throwIfDeviceErrorsOccurred("Cannot compute binary cross entropy cost.");

		float cost_value = *cost;
		cudaFree(cost);

		return cost_value;
	}

	Tensor3D dBinaryCrossEntropyCost(Tensor3D predictions, Tensor3D target, Tensor3D dY) {
		assert(predictions.shape.x == target.shape.x);

		dim3 block_size(256);
		dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
		d_cross_entropy_cost<<<num_of_blocks, block_size>>>(predictions.data_device, target.data_device, dY.data_device,
															predictions.shape.x);
		cudaDeviceSynchronize();
		nn_utils::throwIfDeviceErrorsOccurred("Cannot compute derivative for binary cross entropy.");

		return dY;
	}

}
