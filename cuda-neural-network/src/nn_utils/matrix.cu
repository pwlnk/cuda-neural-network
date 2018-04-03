#include "matrix.hh"
#include "nn_exception.hh"

Matrix::Matrix(size_t x_dim, size_t y_dim) :
	shape(x_dim, y_dim), data_device(nullptr), data_host(nullptr),
	device_allocated(false), host_allocated(false)
{ }

Matrix::Matrix(Shape shape) :
	Matrix(shape.x, shape.y)
{ }

void Matrix::allocateCudaMemory() {
	if (!device_allocated) {
		cudaMalloc(&data_device, shape.x * shape.y * sizeof(float));
		NNException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
		device_allocated = true;
	}
}

void Matrix::allocateHostMemory() {
	if (!host_allocated) {
		data_host = new float[shape.x * shape.y];
		host_allocated = true;
	}
}

void Matrix::allocateIfNotAllocated(Shape shape) {
	if (!device_allocated) {
		this->shape = shape;
		allocateCudaMemory();
	}
}

void Matrix::freeCudaMemory() {
	if (device_allocated) {
		cudaFree(data_device);
		NNException::throwIfDeviceErrorsOccurred("Cannot free cuda memory.");
	}
	data_device = nullptr;
	device_allocated = false;
}

void Matrix::freeHostMemory() {
	if (host_allocated) {
		delete [] data_host;
	}
	data_host = nullptr;
	host_allocated = false;
}

void Matrix::freeCudaAndHostMemory() {
	freeCudaMemory();
	freeHostMemory();
}

void Matrix::copyHostToDevice() {
	if (device_allocated && host_allocated) {
		cudaMemcpy(data_device, data_host, shape.x * shape.y * sizeof(float), cudaMemcpyHostToDevice);
		NNException::throwIfDeviceErrorsOccurred("Cannot copy host data to CUDA device.");
	}
	else {
		throw NNException("Cannot copy host data to not allocated memory on device.");
	}
}

void Matrix::copyDeviceToHost() {
	if (device_allocated && host_allocated) {
		cudaMemcpy(data_host, data_device, shape.x * shape.y * sizeof(float), cudaMemcpyDeviceToHost);
		NNException::throwIfDeviceErrorsOccurred("Cannot copy device data to host.");
	}
	else {
		throw NNException("Cannot copy device data to not allocated memory on host.");
	}
}

float& Matrix::operator[](const int index) {
	return data_host[index];
}

const float& Matrix::operator[](const int index) const {
	return data_host[index];
}
