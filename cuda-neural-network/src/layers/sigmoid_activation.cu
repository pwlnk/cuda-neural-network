#include "sigmoid_activation.hh"
#include "../nn_utils/nn_exception.hh"
#include <iostream>

__device__ float sigmoid(float x) {
	return 1.0f / (1 + exp(-x));
}

__global__ void sigmoid_activation_forward(float* Z, float* A,
										   int Z_x_dim, int Z_y_dim) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		A[index] = sigmoid(Z[index]);
	}
}

__global__ void sigmoid_activation_backprop(float* Z, float* dA, float* dZ,
											int Z_x_dim, int Z_y_dim) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		dZ[index] = dA[index] * sigmoid(Z[index]) * (1 - sigmoid(Z[index]));
	}
}

SigmoidActivation::SigmoidActivation(std::string name)
{
	this->name = name;
}

SigmoidActivation::~SigmoidActivation() {
	A.freeCudaAndHostMemory();
	dZ.freeCudaAndHostMemory();
}

Matrix SigmoidActivation::forward(Matrix Z) {

	this->Z = Z;
	A.allocateIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	sigmoid_activation_forward<<<num_of_blocks, block_size>>>(Z.data_device, A.data_device,
														   	  Z.shape.x, Z.shape.y);
	cudaDeviceSynchronize();
	NNException::throwIfDeviceErrorsOccurred("Cannot perform sigmoid forward prop.");

	return A;
}

Matrix SigmoidActivation::backprop(Matrix dA, float learning_rate) {

	dZ.allocateIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	sigmoid_activation_backprop<<<num_of_blocks, block_size>>>(Z.data_device, dA.data_device, dZ.data_device,
															   Z.shape.x, Z.shape.y);
	cudaDeviceSynchronize();
	NNException::throwIfDeviceErrorsOccurred("Cannot perform sigmoid backprop");

	return dZ;
}
