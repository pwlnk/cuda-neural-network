#include "relu_activation.hh"
#include "../nn_utils/nn_exception.hh"

__global__ void reluActivationForward(float* Z, float* A,
									  int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		A[index] = fmaxf(Z[index], 0);
	}
}

__global__ void reluActivationBackprop(float* Z, float* dA, float* dZ,
									   int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		if (Z[index] > 0) {
			dZ[index] = dA[index];
		}
		else {
			dZ[index] = 0;
		}
	}
}

ReLUActivation::ReLUActivation(std::string name) {
	this->name = name;
}

ReLUActivation::~ReLUActivation() { }

Matrix& ReLUActivation::forward(Matrix& Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	reluActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(),
														 Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform ReLU forward propagation.");

	return A;
}

Matrix& ReLUActivation::backprop(Matrix& dA, float learning_rate) {
	dZ.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	reluActivationBackprop<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(),
													      dZ.data_device.get(),
														  Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform ReLU back propagation");

	return dZ;
}
