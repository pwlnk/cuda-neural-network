#include "sigmoid_activation.hh"
#include "nn_utils.hh"

__device__ float sigmoid(float x) {
	return exp(x) / (1 + exp(x));
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

SigmoidActivation::SigmoidActivation(std::string name) :
	A(), dZ()
{
	this->name = name;
}

SigmoidActivation::~SigmoidActivation() {
	A.freeCudaMemory();
	dZ.freeCudaMemory();
}

nn_utils::Tensor3D SigmoidActivation::forward(nn_utils::Tensor3D Z) {

	this->Z = Z;

	// TODO: should be allocated once, not every time forward is called
	A.shape = Z.shape;
	A.allocateCudaMemory();

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	sigmoid_activation_forward<<<block_size, num_of_blocks>>>(Z.data, A.data,
														   	  Z.shape.x, Z.shape.y);
	cudaDeviceSynchronize();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot perform sigmoid forward prop.");

	return A;
}

nn_utils::Tensor3D SigmoidActivation::backprop(nn_utils::Tensor3D dA) {
	// TODO: should be allocated once, not every time backprop is called
	dZ.shape = Z.shape;
	dZ.allocateCudaMemory();

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	sigmoid_activation_backprop<<<block_size, num_of_blocks>>>(Z.data, dA.data, dZ.data,
															   Z.shape.x, Z.shape.y);
	cudaDeviceSynchronize();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot perform sigmoid backprop");

	return dZ;
}
