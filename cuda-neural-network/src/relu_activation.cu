#include "relu_activation.hh"
#include "nn_utils.hh"

__global__ void relu_activation_forward(float* Z, float* A,
									    int Z_x_dim, int Z_y_dim) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		A[index] = fmaxf(Z[index], 0);
	}
}

__global__ void relu_activation_backprop(float* Z, float* dA, float* dZ,
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

ReLUActivation::ReLUActivation(std::string name) :
		A(), dZ()
{
	this->name = name;
}

ReLUActivation::~ReLUActivation() {
	A.freeCudaMemory();
	dZ.freeCudaMemory();
}

nn_utils::Tensor3D ReLUActivation::forward(nn_utils::Tensor3D Z) {

	this->Z = Z;

	// TODO: should be allocated once, not every time forward is called
	A.shape = Z.shape;
	A.allocateCudaMemory();

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	relu_activation_forward<<<block_size, num_of_blocks>>>(Z.data, A.data,
														   Z.shape.x, Z.shape.y);
	cudaDeviceSynchronize();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot perform ReLU forward prop.");

	return A;
}

nn_utils::Tensor3D ReLUActivation::backprop(nn_utils::Tensor3D dA, float learning_rate) {
	// TODO: should be allocated once, not every time forward is called
	dZ.shape = Z.shape;
	dZ.allocateCudaMemory();

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	relu_activation_backprop<<<block_size, num_of_blocks>>>(Z.data, dA.data, dZ.data,
															Z.shape.x, Z.shape.y);
	cudaDeviceSynchronize();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot perform relu backprop");

	return dZ;
}
