#include "relu_activation.hh"
#include "nn_utils.hh"

__global__ void relu_activation_forward(float* A, float* Z,
									    int A_x_dim, int A_y_dim) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < A_x_dim * A_y_dim) {
		if (A[index] > 0) {
			Z[index] = A[index];
		}
		else {
			Z[index] = 0;
		}
	}
}

ReLUActivation::ReLUActivation(std::string name) {
	this->name = name;
}

ReLUActivation::~ReLUActivation() { }

float* ReLUActivation::forward(float* A, int A_x_dim, int A_y_dim) {
	// TODO: should be initialized only once, not with every forward() call
	cudaMallocManaged(&Z, A_x_dim * A_y_dim * sizeof(float));

	dim3 block_size(256);
	dim3 num_of_blocks((A_y_dim * A_x_dim + block_size.x - 1) / block_size.x);

	relu_activation_forward<<<block_size, num_of_blocks>>>(A, Z,
														   A_x_dim, A_y_dim);
	cudaDeviceSynchronize();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot perform relu forward prop.");

	return Z;
}
