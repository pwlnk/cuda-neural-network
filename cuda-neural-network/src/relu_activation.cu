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

ReLUActivation::ReLUActivation(std::string name) :
		Z()
{
	this->name = name;
}

ReLUActivation::~ReLUActivation() {
	Z.freeCudaMemory();
}

nn_utils::Tensor3D ReLUActivation::forward(nn_utils::Tensor3D A) {

	// TODO: should be allocated once, not every time forward is called
	Z.shape = A.shape;
	Z.allocateCudaMemory();

	dim3 block_size(256);
	dim3 num_of_blocks((A.shape.y * A.shape.x + block_size.x - 1) / block_size.x);

	relu_activation_forward<<<block_size, num_of_blocks>>>(A.data, Z.data,
														   A.shape.x, A.shape.y);
	cudaDeviceSynchronize();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot perform ReLU forward prop.");

	return Z;
}
