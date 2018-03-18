#include <stdlib.h>

#include "linear_layer.hh"
#include "nn_exception.hh"
#include "nn_utils.hh"

__global__ void linear_layer_forward(float* A, float* W, float* Z,
									   int A_x_dim, int A_y_dim,
									   int W_x_dim, int W_y_dim) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < W_x_dim * W_y_dim) {
		int W_x = index % W_x_dim;
		int W_y = index / W_x_dim;

		int Z_x_dim = A_x_dim;
		int Z_y = W_y;
		int Z_x = 0;

		int A_y = W_x;
		int product_val = 0;

		for (int A_x = 0; A_x < A_x_dim; A_x++) {
			Z_x = A_x;
			product_val = W[index] * A[A_y * A_x_dim + A_x];
			atomicAdd(&Z[Z_y * Z_x_dim + Z_x], product_val);
		}
	}
}

LinearLayer::LinearLayer(std::string name, int x_dim, int y_dim) :
	x_dim(x_dim), y_dim(y_dim)
{
	this->name = name;
	allocateWeightsMemory();
	initializeWeightsRandomly();
}

void LinearLayer::allocateWeightsMemory() {
	cudaError_t error = cudaMallocManaged(&W, x_dim * y_dim * sizeof(float));
	if (error != cudaSuccess) {
		std::cout << error << std::endl;
		throw NNException("Cannot initialize layer weights.");
	}
}

void LinearLayer::initializeWeightsRandomly() {
	for (int x = 0; x < x_dim; x++) {
		for (int y = 0; y < y_dim; y++) {
			W[y * x_dim + x] = (static_cast<float>(rand()) / RAND_MAX) * weights_init_threshold;
		}
	}
}

LinearLayer::~LinearLayer() { }

float* LinearLayer::forward(float* A, int A_x_dim, int A_y_dim) {

	// TODO: should be initialized only once, not with every forward() call
	cudaMallocManaged(&Z, A_x_dim * A_y_dim * sizeof(float));

	dim3 block_size(256);
	dim3 num_of_blocks((y_dim * x_dim + block_size.x - 1) / block_size.x);

	linear_layer_forward<<<block_size, num_of_blocks>>>(A, W, Z,
														A_x_dim, A_y_dim,
														x_dim, y_dim);
	cudaDeviceSynchronize();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot perform linear forward prop.");

	return Z;
}

int LinearLayer::getXDim() const {
	return x_dim;
}

int LinearLayer::getYDim() const {
	return y_dim;
}

const float* LinearLayer::getWeightsMatrix() const {
	return W;
}
