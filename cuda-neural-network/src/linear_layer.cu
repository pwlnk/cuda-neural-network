#include <stdlib.h>

#include "linear_layer.hh"
#include "nn_exception.hh"
#include "nn_utils.hh"

__global__ void weightedSum(float* A, float* W, float* Z,
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
		float product_val = 0;

		for (int A_x = 0; A_x < A_x_dim; A_x++) {
			Z_x = A_x;
			product_val = W[index] * A[A_y * A_x_dim + A_x];
			atomicAdd(&Z[Z_y * Z_x_dim + Z_x], product_val);
		}
	}
}

__global__ void addBias(float* Z, float* b, int Z_x_dim, int Z_y_dim) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		int row = static_cast<int>(index / Z_x_dim);
		int col = index % Z_x_dim;
		Z[index] += b[row];
	}
}

__global__ void linearLayerBackprop(float* W, float* dZ, float *dA,
									int W_x_dim, int W_y_dim,
									int dZ_x_dim, int dZ_y_dim) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < W_x_dim * W_y_dim) {
		// W is treated as it would be transposed
		int W_y = index % W_x_dim;
		int W_x = index / W_x_dim;

		int dA_x_dim = dZ_x_dim;
		int dA_y = W_y;
		int dA_x = 0;

		int dZ_y = W_x;
		float product_val = 0;

		for (int dZ_x = 0; dZ_x < dZ_x_dim; dZ_x++) {
			dA_x = dZ_x;
			product_val = W[index] * dZ[dZ_y * dZ_x_dim + dZ_x];
			atomicAdd(&dA[dA_y * dA_x_dim + dA_x], product_val);
		}
	}
}

__global__ void biasGDC(float* dZ, float* b,
						int dZ_x_dim, int dZ_y_dim,
						int b_x_dim,
						float learning_rate) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < dZ_x_dim * dZ_y_dim) {
		int dZ_x = index % dZ_x_dim;
		int dZ_y = index / dZ_x_dim;
		atomicAdd(&b[dZ_y], - learning_rate * (dZ[dZ_y * dZ_x_dim + dZ_x] / dZ_x_dim));
	}
}

__global__ void weightsGDC(float* A, float* dZ, float* W,
						   int A_x_dim, int A_y_dim,
						   int dZ_x_dim, int dZ_y_dim,
						   float learning_rate) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < dZ_x_dim * dZ_y_dim) {
		int dZ_x = index % dZ_x_dim;
		int dZ_y = index / dZ_x_dim;

		int A_x = dZ_x; // A is treated as it would be transposed
		int W_y = dZ_y;
		int W_x_dim = A_y_dim;

		for (int A_y = 0; A_y < A_y_dim; A_y++) {
			int W_x = A_y;
			float product_val = dZ[dZ_y * dZ_x_dim + dZ_x] * A[A_y * A_x_dim + A_x];
			atomicAdd(&W[W_y * W_x_dim + W_x], - learning_rate * (product_val / A_x_dim));
		}
	}
}

LinearLayer::LinearLayer(std::string name, nn_utils::Shape W_shape) :
	W(W_shape), b(W_shape.y, 1)
{
	this->name = name;
	b.allocateCudaMemory();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot initialize layer bias.");
	W.allocateCudaMemory();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot initialize layer weights.");

	initializeBiasWithZeros();
	initializeWeightsRandomly();
}

void LinearLayer::initializeWeightsRandomly() {
	for (int x = 0; x < W.shape.x; x++) {
		for (int y = 0; y < W.shape.y; y++) {
			W.data[y * W.shape.x + x] = (static_cast<float>(rand()) / RAND_MAX) * weights_init_threshold;
		}
	}
}

void LinearLayer::initializeBiasWithZeros() {
	for (int x = 0; x < b.shape.x; x++) {
		b.data[x] = 0;
	}
}

LinearLayer::~LinearLayer() {
	W.freeCudaMemory();
	Z.freeCudaMemory();
}

nn_utils::Tensor3D LinearLayer::forward(nn_utils::Tensor3D A) {

	this->A = A;

	// TODO: should be initialized only once, not with every forward() call
	cudaMallocManaged(&Z.data, W.shape.y * A.shape.x * sizeof(float));

	if (W.shape.x != A.shape.y) {
		throw NNException("Weight matrix and input matrix don't match.");
	}

	Z.shape = nn_utils::Shape(A.shape.x, W.shape.y);
	cudaMemset(Z.data, 0, Z.shape.x * Z.shape.y * sizeof(float));

	dim3 block_size(256);
	dim3 num_of_blocks((W.shape.y * W.shape.x + block_size.x - 1) / block_size.x);
	weightedSum<<<block_size, num_of_blocks>>>(A.data, W.data, Z.data,
											   A.shape.x, A.shape.y,
											   W.shape.x, W.shape.y);
	cudaDeviceSynchronize();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot perform linear forward prop.");

	block_size.x = 256;
	num_of_blocks.x = ((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	addBias<<<block_size, num_of_blocks>>>(Z.data, b.data, Z.shape.x, Z.shape.y);
	cudaDeviceSynchronize();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot perform linear forward prop.");

	return Z;
}

nn_utils::Tensor3D LinearLayer::backprop(nn_utils::Tensor3D dZ, float learning_rate) {
	// TODO: should be allocated once, not every time backprop is called
	dA.shape = A.shape;
	dA.allocateCudaMemory();
	cudaMemset(dA.data, 0, dA.shape.x * dA.shape.y * sizeof(float));

	// compute dA
	dim3 block_size(256);
	dim3 num_of_blocks((W.shape.y * W.shape.x + block_size.x - 1) / block_size.x);
	linearLayerBackprop<<<block_size, num_of_blocks>>>(W.data, dZ.data, dA.data,
														W.shape.x, W.shape.y,
														dZ.shape.x, dZ.shape.y);
	cudaDeviceSynchronize();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot perform linear forward prop.");

	// compute db and do GDC
	block_size.x = 256;
	num_of_blocks.x = (dZ.shape.y * dZ.shape.x + block_size.x - 1) / block_size.x;
	biasGDC<<<block_size, num_of_blocks>>>(dZ.data, b.data,
										   dZ.shape.x, dZ.shape.y,
										   b.shape.x, learning_rate);
	cudaDeviceSynchronize();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot perform bias GDC.");

	// compute dW and do GDC
	block_size.x = 256;
	num_of_blocks.x = (dZ.shape.y * dZ.shape.x + block_size.x - 1) / block_size.x;
	weightsGDC<<<block_size, num_of_blocks>>>(A.data, dZ.data, W.data,
											  A.shape.x, A.shape.y,
											  dZ.shape.x, dZ.shape.y,
											  learning_rate);
	cudaDeviceSynchronize();
	nn_utils::throwIfDeviceErrorsOccurred("Cannot perform weights GDC.");

	return dA;
}

int LinearLayer::getXDim() const {
	return W.shape.x;
}

int LinearLayer::getYDim() const {
	return W.shape.y;
}

const nn_utils::Tensor3D LinearLayer::getWeightsMatrix() const {
	return W;
}

const nn_utils::Tensor3D LinearLayer::getBiasVector() const {
	return b;
}
