#include <iostream>
#include <time.h>

#include "neural_network.hh"
#include "linear_layer.hh"
#include "relu_activation.hh"
#include "sigmoid_activation.hh"
#include "nn_exception.hh"

int main() {

	srand( time(NULL) );

	nn_utils::Tensor3D X(100, 1);
	X.allocateCudaMemory();
	X.allocateHostMemory();

	nn_utils::Tensor3D target(100, 1);
	target.allocateCudaMemory();
	target.allocateHostMemory();

	for (int i = 0; i < X.shape.x; i++) {
		X[i] = i;
		target[i] = i <= 50 ? 0 : 1;
	}

	X.copyHostToDevice();
	target.copyHostToDevice();

	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear_1", nn_utils::Shape(1, 25)));
	nn.addLayer(new ReLUActivation("relu_1"));
	nn.addLayer(new LinearLayer("linear_2", nn_utils::Shape(25, 1)));
	nn.addLayer(new SigmoidActivation("sigmoid_output"));

	nn_utils::Tensor3D Y;

	for (int i = 0; i < 1000; i++) {
		Y = nn.forward(X);
		nn.backprop(Y, target);
	}

	Y.allocateHostMemory();
	Y.copyDeviceToHost();

	std::cout << "Prediction: " << Y[1]
								  << ", Target: " << target[1]
								  << ", Cost: " << nn_utils::binaryCrossEntropyCost(Y, target) << std::endl;

	return 0;
}
