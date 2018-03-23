#include <iostream>
#include <time.h>

#include "neural_network.hh"
#include "linear_layer.hh"
#include "relu_activation.hh"
#include "sigmoid_activation.hh"
#include "nn_exception.hh"

int main() {

	srand( time(NULL) );

	nn_utils::Tensor3D X(1, 100);
	X.allocateCudaMemory();
	for (int i = 0; i < X.shape.y; i++) {
		X.data[i] = -80;
	}

	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear_1", nn_utils::Shape(100, 40)));
	nn.addLayer(new ReLUActivation("relu_1"));
	nn.addLayer(new LinearLayer("linear_2", nn_utils::Shape(40, 20)));
	nn.addLayer(new ReLUActivation("relu_2"));
	nn.addLayer(new LinearLayer("linear_3", nn_utils::Shape(20, 1)));
	nn.addLayer(new SigmoidActivation("sigmoid_output"));

	nn_utils::Tensor3D Y;
	nn_utils::Tensor3D target;
	target.allocateCudaMemory();
	target[0] = 0;

	for (int i = 0; i < 100; i++) {
		Y = nn.forward(X);
		std::cout << "Prediction: " << Y.data[0]
				  << ", Target: " << target.data[0]
				  << ", Cost: " << nn_utils::binaryCrossEntropyCost(Y, target) << std::endl;
		nn.backprop(Y, target);
	}

	return 0;
}
