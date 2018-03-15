#include <iostream>

#include "neural_network.hh"
#include "linear_layer.hh"
#include "relu_activation.hh"

int main() {

	NeuralNetwork neuralNet;

	neuralNet.addLayer(new LinearLayer("linear_layer_1"));
	neuralNet.addLayer(new ReLUActivation("relu_1"));
	neuralNet.addLayer(new LinearLayer("linear_layer_2"));
	neuralNet.addLayer(new ReLUActivation("relu_2"));
	neuralNet.addLayer(new LinearLayer("linear_layer_3"));
	neuralNet.addLayer(new ReLUActivation("relu_3"));
	neuralNet.addLayer(new LinearLayer("linear_layer_4"));
	neuralNet.addLayer(new ReLUActivation("relu_4"));

	neuralNet.forward(nullptr);

	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

	return 0;
}
