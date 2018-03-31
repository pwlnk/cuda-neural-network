#include <iostream>
#include <time.h>
#include <vector>

#include "neural_network.hh"
#include "linear_layer.hh"
#include "relu_activation.hh"
#include "sigmoid_activation.hh"
#include "nn_exception.hh"

int main() {

	srand( time(NULL) );

	std::vector<nn_utils::Tensor3D> dataset;
	std::vector<nn_utils::Tensor3D> targets;

	for (int i = 0; i < 20; i++) {
		dataset.push_back(nn_utils::Tensor3D(100, 2));
		targets.push_back(nn_utils::Tensor3D(100, 1));

		dataset[i].allocateCudaMemory();
		dataset[i].allocateHostMemory();

		targets[i].allocateCudaMemory();
		targets[i].allocateHostMemory();

		for (int k = 0; k < dataset[i].shape.x; k++) {
			dataset[i][k] = (float(rand()) / RAND_MAX - 0.5);
			dataset[i][dataset[i].shape.x + k] = (float(rand()) / RAND_MAX - 0.5);
//			targets[i][k] = dataset[i][k * 2] > 0 && dataset[i][k * 2 + 1] > 0 ? 1 : 0;
			targets[i][k] = dataset[i][k] > 0 ? 1 : 0;
		}

		dataset[i].copyHostToDevice();
		targets[i].copyHostToDevice();
	}

	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear_1", nn_utils::Shape(2, 100)));
	nn.addLayer(new ReLUActivation("relu_1"));
	nn.addLayer(new LinearLayer("linear_2", nn_utils::Shape(100, 1)));
	nn.addLayer(new SigmoidActivation("sigmoid_output"));

	nn_utils::Tensor3D Y;

	for (int epoch = 0; epoch < 1001; epoch++) {
		float cost = 0.0;

		for (int batch = 0; batch < 20; batch++) {
			Y = nn.forward(dataset[batch]);
			nn.backprop(Y, targets[batch]);
			cost += nn_utils::binaryCrossEntropyCost(Y, targets[batch]);
		}

		if (epoch % 100 == 0) {
			std::cout << "epoch: " << epoch << ", cost: " << cost / 20 << std::endl;
		}
	}

	return 0;
}
