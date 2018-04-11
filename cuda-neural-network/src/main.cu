#include <iostream>
#include <time.h>

#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/relu_activation.hh"
#include "layers/sigmoid_activation.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/bce_cost.hh"

#include "coordinates_dataset.hh"

int main() {

	srand( time(NULL) );

	CoordinatesDataset dataset(100, 20);
	BCECost bce_cost;

	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear_1", Shape(2, 30)));
	nn.addLayer(new ReLUActivation("relu_1"));
	nn.addLayer(new LinearLayer("linear_2", Shape(30, 1)));
	nn.addLayer(new SigmoidActivation("sigmoid_output"));

	Matrix Y;

	for (int epoch = 0; epoch < 1501; epoch++) {
		float cost = 0.0;

		for (int batch = 0; batch < dataset.getNumOfBatches(); batch++) {
			Y = nn.forward(dataset.getBatches().at(batch));
			nn.backprop(Y, dataset.getTargets().at(batch));
			cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
		}

		if (epoch % 100 == 0) {
			std::cout 	<< "epoch: " << epoch
						<< ", cost: " << cost / dataset.getNumOfBatches()
						<< std::endl;
		}
	}

	return 0;
}
