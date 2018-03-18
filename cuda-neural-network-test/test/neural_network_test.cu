#include <vector>

#include "gtest/gtest.h"
#include "neural_network.hh"
#include "linear_layer.hh"
#include "relu_activation.hh"
#include "test_utils.hh"

namespace {

	class NeuralNetworkTest : public ::testing::Test {
	protected:
		NeuralNetwork neural_network;
		nn_utils::Tensor3D X;

		virtual void TearDown() {
			X.freeCudaMemory();
		}
	};

	TEST_F(NeuralNetworkTest, ShouldStoreAddedLayersInOrder) {
		// given
		LinearLayer* lin_layer_1 = new LinearLayer("linear_layer_1", nn_utils::Shape(10, 10));
		LinearLayer* lin_layer_2 = new LinearLayer("linear_layer_2", nn_utils::Shape(20, 20));
		ReLUActivation* relu_activation_1 = new ReLUActivation("relu_activation_1");
		ReLUActivation* relu_activation_2 = new ReLUActivation("relu_activation_2");

		// when
		neural_network.addLayer(lin_layer_1);
		neural_network.addLayer(relu_activation_1);
		neural_network.addLayer(lin_layer_2);
		neural_network.addLayer(relu_activation_2);

		// then
		std::vector<NNLayer*> layers = neural_network.getLayers();
		ASSERT_EQ(layers.size(), 4);
		ASSERT_STREQ(layers.at(0)->getName().c_str(), "linear_layer_1");
		ASSERT_STREQ(layers.at(1)->getName().c_str(), "relu_activation_1");
		ASSERT_STREQ(layers.at(2)->getName().c_str(), "linear_layer_2");
		ASSERT_STREQ(layers.at(3)->getName().c_str(), "relu_activation_2");
	}

	TEST_F(NeuralNetworkTest, ShouldPerformForwardProp) {
		// given
		nn_utils::Tensor3D X(10, 20);
		X.allocateCudaMemory();

		nn_utils::Shape output_shape(X.shape.x, 5);

		LinearLayer* linear_layer_1 = new LinearLayer("linear_layer_1", nn_utils::Shape(X.shape.y, 4));
		ReLUActivation* relu_layer = new ReLUActivation("relu_layer");
		LinearLayer* linear_layer_2 = new LinearLayer("linear_layer_2", nn_utils::Shape(4, output_shape.y));

		testutils::initializeTensorWithValue(X, 4);
		testutils::initializeTensorWithValue(linear_layer_1->W, 2);
		testutils::initializeTensorWithValue(linear_layer_2->W, 3);

		// when
		neural_network.addLayer(linear_layer_1);
		neural_network.addLayer(relu_layer);
		neural_network.addLayer(linear_layer_2);
		nn_utils::Tensor3D Y = neural_network.forward(X);

		// then
		ASSERT_NE(Y.data, nullptr);
		ASSERT_EQ(Y.shape.x, output_shape.x);
		ASSERT_EQ(Y.shape.y, output_shape.y);

		for (int out_x = 0; out_x < output_shape.x; out_x++) {
			for (int out_y = 0; out_y < output_shape.y; out_y++) {
				ASSERT_EQ(Y.data[out_y * output_shape.x + out_x], 1920);
			}
		}
	}

}
