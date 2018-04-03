#include "gtest/gtest.h"
#include "layers/sigmoid_activation.hh"
#include "test_utils.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/matrix.hh"

#include <iostream>

namespace {

	class SigmoidActivationTest : public ::testing::Test {
	protected:
		SigmoidActivation sigmoid_layer;
		Matrix Z;

		SigmoidActivationTest() :
			sigmoid_layer("some_sigmoid_layer")
		{ }
	};

	TEST_F(SigmoidActivationTest, ShouldHaveName) {
		// given
		// when
		std::string layer_name = sigmoid_layer.getName();

		//then
		ASSERT_STREQ(layer_name.c_str(), "some_sigmoid_layer");
	}

	TEST_F(SigmoidActivationTest, ShouldPerformForwardProp) {
		// given
		Z.shape.x = 20;
		Z.shape.y = 10;
		Z.allocateMemory();

		testutils::initializeTensorRandomlyInRange(Z, -10, 10);
		Z.copyHostToDevice();

		// when
		Matrix A = sigmoid_layer.forward(Z);
		A.copyDeviceToHost();

		// then
		ASSERT_NE(A.data_device, nullptr);
		ASSERT_EQ(A.shape.x, A.shape.x);
		ASSERT_EQ(A.shape.y, A.shape.y);

		for (int Z_x = 0; Z_x < Z.shape.x; Z_x++) {
			for (int Z_y = 0; Z_y < Z.shape.y; Z_y++) {
				float Z_sigmoid = testutils::sigmoid(Z[Z_y * Z.shape.x + Z_x]);
				ASSERT_FLOAT_EQ(A[Z_y * Z.shape.x + Z_x], Z_sigmoid);
			}
		}
	}

	TEST_F(SigmoidActivationTest, ShouldPerformBackprop) {
		// given
		Z.shape.x = 10;
		Z.shape.y = 5;
		Z.allocateMemory();

		testutils::initializeTensorWithValue(Z, 3);
		Z.copyHostToDevice();

		Matrix dA(10, 5);
		dA.allocateMemory();

		testutils::initializeTensorWithValue(dA, 2);
		dA.copyHostToDevice();

		float expected_dZ = 2 * testutils::sigmoid(3) * (1 - testutils::sigmoid(3));

		// when
		Matrix A = sigmoid_layer.forward(Z);
		Matrix dZ = sigmoid_layer.backprop(dA);
		dZ.copyDeviceToHost();

		// then
		ASSERT_NE(dZ.data_device, nullptr);
		ASSERT_EQ(dZ.shape.x, Z.shape.x);
		ASSERT_EQ(dZ.shape.y, Z.shape.y);
		for (int dZ_x = 0; dZ_x < dZ.shape.x; dZ_x++) {
			for (int dZ_y = 0; dZ_y < dZ.shape.y; dZ_y++) {
				ASSERT_EQ(dZ[dZ_y * dZ.shape.x + dZ_x], expected_dZ);
			}
		}
	}

}
