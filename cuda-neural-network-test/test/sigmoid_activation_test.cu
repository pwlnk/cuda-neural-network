#include "gtest/gtest.h"
#include "sigmoid_activation.hh"
#include "test_utils.hh"

namespace {

	class SigmoidActivationTest : public ::testing::Test {
	protected:
		SigmoidActivation sigmoid_layer;
		nn_utils::Tensor3D Z;

		SigmoidActivationTest() :
			sigmoid_layer("some_sigmoid_layer")
		{ }

		virtual void TearDown() {
			cudaFree(Z.data);
		}
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
		Z.allocateCudaMemory();
		testutils::initializeTensorRandomlyInRange(Z, -10, 10);

		// when
		nn_utils::Tensor3D A = sigmoid_layer.forward(Z);

		// then
		ASSERT_NE(A.data, nullptr);
		ASSERT_EQ(A.shape.x, A.shape.x);
		ASSERT_EQ(A.shape.y, A.shape.y);

		for (int Z_x = 0; Z_x < Z.shape.x; Z_x++) {
			for (int Z_y = 0; Z_y < Z.shape.y; Z_y++) {
				float Z_sigmoid = testutils::sigmoid(Z.data[Z_y * Z.shape.x + Z_x]);
				ASSERT_FLOAT_EQ(A.data[Z_y * Z.shape.x + Z_x], Z_sigmoid);
			}
		}
	}

	TEST_F(SigmoidActivationTest, ShouldPerformBackprop) {
		// given
		Z.shape.x = 10;
		Z.shape.y = 5;
		Z.allocateCudaMemory();
		testutils::initializeTensorWithValue(Z, 3);

		nn_utils::Tensor3D dA(10, 5);
		dA.allocateCudaMemory();
		testutils::initializeTensorWithValue(dA, 2);

		float expected_dZ = 2 * testutils::sigmoid(3) * (1 - testutils::sigmoid(3));

		// when
		nn_utils::Tensor3D A = sigmoid_layer.forward(Z);
		nn_utils::Tensor3D dZ = sigmoid_layer.backprop(dA);

		// then
		ASSERT_NE(dZ.data, nullptr);
		ASSERT_EQ(dZ.shape.x, Z.shape.x);
		ASSERT_EQ(dZ.shape.y, Z.shape.y);
		for (int dZ_x = 0; dZ_x < dZ.shape.x; dZ_x++) {
			for (int dZ_y = 0; dZ_y < dZ.shape.y; dZ_y++) {
				ASSERT_EQ(dZ.data[dZ_y * dZ.shape.x + dZ_x], expected_dZ);
			}
		}
	}

}
