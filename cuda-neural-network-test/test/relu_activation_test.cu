#include "gtest/gtest.h"
#include "relu_activation.hh"
#include "test_utils.hh"

namespace {

	class ReLUActivationTest : public ::testing::Test {
	protected:
		ReLUActivation relu_layer;
		nn_utils::Tensor3D Z;

		ReLUActivationTest() :
			relu_layer("some_relu_layer")
		{ }

		virtual void TearDown() {
			cudaFree(Z.data);
		}
	};

	TEST_F(ReLUActivationTest, ShouldHaveName) {
		// given
		// when
		std::string layer_name = relu_layer.getName();

		//then
		ASSERT_STREQ(layer_name.c_str(), "some_relu_layer");
	}

	TEST_F(ReLUActivationTest, ShouldPerformForwardProp) {
		// given
		Z.shape.x = 20;
		Z.shape.y = 10;
		Z.allocateCudaMemory();
		testutils::initializeTensorRandomlyInRange(Z, -10, 10);

		// when
		nn_utils::Tensor3D A = relu_layer.forward(Z);

		// then
		ASSERT_NE(A.data, nullptr);
		for (int Z_x = 0; Z_x < Z.shape.x; Z_x++) {
			for (int Z_y = 0; Z_y < Z.shape.y; Z_y++) {
				if (Z.data[Z_y * Z.shape.x + Z_x] < 0) {
					ASSERT_EQ(A.data[Z_y * Z.shape.x + Z_x], 0);
				}
				else {
					ASSERT_EQ(A.data[Z_y * Z.shape.x + Z_x], Z.data[Z_y * Z.shape.x + Z_x]);
				}
			}
		}
	}

	TEST_F(ReLUActivationTest, ShouldPerformBackprop) {
		// given
		Z.shape.x = 10;
		Z.shape.y = 5;
		Z.allocateCudaMemory();

		for (int i = 0; i < Z.shape.x; i++) {
			for (int j = 0; j < Z.shape.y; j++) {
				if (i < 3) {
					Z.data[j * Z.shape.x + i] = 4;
				}
				else {
					Z.data[j * Z.shape.x + i] = -2;
				}
			}
		}

		nn_utils::Tensor3D dA(10, 5);
		dA.allocateCudaMemory();
		testutils::initializeTensorWithValue(dA, 2);

		// when
		nn_utils::Tensor3D A = relu_layer.forward(Z);
		nn_utils::Tensor3D dZ = relu_layer.backprop(dA);

		// then
		ASSERT_NE(dZ.data, nullptr);
		ASSERT_EQ(dZ.shape.x, Z.shape.x);
		ASSERT_EQ(dZ.shape.y, Z.shape.y);
		for (int dZ_x = 0; dZ_x < dZ.shape.x; dZ_x++) {
			for (int dZ_y = 0; dZ_y < dZ.shape.y; dZ_y++) {
				if (Z.data[dZ_y * Z.shape.x + dZ_x] > 0) {
					ASSERT_EQ(dZ.data[dZ_y * dZ.shape.x + dZ_x], 1 * 2);
				}
				else {
					ASSERT_EQ(dZ.data[dZ_y * dZ.shape.x + dZ_x], 0);
				}
			}
		}
	}

}
