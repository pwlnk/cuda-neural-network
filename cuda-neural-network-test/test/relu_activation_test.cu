#include "gtest/gtest.h"
#include "relu_activation.hh"
#include "test_utils.hh"

namespace {

	class ReLUActivationTest : public ::testing::Test {
	protected:
		ReLUActivation relu_layer;
		nn_utils::Tensor3D A;

		ReLUActivationTest() :
			relu_layer("some_relu_layer")
		{ }

		virtual void TearDown() {
			cudaFree(A.data);
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
		A.shape.x = 20;
		A.shape.y = 10;
		A.allocateCudaMemory();
		testutils::initializeTensorRandomlyInRange(A, -10, 10);

		// when
		nn_utils::Tensor3D Z = relu_layer.forward(A);

		// then
		ASSERT_NE(Z.data, nullptr);
		for (int A_x = 0; A_x < A.shape.x; A_x++) {
			for (int A_y = 0; A_y < A.shape.y; A_y++) {
				if (A.data[A_y * A.shape.x + A_x] < 0) {
					ASSERT_EQ(Z.data[A_y * A.shape.x + A_x], 0);
				}
				else {
					ASSERT_EQ(Z.data[A_y * A.shape.x + A_x], A.data[A_y * A.shape.x + A_x]);
				}
			}
		}
	}

}
