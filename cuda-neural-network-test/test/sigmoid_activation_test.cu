#include "gtest/gtest.h"
#include "sigmoid_activation.hh"
#include "test_utils.hh"

namespace {

	class SigmoidActivationTest : public ::testing::Test {
	protected:
		SigmoidActivation sigmoid_layer;
		nn_utils::Tensor3D A;

		SigmoidActivationTest() :
			sigmoid_layer("some_sigmoid_layer")
		{ }

		virtual void TearDown() {
			cudaFree(A.data);
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
		A.shape.x = 20;
		A.shape.y = 10;
		A.allocateCudaMemory();
		testutils::initializeTensorRandomlyInRange(A, -10, 10);

		// when
		nn_utils::Tensor3D Z = sigmoid_layer.forward(A);

		// then
		ASSERT_NE(Z.data, nullptr);
		ASSERT_EQ(Z.shape.x, A.shape.x);
		ASSERT_EQ(Z.shape.y, A.shape.y);

		for (int A_x = 0; A_x < A.shape.x; A_x++) {
			for (int A_y = 0; A_y < A.shape.y; A_y++) {
				float A_sigmoid = testutils::sigmoid(A.data[A_y * A.shape.x + A_x]);
				ASSERT_FLOAT_EQ(Z.data[A_y * A.shape.x + A_x], A_sigmoid);
			}
		}
	}

}
