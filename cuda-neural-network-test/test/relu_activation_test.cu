#include "gtest/gtest.h"
#include "relu_activation.hh"
#include "test_utils.hh"

namespace {

	class ReLUActivationTest : public ::testing::Test {
	protected:
		ReLUActivation relu_layer;

		ReLUActivationTest() :
			relu_layer("some_relu_layer")
		{ }
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
		float* A;
		int A_x_dim = 20;
		int A_y_dim = 10;

		cudaMallocManaged(&A, A_x_dim * A_y_dim * sizeof(float));
		testutils::initializeMatrixRandomlyInRange(A, A_x_dim, A_y_dim, -10, 10);

		// when
		float* Z = relu_layer.forward(A, A_x_dim, A_y_dim);

		// then
		ASSERT_NE(Z, nullptr);
		for (int A_x = 0; A_x < A_x_dim; A_x++) {
			for (int A_y = 0; A_y < A_y_dim; A_y++) {
				if (A[A_y * A_x_dim + A_x] < 0) {
					ASSERT_EQ(Z[A_y * A_x_dim + A_x], 0);
				}
				else {
					ASSERT_EQ(Z[A_y * A_x_dim + A_x], A[A_y * A_x_dim + A_x]);
				}
			}
		}
	}

}
