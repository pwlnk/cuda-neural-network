#include "gtest/gtest.h"
#include "test_utils.hh"
#include "linear_layer.hh"

namespace {

	class LinearLayerTest : public ::testing::Test {
	protected:
		LinearLayer linear_layer;

		LinearLayerTest() :
			linear_layer("some_linear_layer", 10, 20)
		{ }
	};

	TEST_F(LinearLayerTest, ShouldHaveName) {
		// given
		// when
		std::string layer_name = linear_layer.getName();

		//then
		ASSERT_STREQ(layer_name.c_str(), "some_linear_layer");
	}

	TEST_F(LinearLayerTest, ShouldHaveInitializedWeightsMatrixSize) {
		// given
		// when
		int x_dim = linear_layer.getXDim();
		int y_dim = linear_layer.getYDim();

		// then
		EXPECT_EQ(x_dim, 10);
		EXPECT_EQ(y_dim, 20);
	}

	TEST_F(LinearLayerTest, ShouldHaveWeightsInitializedRandomlyWithNumbersLowerThan0p01) {
		// given
		// when
		const float* W = linear_layer.getWeightsMatrix();
		int x_dim = linear_layer.getXDim();
		int y_dim = linear_layer.getYDim();

		// then
		float prev_weight_val = -1.0;
		for (int x = 0; x < x_dim; x++) {
			for (int y = 0; y < y_dim; y++) {
				ASSERT_GE(W[y * x_dim + x], 0);
				ASSERT_LE(W[y * x_dim + x], 0.01);
				ASSERT_NE(W[y * x_dim + x], prev_weight_val);
				prev_weight_val = W[y * x_dim + x];
			}
		}
	}

	TEST_F(LinearLayerTest, ShouldPerformForwardProp) {
		// given
		float* A;
		int A_x_dim = linear_layer.getYDim();
		int A_y_dim = 10;
		int Z_x_dim = A_x_dim;
		int Z_y_dim = A_x_dim;
		int W_x_dim = linear_layer.getXDim();

		cudaMallocManaged(&A, A_x_dim * A_y_dim * sizeof(float));
		testutils::initializeMatrixWithValue( linear_layer.W,
											  linear_layer.getXDim(),
											  linear_layer.getYDim(),
											  2);
		testutils::initializeMatrixWithValue(A, A_x_dim, A_y_dim, 3);

		// when
		float* Z = linear_layer.forward(A, A_x_dim, A_y_dim);

		// then
		ASSERT_NE(Z, nullptr);
		for (int Z_x = 0; Z_x < Z_x_dim; Z_x++) {
			for (int Z_y = 0; Z_y < Z_y_dim; Z_y++) {
				ASSERT_EQ(Z[Z_y * Z_x_dim + Z_x], 2 * 3 * W_x_dim);
			}
		}
	}

}
