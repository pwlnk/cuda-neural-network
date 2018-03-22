#include "gtest/gtest.h"
#include "test_utils.hh"
#include "linear_layer.hh"

namespace {

	class LinearLayerTest : public ::testing::Test {
	protected:
		LinearLayer linear_layer;
		nn_utils::Tensor3D A;

		LinearLayerTest() :
			linear_layer("some_linear_layer", nn_utils::Shape(10, 20))
		{ }

		virtual void TearDown() {
			cudaFree(A.data);
		}
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

	TEST_F(LinearLayerTest, ShouldHaveInitializedBiasVectorWithZeros) {
		// given
		// when
		const nn_utils::Tensor3D b = linear_layer.getBiasVector();

		// then
		ASSERT_EQ(b.shape.x, linear_layer.getYDim());
		ASSERT_EQ(b.shape.y, 1);
		for (int x = 0; x < b.shape.x; x++) {
			ASSERT_EQ(b.data[x], 0);
		}
	}

	TEST_F(LinearLayerTest, ShouldHaveWeightsInitializedRandomlyWithNumbersLowerThan0p01) {
		// given
		// when
		const nn_utils::Tensor3D W = linear_layer.getWeightsMatrix();

		// then
		float prev_weight_val = -1.0;
		for (int x = 0; x < W.shape.x; x++) {
			for (int y = 0; y < W.shape.y; y++) {
				ASSERT_GE(W.data[y * W.shape.x + x], 0);
				ASSERT_LE(W.data[y * W.shape.x + x], 0.01);
				ASSERT_NE(W.data[y * W.shape.x + x], prev_weight_val);
				prev_weight_val = W.data[y * W.shape.x + x];
			}
		}
	}

	TEST_F(LinearLayerTest, ShouldPerformForwardProp) {
		// given
		float bias_val = 5;

		A.shape.x = linear_layer.getYDim();
		A.shape.y = 10;
		A.allocateCudaMemory();

		testutils::initializeTensorWithValue(linear_layer.W, 2);
		testutils::initializeTensorWithValue(linear_layer.b, bias_val);
		testutils::initializeTensorWithValue(A, 3);

		// when
		nn_utils::Tensor3D Z = linear_layer.forward(A);

		// then
		ASSERT_NE(Z.data, nullptr);
		for (int Z_x = 0; Z_x < Z.shape.x; Z_x++) {
			for (int Z_y = 0; Z_y < Z.shape.y; Z_y++) {
				ASSERT_EQ(Z.data[Z_y * Z.shape.x + Z_x], 2 * 3 * linear_layer.getXDim() + bias_val);
			}
		}
	}

}
