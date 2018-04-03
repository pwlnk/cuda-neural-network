#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "test_utils.hh"
#include "layers/linear_layer.hh"
#include "nn_utils/matrix.hh"

namespace {

	class LinearLayerTest : public ::testing::Test {
	protected:
		LinearLayer linear_layer;
		Shape W_shape = Shape(2, 4);

		Matrix A;
		Matrix dZ;

		LinearLayerTest() :
			A(Shape(3, 2)), dZ(Shape(3, 4)),
			linear_layer("some_linear_layer", Shape(2, 4))
		{
			A.allocateMemory();
			dZ.allocateMemory();
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
		ASSERT_EQ(x_dim, W_shape.x);
		ASSERT_EQ(y_dim, W_shape.y);
	}

	TEST_F(LinearLayerTest, ShouldHaveInitializedBiasVectorWithZeros) {
		// given
		// when
		Matrix b = linear_layer.getBiasVector();
		b.copyDeviceToHost();

		// then
		ASSERT_EQ(b.shape.x, W_shape.y);
		ASSERT_EQ(b.shape.y, 1);
		for (int x = 0; x < b.shape.x; x++) {
			ASSERT_EQ(b[x], 0);
		}
	}

	TEST_F(LinearLayerTest, ShouldHaveWeightsInitializedRandomly) {
		// given
		// when
		Matrix W = linear_layer.getWeightsMatrix();
		W.copyDeviceToHost();

		// then
		float prev_weight_value = -1.0;
		for (int x = 0; x < W.shape.x; x++) {
			for (int y = 0; y < W.shape.y; y++) {
				ASSERT_NE(W[y * W.shape.x + x], prev_weight_value);
				prev_weight_value = W[y * W.shape.x + x];
			}
		}
	}

	TEST_F(LinearLayerTest, ShouldReturnOutputAfterForwardProp) {
		// given
		std::vector<float> b_cols_values = {1, 2, 3, 4};
		std::vector<float> W_rows_values = {2, 4, 6, 8};
		std::vector<float> A_cols_values = {3, 5, 7};

		testutils::initializeEachTensorRowWithValue(linear_layer.W, W_rows_values);
		testutils::initializeEachTensorColWithValue(linear_layer.b, b_cols_values);
		testutils::initializeEachTensorColWithValue(A, A_cols_values);

		linear_layer.W.copyHostToDevice();
		linear_layer.b.copyHostToDevice();
		A.copyHostToDevice();

		// when
		Matrix Z = linear_layer.forward(A);
		Z.copyDeviceToHost();

		// then
		ASSERT_NE(Z.data_device, nullptr);
		ASSERT_EQ(Z.shape.x, A.shape.x);
		ASSERT_EQ(Z.shape.y, W_shape.y);

		for (int x = 0; x < Z.shape.x; x++) {
			for (int y = 0; y < Z.shape.y; y++) {
				float cell_value = W_rows_values[y] * A_cols_values[x] * W_shape.x + b_cols_values[y];
				ASSERT_EQ(Z[y * Z.shape.x + x], cell_value);
			}
		}
	}

	// dA = dot(W^T, dZ)
	TEST_F(LinearLayerTest, ShouldReturnDerivativeAfterBackprop) {
		// given
		std::vector<float> W_cols_values = {6, 8};
		std::vector<float> dZ_cols_values = {3, 5, 7};

		testutils::initializeEachTensorColWithValue(linear_layer.W, W_cols_values);
		testutils::initializeEachTensorColWithValue(dZ, dZ_cols_values);

		linear_layer.W.copyHostToDevice();
		dZ.copyHostToDevice();

		// when
		Matrix Z = linear_layer.forward(A);
		Matrix dA = linear_layer.backprop(dZ);
		dA.copyDeviceToHost();

		// then
		ASSERT_NE(dA.data_device, nullptr);
		ASSERT_EQ(dA.shape.x, A.shape.x);
		ASSERT_EQ(dA.shape.y, A.shape.y);

		for (int x = 0; x < dA.shape.x; x++) {
			for (int y = 0; y < dA.shape.y; y++) {
				float cell_value = W_cols_values[y] * dZ_cols_values[x] * W_shape.y;
				ASSERT_EQ(dA[y * dA.shape.x + x], cell_value);
			}
		}
	}

	TEST_F(LinearLayerTest, ShouldUptadeItsBiasDuringBackprop) {
		// given
		std::vector<float> b_cols_values = {1, 2, 3, 4};
		std::vector<float> dZ_rows_values = {3, 5, 7, 9};
		float learning_rate = 0.1;

		testutils::initializeEachTensorColWithValue(linear_layer.b, b_cols_values);
		testutils::initializeEachTensorRowWithValue(dZ, dZ_rows_values);

		linear_layer.b.copyHostToDevice();
		dZ.copyHostToDevice();

		// when
		Matrix Z = linear_layer.forward(A);
		Matrix dA = linear_layer.backprop(dZ, learning_rate);

		linear_layer.b.copyDeviceToHost();

		// then
		ASSERT_NE(linear_layer.b.data_device, nullptr);

		for (int x = 0; x < linear_layer.b.shape.x; x++) {
			float bias_after_gdc = b_cols_values[x] - learning_rate * dZ_rows_values[x];
			ASSERT_NEAR(linear_layer.b[x], bias_after_gdc, 0.0001);
		}
	}

	TEST_F(LinearLayerTest, ShouldUptadeItsWeightsDuringBackprop) {
		// given
		std::vector<float> W_cols_values = {2, 4};
		std::vector<float> dZ_rows_values = {3, 5, 7, 9};
		std::vector<float> A_rows_values = {2, 4};
		float learning_rate = 0.1;

		testutils::initializeEachTensorColWithValue(linear_layer.W, W_cols_values);
		testutils::initializeEachTensorRowWithValue(dZ, dZ_rows_values);
		testutils::initializeEachTensorRowWithValue(A, A_rows_values);

		linear_layer.W.copyHostToDevice();
		dZ.copyHostToDevice();
		A.copyHostToDevice();

		// when
		Matrix Z = linear_layer.forward(A);
		Matrix dA = linear_layer.backprop(dZ, learning_rate);

		linear_layer.W.copyDeviceToHost();

		// then
		ASSERT_NE(linear_layer.W.data_device, nullptr);

		for (int x = 0; x < W_shape.x; x++) {
			for (int y = 0; y < W_shape.y; y++) {
				float weight_after_gdc = W_cols_values[x] - learning_rate * dZ_rows_values[y] * A_rows_values[x];
				ASSERT_NEAR(linear_layer.W[y * W_shape.x + x], weight_after_gdc, 0.0001);
			}
		}
	}

}
