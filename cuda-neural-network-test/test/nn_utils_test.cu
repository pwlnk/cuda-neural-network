#include "gtest/gtest.h"
#include "test_utils.hh"
#include "nn_utils.hh"

#include <iostream>

namespace {

	class NNUtilsTest : public ::testing::Test {
	protected:

		nn_utils::Tensor3D target;
		nn_utils::Tensor3D predictions;

		virtual void SetUp() {
			target.shape = nn_utils::Shape(100, 1);
			target.allocateCudaMemory();
			target.allocateHostMemory();

			predictions.shape = nn_utils::Shape(100, 1);
			predictions.allocateCudaMemory();
			predictions.allocateHostMemory();
		}

		virtual void TearDown() {
			target.freeCudaAndHostMemory();
			predictions.freeCudaAndHostMemory();
		}

	};

	TEST_F(NNUtilsTest, ShouldCalculateBinaryCrossEntropyLoss) {
		// given
		testutils::initializeTensorWithValue(predictions, 0.0001);
		testutils::initializeTensorWithValue(target, 1);

		predictions.copyHostToDevice();
		target.copyHostToDevice();

		// when
		float cost = nn_utils::binaryCrossEntropyCost(predictions, target);

		// then
		ASSERT_NEAR(cost, -log(0.0001), 0.0001);
	}


	TEST_F(NNUtilsTest, ShouldCalculateDerivativeOfBinaryCrossEntropyLoss) {
		// given
		testutils::initializeTensorWithValue(predictions, 0.0001);
		testutils::initializeTensorWithValue(target, 1);

		nn_utils::Tensor3D dY(predictions.shape);
		dY.allocateCudaMemory();

		predictions.copyHostToDevice();
		target.copyHostToDevice();

		float expected_derivative = (0.0001 - 1) / ((1 - 0.0001) * 0.0001);

		// when
		nn_utils::Tensor3D d_cross_entropy_loss = nn_utils::dBinaryCrossEntropyCost(predictions, target, dY);

		d_cross_entropy_loss.allocateHostMemory();
		d_cross_entropy_loss.copyDeviceToHost();

		// then
		for (int i = 0; i < predictions.shape.x; i++) {
			ASSERT_NEAR(d_cross_entropy_loss[i], expected_derivative, 0.00001);
		}

		dY.freeCudaAndHostMemory();
	}

}
