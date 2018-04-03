#include "gtest/gtest.h"
#include "test_utils.hh"
#include "nn_utils/nn_utils.hh"
#include "nn_utils/matrix.hh"

#include <iostream>

namespace {

	class NNUtilsTest : public ::testing::Test {
	protected:

		Matrix target;
		Matrix predictions;

		virtual void SetUp() {
			target.shape = Shape(100, 1);
			target.allocateCudaMemory();
			target.allocateHostMemory();

			predictions.shape = Shape(100, 1);
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

		Matrix dY(predictions.shape);
		dY.allocateCudaMemory();

		predictions.copyHostToDevice();
		target.copyHostToDevice();

		float expected_derivative = (0.0001 - 1) / ((1 - 0.0001) * 0.0001);

		// when
		Matrix d_cross_entropy_loss = nn_utils::dBinaryCrossEntropyCost(predictions, target, dY);

		d_cross_entropy_loss.allocateHostMemory();
		d_cross_entropy_loss.copyDeviceToHost();

		// then
		for (int i = 0; i < predictions.shape.x; i++) {
			ASSERT_NEAR(d_cross_entropy_loss[i], expected_derivative, 0.00001);
		}

		dY.freeCudaAndHostMemory();
	}

}
