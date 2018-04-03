#include "gtest/gtest.h"
#include "test_utils.hh"
#include "nn_utils/bce_cost.hh"
#include "nn_utils/matrix.hh"

#include <iostream>

namespace {

	class BCECostTest : public ::testing::Test {
	protected:
		BCECost bce_cost;

		Matrix target;
		Matrix predictions;

		virtual void SetUp() {
			target.shape = Shape(100, 1);
			target.allocateMemory();

			predictions.shape = Shape(100, 1);
			predictions.allocateMemory();
		}
	};

	TEST_F(BCECostTest, ShouldCalculateBinaryCrossEntropyLoss) {
		// given
		testutils::initializeTensorWithValue(predictions, 0.0001);
		testutils::initializeTensorWithValue(target, 1);

		predictions.copyHostToDevice();
		target.copyHostToDevice();

		// when
		float cost = bce_cost.cost(predictions, target);

		// then
		ASSERT_NEAR(cost, -log(0.0001), 0.0001);
	}


	TEST_F(BCECostTest, ShouldCalculateDerivativeOfBinaryCrossEntropyLoss) {
		// given
		testutils::initializeTensorWithValue(predictions, 0.0001);
		testutils::initializeTensorWithValue(target, 1);

		Matrix dY(predictions.shape);
		dY.allocateMemory();

		predictions.copyHostToDevice();
		target.copyHostToDevice();

		float expected_derivative = (0.0001 - 1) / ((1 - 0.0001) * 0.0001);

		// when
		Matrix d_cross_entropy_loss = bce_cost.dCost(predictions, target, dY);
		d_cross_entropy_loss.copyDeviceToHost();

		// then
		for (int i = 0; i < predictions.shape.x; i++) {
			ASSERT_NEAR(d_cross_entropy_loss[i], expected_derivative, 0.00001);
		}
	}

}
