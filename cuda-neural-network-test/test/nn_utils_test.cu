#include "gtest/gtest.h"
#include "test_utils.hh"
#include "nn_utils.hh"

namespace {

	class NNUtilsTest : public ::testing::Test {
	protected:

		nn_utils::Tensor3D target;
		nn_utils::Tensor3D predictions;

		virtual void TearDown() {
			target.freeCudaMemory();
			predictions.freeCudaMemory();
		}

	};

	TEST_F(NNUtilsTest, ShouldCalculateBinaryCrossEntropyLoss) {
		// given
		predictions = nn_utils::Shape(100, 1);
		target = nn_utils::Shape(100, 1);
		predictions.allocateCudaMemory();
		target.allocateCudaMemory();

		testutils::initializeTensorWithValue(predictions, 0.0001);
		testutils::initializeTensorWithValue(target, 1);

		// when
		float loss = nn_utils::binaryCrossEntropyCost(predictions, target);

		// then
		ASSERT_NEAR(loss, -log(0.0001), 0.0001);
	}



	TEST_F(NNUtilsTest, ShouldCalculateDerivativeOfBinaryCrossEntropyLoss) {
		// given
		predictions = nn_utils::Shape(100, 1);
		target = nn_utils::Shape(100, 1);
		predictions.allocateCudaMemory();
		target.allocateCudaMemory();

		testutils::initializeTensorWithValue(predictions, 0.0001);
		testutils::initializeTensorWithValue(target, 1);

		float expected_derivative = - (0.0001 - 1) / ((1 - 0.0001) * 0.0001);

		// when
		float d_cross_entropy_loss = nn_utils::dBinaryCrossEntropyCost(predictions, target);

		// then
		for (int i = 0; i < predictions.shape.x; i++) {
			ASSERT_NEAR(d_cross_entropy_loss, expected_derivative, 0.00001);
		}
	}

}
