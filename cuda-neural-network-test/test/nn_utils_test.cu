#include "gtest/gtest.h"
#include "nn_utils.hh"
#include "test_utils.hh"

#include <math.h>

namespace {

	class NNUtilsTest : public ::testing::Test {
	protected:

		virtual void SetUp() {

		}

		virtual void TearDown() {

		}
	};

	TEST_F(NNUtilsTest, ShouldCalculateBinaryCrossEntropyLoss) {
		// given
		nn_utils::Tensor3D predictions(100, 1);
		nn_utils::Tensor3D target(100, 1);
		predictions.allocateCudaMemory();
		target.allocateCudaMemory();

		testutils::initializeTensorWithValue(predictions, 0.0001);
		testutils::initializeTensorWithValue(target, 1);

		// when
		float loss = nn_utils::binaryCrossEntropyLoss(predictions, target);

		// then
		ASSERT_NEAR(loss, -log(0.0001), 0.0001);
	}

}
