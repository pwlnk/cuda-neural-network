#include <time.h>
#include "gtest/gtest.h"

#include "bce_cost_test.cu"
#include "linear_layer_test.cu"
#include "sigmoid_activation_test.cu"
#include "relu_activation_test.cu"
#include "neural_network_test.cu"

int main(int argc, char **argv)
{
	srand( time(NULL) );
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
