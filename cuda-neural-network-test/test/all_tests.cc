#include "gtest/gtest.h"
#include "linear_layer_test.cc"

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
