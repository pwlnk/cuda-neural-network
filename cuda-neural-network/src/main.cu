#include <iostream>
#include <time.h>

#include "neural_network.hh"
#include "linear_layer.hh"
#include "relu_activation.hh"

int main() {

	srand( time(NULL) );

	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

	return 0;
}
