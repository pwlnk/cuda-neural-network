#include "relu_activation.hh"

ReLUActivation::ReLUActivation(std::string name) {
	this->name = name;
}

ReLUActivation::~ReLUActivation() { }

float* ReLUActivation::forward(float* A) {
	std::cout << this->getName() << std:: endl;
	return nullptr;
}
