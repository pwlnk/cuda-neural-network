#include "linear_layer.hh"

LinearLayer::LinearLayer(std::string name) {
	this->name = name;
}

LinearLayer::~LinearLayer() { }

float* LinearLayer::forward(float* A) {
	std::cout << this->getName() << std:: endl;
	return nullptr;
}
