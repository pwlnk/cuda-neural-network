#pragma once

#include <iostream>

class NNLayer {
protected:
	std::string name;

public:
	virtual ~NNLayer() = 0;

	virtual float* forward(float* A) = 0;
	std::string getName() { return this->name; };

};

inline NNLayer::~NNLayer() {}
