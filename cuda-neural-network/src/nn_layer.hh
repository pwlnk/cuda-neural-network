#pragma once

#include <iostream>

class NNLayer {
protected:
	std::string name;

public:
	virtual ~NNLayer() = 0;

	virtual float* forward(float* A, int A_x_dim, int A_y_dim) = 0;
	std::string getName() { return this->name; };

};

inline NNLayer::~NNLayer() {}
