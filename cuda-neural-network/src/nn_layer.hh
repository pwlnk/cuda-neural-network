#pragma once

#include <iostream>
#include "nn_utils.hh"

class NNLayer {
protected:
	std::string name;

public:
	virtual ~NNLayer() = 0;

	virtual nn_utils::Tensor3D forward(nn_utils::Tensor3D A) = 0;
	std::string getName() { return this->name; };

};

inline NNLayer::~NNLayer() {}
