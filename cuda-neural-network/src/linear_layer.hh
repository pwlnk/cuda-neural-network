#pragma once

#include "nn_layer.hh"

// for unit testing purposes only
namespace {
	class LinearLayerTest_ShouldPerformForwardProp_Test;
}

class LinearLayer : public NNLayer {
private:
	const float weights_init_threshold = 0.01;

	int x_dim, y_dim;
	float* W;
	float* Z;

	void allocateWeightsMemory();
	void initializeWeightsRandomly();

	// for unit testing purposes only
	friend class ::LinearLayerTest_ShouldPerformForwardProp_Test;

public:
	LinearLayer(std::string name, int x_dim, int y_dim);
	~LinearLayer();

	float* forward(float* A, int A_x_dim, int A_y_dim);
	int getXDim() const;
	int getYDim() const;
	const float* getWeightsMatrix() const;
};
