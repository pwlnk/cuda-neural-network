#pragma once

#include "nn_layer.hh"

// for unit testing purposes only
namespace {
	class LinearLayerTest_ShouldPerformForwardProp_Test;
	class NeuralNetworkTest_ShouldPerformForwardProp_Test;
}

class LinearLayer : public NNLayer {
private:
	const float weights_init_threshold = 0.01;

	nn_utils::Tensor3D W;
	nn_utils::Tensor3D Z;

	void allocateWeightsMemory();
	void initializeWeightsRandomly();

	// for unit testing purposes only
	friend class LinearLayerTest_ShouldPerformForwardProp_Test;
	friend class NeuralNetworkTest_ShouldPerformForwardProp_Test;

public:
	LinearLayer(std::string name, nn_utils::Shape W_shape);
	~LinearLayer();

	nn_utils::Tensor3D forward(nn_utils::Tensor3D A);
	int getXDim() const;
	int getYDim() const;
	const nn_utils::Tensor3D getWeightsMatrix() const;
};
