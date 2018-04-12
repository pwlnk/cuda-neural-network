# CUDA Neural Network Implementation

It is a simple artificial neural network implementation using CUDA technology. This repository was created for the blog post available at [luniak.io/cuda-neural-network-implementation-part-1](http://luniak.io/cuda-neural-network-implementation-part-1) where much more information on this implementation can be found. It is just an educational implementation that has many performance issues and a lot can be improved.

## Requirements and Technical Info

This repository contains **Eclipse Nsight** project.

To run this project **CUDA Toolkit** is required.

During compilation **C++11** support has to be enabled.

## Creating a Network

In order to create new neural network you need to create a `NeuralNetwork` object and add some layers to it. Available layers are:

- `LinearLayer`
- `ReLUActivation`
- `SigmoidActivation`

Layers set can be easily expanded by creating new layers classes derived from `NNLayer` and implementing `forward()` and `backward()` methods. Additionally there is a `BCECost` class that implements _binary cross-entropy_ cost function. Below is an example of a simple network with two linear layers, one of them activated with _ReLU_ function and the last one with _sigmoid_ function.

```cpp
NeuralNetwork nn;
nn.addLayer(new LinearLayer("linear_1", Shape(2, 30)));
nn.addLayer(new ReLUActivation("relu_1"));
nn.addLayer(new LinearLayer("linear_2", Shape(30, 1)));
nn.addLayer(new SigmoidActivation("sigmoid_output"));
```

## Forward and Backward Pass

`NeuralNetwork` class implements `forward()` and `backprop()` methods. In order to make a prediction with created neural network you should call `forward()` function with input data as an argument (as `Matrix` object). If you want to perform backpropagation and update network weights you should call `backprop()` function with two vectors (`Matrix` objects), one with predicted values and second one with target values. Below is an example of a network training.

```cpp
Matrix Y;
for (int epoch = 0; epoch < 1001; epoch++) {
  float cost = 0.0;

  for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
    Y = nn.forward(dataset.getBatches().at(batch));
    nn.backprop(Y, dataset.getTargets().at(batch));
    cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
  }

  if (epoch % 100 == 0) {
    std::cout << "Epoch: " << epoch
              << ", Cost: " << cost / dataset.getNumOfBatches()
              << std::endl;
  }
}
```

## Coordinates Dataset

`CoordinatesDataset` class generates random points in 2D space and assign a class for each of them. Points that lies within 1st or 3rd quadrant have class `1` other points have class `0`. Points are stored in `baches` vector and class information in `targets` vector. During dataset creation one has to specify batch size and number of batches. 

```cpp
CoordinatesDataset dataset(100, 20);   // 20 batches, each containing 100 2D points
std::vector<Matrix> batches = dataset.getBatches();
std::vector<Matrix> targets = dataset.getTargets();
```
