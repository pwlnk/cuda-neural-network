#pragma once

#include "shape.hh"

struct Matrix {
	Shape shape;
	float* data_device;
	float* data_host;

	bool device_allocated;
	bool host_allocated;

	Matrix(size_t x_dim = 1, size_t y_dim = 1);
	Matrix(Shape shape);

	void allocateCudaMemory();
	void allocateHostMemory();
	void allocateIfNotAllocated(Shape shape);

	void copyHostToDevice();
	void copyDeviceToHost();

	void freeCudaMemory();
	void freeHostMemory();
	void freeCudaAndHostMemory();

	float& operator[](const int index);
	const float& operator[](const int index) const;
};
