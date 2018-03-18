#pragma once

namespace nn_utils {

	void throwIfDeviceErrorsOccurred(const char* exception_message);

	struct Shape {
		size_t x, y, z;

		Shape(size_t x = 1, size_t y = 1, size_t z = 1);
	};

	struct Tensor3D {
		Shape shape;
		float* data;

		Tensor3D(size_t x_dim = 1, size_t y_dim = 1, size_t z_dim = 1);
		Tensor3D(Shape shape);

		void allocateCudaMemory();
		void freeCudaMemory();
	};

}
