#pragma once

namespace nn_utils {

	void throwIfDeviceErrorsOccurred(const char* exception_message);

	struct Shape {
		size_t x, y, z;

		Shape(size_t x = 1, size_t y = 1, size_t z = 1);
	};

	struct Tensor3D {
		Shape shape;
		float* data_device;
		float* data_host;

		bool device_allocated;
		bool host_allocated;

		Tensor3D(size_t x_dim = 1, size_t y_dim = 1, size_t z_dim = 1);
		Tensor3D(Shape shape);

		void allocateCudaMemory();
		void allocateHostMemory();
		void allocateIfNotAllocated(nn_utils::Shape shape);

		void copyHostToDevice();
		void copyDeviceToHost();

		void freeCudaMemory();
		void freeHostMemory();
		void freeCudaAndHostMemory();

		float& operator[](const int index);
		const float& operator[](const int index) const;
	};

	float binaryCrossEntropyCost(Tensor3D predictions, Tensor3D target);
	Tensor3D dBinaryCrossEntropyCost(Tensor3D predictions, Tensor3D target, Tensor3D dY);
}
