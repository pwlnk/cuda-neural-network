#include "nn_utils.hh"
#include "nn_exception.hh"

namespace nn_utils {

	void throwIfDeviceErrorsOccurred(const char* exception_message) {
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			throw NNException(exception_message);
		}
	}

}
