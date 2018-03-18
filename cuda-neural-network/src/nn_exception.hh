#pragma once

#include <exception>

class NNException : std::exception {
private:
	const char* exception_message;

public:
	NNException(const char* exception_message) :
		exception_message(exception_message)
	{ }

	virtual const char* what() const throw()
	{
		return exception_message;
	}
};
