//
// Created by Prateek Bansal on 8/2/25.
//

#include "exception.h"
#include <stdexcept>
#include <string>

ShapeMismatchError::ShapeMismatchError(const std::string& message)
        : std::runtime_error(message) {};