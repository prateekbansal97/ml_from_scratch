//
// Created by Prateek Bansal on 8/2/25.
//

#ifndef ML_FROM_SCRATCH_EXCEPTION_H
#define ML_FROM_SCRATCH_EXCEPTION_H


#include <stdexcept>
#include <string>

class ShapeMismatchError : public std::runtime_error {
public:
    explicit ShapeMismatchError(const std::string& message);

};

#endif //ML_FROM_SCRATCH_EXCEPTION_H
