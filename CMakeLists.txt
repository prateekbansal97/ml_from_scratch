cmake_minimum_required(VERSION 4.0)
project(ml_from_scratch)

set(CMAKE_CXX_STANDARD 14)

include_directories(mnist_imp/cpp)
include_directories(mnist_imp/cpp/include)

include_directories(/usr/local/include/eigen3)

add_executable(mnist
        mnist_imp/cpp/src/main.cpp
        mnist_imp/cpp/src/dataset.cpp
        )
