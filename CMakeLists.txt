cmake_minimum_required(VERSION 3.10)
project(ml_from_scratch)

set(CMAKE_CXX_STANDARD 17)

add_executable(mnist
        mnist_imp/cpp/src/main.cpp
        mnist_imp/cpp/src/dataset.cpp
        mnist_imp/cpp/src/activation.cpp
        mnist_imp/cpp/include/activation.h
        mnist_imp/cpp/include/model.h
        mnist_imp/cpp/src/model.cpp
)

target_include_directories(mnist PRIVATE
        mnist_imp/cpp
        mnist_imp/cpp/include
        /usr/local/include/eigen3
)