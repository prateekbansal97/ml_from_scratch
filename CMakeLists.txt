cmake_minimum_required(VERSION 3.16)
project(ml_from_scratch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)

# ======================================
# >>> DEBUG CONFIG (UNCOMMENT TO USE)
# ======================================
# set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
# set(CDEBUG_FLAGS
#     -O0
#     -g
#     -fno-omit-frame-pointer
# )
# # Optional sanitizers:
# # set(CDEBUG_SANITIZERS -fsanitize=address -fsanitize=undefined)
# add_compile_definitions(DEBUG_TRAINING)
# message(STATUS "Configuring in DEBUG mode")

# ======================================
# >>> RELEASE CONFIG (UNCOMMENT TO USE)
# ======================================
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CRELEASE_FLAGS
        -O3
        -march=native
        -ffast-math
        -funroll-loops
        -DNDEBUG
)
message(STATUS "Configuring in RELEASE mode")

# ======================================
# Target
# ======================================
add_executable(mnist
        mnist_imp/cpp/src/main.cpp
        mnist_imp/cpp/src/dataset.cpp
        mnist_imp/cpp/src/activation.cpp
        mnist_imp/cpp/include/activation.h
        mnist_imp/cpp/include/model.h
        mnist_imp/cpp/src/model.cpp
        mnist_imp/cpp/src/exception.cpp
        mnist_imp/cpp/include/exception.h
)

target_include_directories(mnist PRIVATE
        mnist_imp/cpp
        mnist_imp/cpp/include
        ${EIGEN3_INCLUDE_DIR}
)

target_compile_definitions(mnist PRIVATE EIGEN_USE_THREADS)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(mnist PRIVATE ${CDEBUG_FLAGS} ${CDEBUG_SANITIZERS})
    target_link_options(mnist PRIVATE ${CDEBUG_SANITIZERS})
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(mnist PRIVATE ${CRELEASE_FLAGS})
endif()