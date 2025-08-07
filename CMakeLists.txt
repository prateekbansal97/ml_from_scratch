cmake_minimum_required(VERSION 3.10)
project(ml_from_scratch)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
#set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")

find_package(Eigen3 REQUIRED)

if(UNIX AND NOT APPLE)
    # Linux-specific flags for profiling
    message(STATUS "Detected Linux")
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -funroll-loops -DNDEBUG")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -pg -fno-omit-frame-pointer")
    
    #    set(CMAKE_BUILD_TYPE Debug)
    #define EIGEN_RUNTIME_NO_MALLOC 0
    #define EIGEN_USE_THREADS
    #define EIGEN_DONT_PARALLELIZE 0
    #define EIGEN_VECTORIZE
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG_TRAINING)
endif()


add_definitions(-DEIGEN_USE_THREADS)
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


#find_package(OpenMP REQUIRED)
#if(OpenMP_CXX_FOUND)
#    target_link_libraries(mnist PRIVATE OpenMP::OpenMP_CXX)
#endif()

target_include_directories(mnist PRIVATE
        mnist_imp/cpp
        mnist_imp/cpp/include
        ${EIGEN3_INCLUDE_DIR}
)
