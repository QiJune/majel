cmake_minimum_required(VERSION 3.0)
project(majel)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(PROJ_ROOT ${CMAKE_CURRENT_SOURCE_DIR})


include(generic)


find_package(Glog REQUIRED)
find_package(Gflags REQUIRED)
find_package(Boost REQUIRED)
find_package(GTEST REQUIRED)
find_package(CUDA QUIET)

option(WITH_GPU         "Compile PaddlePaddle with NVIDIA GPU"          ${CUDA_FOUND})
option(WITH_TESTING     "Compile PaddlePaddle with unit testing"        ON)


if(NOT WITH_GPU)
  add_definitions(-DPADDLE_ONLY_CPU)
endif()  



cc_library(place SRCS place.cc)
cc_test(place_test SRCS place_test.cc DEPS place glog gflags)

cc_library(ddim SRCS ddim.cc)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)

if (WITH_GPU)
  nv_library(malloc SRCS malloc.cc)
else()
  cc_library(malloc SRCS malloc.cc)
endif()

cc_library(cuda_device SRCS cuda_device.cc)

cc_library(allocation SRCS allocation.cc)
cc_test(allocation_test SRCS allocation_test.cc DEPS allocation malloc cuda_device place glog gflags)

nv_test(dim_test SRCS dim_test.cu DEPS ddim)
nv_test(array_test SRCS array_test.cc DEPS allocation malloc ddim cuda_device place glog gflags)

nv_library(darray SRCS darray.cc)
nv_test(darray_test SRCS darray_test.cc DEPS darray allocation malloc ddim cuda_device place glog gflags)
