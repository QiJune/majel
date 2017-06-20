#include "cuda_device.h"
#include <glog/logging.h>

#define CHECK_CUDA(cudaFunc)                                         \
  do {                                                               \
    cudaError_t cudaStat = cudaFunc;                                 \
    CHECK_EQ(cudaSuccess, cudaStat) << "Cuda Error: "                \
                                    << cudaGetErrorString(cudaStat); \
  } while (0)

#ifndef PADDLE_ONLY_CPU

namespace majel {
namespace gpu {
namespace detail {

const char* get_device_error_string() {
  cudaError_t err = cudaGetLastError();
  return cudaGetErrorString(err);
}

const char* get_device_error_string(size_t err) {
  return cudaGetErrorString((cudaError_t)err);
}

void* malloc(size_t size) {
  void* dest_d;

  CHECK(size) << __func__ << ": the size for device memory is 0, please check.";
  CHECK_CUDA(cudaMalloc((void**)&dest_d, size));
  return dest_d;
}

void free(void* dest_d) {
  CHECK_NOTNULL(dest_d);

  cudaError_t err = cudaFree(dest_d);
  CHECK(cudaSuccess == err || cudaErrorCudartUnloading == err)
      << get_device_error_string();
}

void memcpy_sync(void* dst,
                 const void* src,
                 size_t count,
                 enum cudaMemcpyKind kind) {
  cudaMemcpy(dst, src, count, kind);
}

void set_device(int device) { cudaSetDevice(device); }

}  // namespace detail
}  // namespace gpu
}  // namespace majel
#endif