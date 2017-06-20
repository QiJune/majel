#pragma once

#ifndef PADDLE_ONLY_CPU
#include <cuda_runtime.h>

namespace majel {
namespace gpu {
namespace detail {

const char* get_device_error_string();

const char* get_device_error_string(size_t err);

void* malloc(size_t size);

void free(void* ptr);

void memcpy_sync(void* dst,
                 const void* src,
                 size_t count,
                 enum cudaMemcpyKind kind);

void set_device(int device);

}  // namespace detail
}  // namespace gpu
}  // namespace majel
#endif
