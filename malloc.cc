#include "malloc.h"
#include <glog/logging.h>
#include <memory>

#include "cuda_device.h"

namespace majel {
namespace malloc {
namespace detail {

class DefaultAllocator {
public:
  static void* malloc(majel::Place place, size_t size);

  static void free(majel::Place, void* ptr);
};

class DefaultAllocatorMallocVisitor : public boost::static_visitor<void*> {
public:
  DefaultAllocatorMallocVisitor(size_t size) : size_(size) {}

  void* operator()(majel::CpuPlace p) {
    void* address;
    CHECK_EQ(posix_memalign(&address, 32ul, size_), 0);
    return address;
  }

#ifndef PADDLE_ONLY_CPU
  void* operator()(majel::GpuPlace p) {
    majel::gpu::detail::set_device(p.device);
    void* address = majel::gpu::detail::malloc(size_);
    return address;
  }
#endif

private:
  size_t size_;
};

class DefaultAllocatorFreeVisitor : public boost::static_visitor<void> {
public:
  DefaultAllocatorFreeVisitor(void* ptr) : ptr_(ptr) {}
  void operator()(majel::CpuPlace p) {
    if (ptr_) {
      ::free(ptr_);
    }
  }

#ifndef PADDLE_ONLY_CPU
  void operator()(majel::GpuPlace p) {
    majel::gpu::detail::set_device(p.device);
    if (ptr_) {
      majel::gpu::detail::free(ptr_);
    }
  }
#endif

private:
  void* ptr_;
};

void* DefaultAllocator::malloc(majel::Place place, size_t size) {
  DefaultAllocatorMallocVisitor visitor(size);
  return boost::apply_visitor(visitor, place);
}

void DefaultAllocator::free(majel::Place place, void* ptr) {
  DefaultAllocatorFreeVisitor visitor(ptr);
  boost::apply_visitor(visitor, place);
}

}  // namespace detail
}  // namespace malloc
}  // namespace majel
namespace majel {
namespace malloc {

void* malloc(majel::Place place, size_t size) {
  return detail::DefaultAllocator::malloc(place, size);
}

void free(majel::Place place, void* ptr) {
  detail::DefaultAllocator::free(place, ptr);
}
}  // namespace malloc
}  // namespace majel
