#pragma once

#include <memory>
#include <vector>

#include "buffer.h"
#include "cuda_device.h"
#include "ddim.h"
#include "place.h"
#include "reference.h"
#include "float16.h"

namespace majel {

template <typename T, int D>
class Array : public Buffer {
public:
  typedef T value_type;
  typedef majel::Reference<T> reference_type;
  typedef T* pointer_type;
  typedef Dim<D> index_type;

  static const int dimensionality = 0;

  Array() : ptr_(nullptr){};

  Array(const Dim<D> size, Place place)
      : Buffer(std::make_shared<Allocation>(sizeof(T) * product(size), place)),
        size_(size),
        stride_(contiguous_strides(size)) {
    ptr_ = static_cast<T*>(data()->ptr());
  }

  Array(const Dim<D> size) : Array(size, majel::get_place()) {}

  Array(std::shared_ptr<Allocation> alloc,
        const Dim<D> size,
        const Dim<D> stride,
        T* ptr)
      : Buffer(alloc), size_(size), stride_(stride), ptr_(ptr) {
    if (product(size_) > 0) {
      assert(ptr_ >= data()->ptr() && ptr_ < data()->end());
    }
  }

  Array(std::shared_ptr<Allocation> alloc,
        const Dim<D> size,
        const Dim<D> stride)
      : Array(alloc, size, stride, static_cast<T*>(alloc->ptr())) {}

  Array(std::shared_ptr<Allocation> alloc, const Dim<D> size)
      : Array(alloc, size, contiguous_strides(size)) {}

  majel::Reference<T> operator[](const Dim<D>& idx) {
    T* location = index(idx);
    return majel::Reference<T>(majel::PlacedPointer<T>(place(), location),
                               data());
  }

  T operator[](const Dim<D>& idx) const {
    T* location = index(idx);
    return majel::Reference<T>(majel::PlacedPointer<T>(place(), location),
                               data());
  }

  T* raw_ptr() const { return ptr_; }

  Dim<D> size() const { return size_; }

  int numel() const { return product(size_); }

  Dim<D> stride() const { return stride_; }

  std::shared_ptr<Allocation> data() const { return Buffer::data(); }

  const Place place() const { return Buffer::get_place(); }

  bool is_contiguous() const { return contiguous(size_, stride_); }

  void normalize() { normalize_strides(size_, stride_); }

  T* index(const Dim<D>& idx) const {
    return raw_ptr() + linearize(idx, stride_);
  }

  std::string get_type_name() const {
    if (std::is_same<T, float>::value) {
      return "float";
    } else if (std::is_same<T, int>::value) {
      return "int";
    } else if (std::is_same<T, double>::value) {
      return "double";
    } else if (std::is_same<T, majel::float16>::value) {
      return "float16";
    } else {
      return "unknown";
    }
  }

  std::string get_type_and_dim() const {
    return get_type_name() + " " + std::to_string(D);
  }

private:
  Dim<D> size_;
  Dim<D> stride_;
  T* ptr_;
};

template <typename T, int D>
T get(const Array<T, D>& arr, const Dim<D>& idx) {
  return arr[idx];
}

template <typename T, int D>
void set(Array<T, D>& arr, const Dim<D>& idx, const T& value) {
  arr[idx] = value;
};

template <typename T, int OldD, int NewD>
Array<T, NewD> reshape(const Array<T, OldD>& x, Dim<NewD> new_size) {
  CHECK(contiguous(x.size(), x.stride()))
      << "Reshaping non-contiguous Arrays is not currently implemented.";
  CHECK(x.numel() == product(new_size))
      << "Reshaping Arrays must preserve the number of elements.";
  return Array<T, NewD>(
      x.data(), new_size, contiguous_strides(new_size), x.raw_ptr());
}

template <typename T, int D>
Array<T, 1> flatten(const Array<T, D>& x) {
  return reshape(x, Dim<1>(x.numel()));
}

template <typename T, int D>
bool is_same(const Array<T, D>& a, const Array<T, D>& b) {
  return (a.data() == b.data()) && (a.size() == b.size()) &&
         (a.stride() == b.stride()) && (a.raw_ptr() == b.raw_ptr());
}

template <typename T>
Array<T, 1> make_array(const std::vector<T>& input, Place place) {
  std::shared_ptr<majel::Allocation> alloc =
      std::make_shared<majel::Allocation>(sizeof(T) * input.size(), place);

  T* ptr = static_cast<T*>(alloc->ptr());
#ifndef PADDLE_ONLY_CPU
  if (is_gpu_place(place)) {
    gpu::detail::memcpy_sync(
        ptr, input.data(), sizeof(T) * input.size(), cudaMemcpyHostToDevice);
  } else if (is_cpu_place(place)) {
    memcpy(ptr, input.data(), sizeof(T) * input.size());
  }
#else
  memcpy(ptr, input.data(), sizeof(T) * input.size());
#endif

  return Array<T, 1>(alloc, make_dim(input.size()));
}

template <typename T>
Array<T, 1> make_array(const std::vector<T>& input) {
  return make_array(input, CpuPlace());
}
}  // namespace majel
