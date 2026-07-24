#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <utility>
#include <type_traits>

constexpr std::uint64_t KiB(std::size_t n) { return ((uint64_t)(n) << 10); }
constexpr std::uint64_t MiB(std::size_t n) { return ((uint64_t)(n) << 20); }
constexpr std::uint64_t GiB(std::size_t n) { return ((uint64_t)(n) << 30); }

class ArenaAllocator {
  std::byte *buffer_{};
  std::size_t offset_{0};
  std::size_t size_{0};

public:
  ArenaAllocator(std::size_t size) {
    buffer_ = reinterpret_cast<std::byte *>(::operator new(size));
    memset(buffer_, 0, size);
    size_ = size;
  }

  void *allocate(std::size_t size,
                 std::size_t alignment = alignof(std::max_align_t)) {
    uintptr_t cur{reinterpret_cast<uintptr_t>(buffer_ + offset_)};
    uintptr_t aligned{(cur + alignment - 1) & ~(alignment - 1)};
    size_t padding{aligned - cur};
    if (offset_ + padding + size > size_) {
      return nullptr;
    }
    offset_ += padding;
    void *result{buffer_ + offset_};
    offset_ += size;
    return result;
  }

  template <typename T, typename... Args> T *allocate(Args &&...args) {
    static_assert(std::is_trivially_destructible_v<T>,
                  "FATAL: Object placed in Arena must be trivially "
                  "destructible to prevent memory leaks.");
    void *object{allocate(sizeof(T), alignof(T))};
    if (object == nullptr) {
      return nullptr;
    }
    if constexpr (sizeof...(Args) == 0) {
      return ::new (object) T;
    } else {
      return ::new (object) T{std::forward<Args>(args)...};
    }
  }

  ~ArenaAllocator() { ::operator delete(buffer_); }
  ArenaAllocator(const ArenaAllocator &) = delete;
  ArenaAllocator &operator=(const ArenaAllocator &) = delete;
  ArenaAllocator(ArenaAllocator &&) = delete;
  ArenaAllocator &operator=(ArenaAllocator &&) = delete;
  void reset() { offset_ = 0; };
  std::size_t offset() const { return offset_; }
  std::size_t capacity() const { return size_; }
};
