#include "minitensor/Storage.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace minitensor {

void Storage::allocate(std::size_t size) {
    if (size == 0) {
        buffer_ = nullptr;
        size_ = 0;
        return;
    }

    if (size > std::numeric_limits<std::size_t>::max() /
                   sizeof(double)) {
        throw std::overflow_error("Storage size overflow");
    }

    const std::size_t bytes = size * sizeof(double);

    buffer_ = static_cast<double*>(
        ::operator new(
            bytes,
            std::align_val_t{ALIGNMENT}
        )
    );

    size_ = size;
}

void Storage::release() noexcept {
    if (buffer_ != nullptr) {
        ::operator delete(
            buffer_,
            std::align_val_t{ALIGNMENT}
        );
    }

    buffer_ = nullptr;
    size_ = 0;
}

Storage::Storage(
    std::size_t size,
    double initialValue
)
    : buffer_(nullptr),
      size_(0) {
    allocate(size);

    if (size_ > 0) {
        std::fill_n(buffer_, size_, initialValue);
    }
}

Storage::~Storage() {
    release();
}

Storage::Storage(const Storage& other)
    : buffer_(nullptr),
      size_(0) {
    allocate(other.size_);

    if (size_ > 0) {
        std::memcpy(
            buffer_,
            other.buffer_,
            size_ * sizeof(double)
        );
    }
}

Storage& Storage::operator=(const Storage& other) {
    if (this != &other) {
        Storage temp(other);
        swap(temp);
    }

    return *this;
}

Storage::Storage(Storage&& other) noexcept
    : buffer_(other.buffer_),
      size_(other.size_) {
    other.buffer_ = nullptr;
    other.size_ = 0;
}

Storage& Storage::operator=(Storage&& other) noexcept {
    if (this != &other) {
        release();

        buffer_ = other.buffer_;
        size_ = other.size_;

        other.buffer_ = nullptr;
        other.size_ = 0;
    }

    return *this;
}

std::size_t Storage::size() const noexcept {
    return size_;
}

std::size_t Storage::capacity() const noexcept {
    return size_;
}

double* Storage::data() noexcept {
    return buffer_;
}

const double* Storage::data() const noexcept {
    return buffer_;
}

void Storage::fill(double value) {
    if (size_ > 0) {
        std::fill_n(buffer_, size_, value);
    }
}

void Storage::swap(Storage& other) noexcept {
    std::swap(buffer_, other.buffer_);
    std::swap(size_, other.size_);
}

} // namespace minitensor
