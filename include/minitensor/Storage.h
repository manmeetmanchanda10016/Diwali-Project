
#ifndef MINITENSOR_STORAGE_H
#define MINITENSOR_STORAGE_H

#include <cstddef>

namespace minitensor {

class Storage {
private:
    friend class Tensor;
    struct UninitializedTag {};
    Storage(UninitializedTag, std::size_t size);

    static constexpr std::size_t ALIGNMENT = 64;

    double* buffer_;
    std::size_t size_;

    void allocate(std::size_t size);
    void release() noexcept;

public:
    explicit Storage(
        std::size_t size,
        double initialValue = 0.0
    );

    ~Storage();

    Storage(const Storage& other);
    Storage& operator=(const Storage& other);

    Storage(Storage&& other) noexcept;
    Storage& operator=(Storage&& other) noexcept;

    std::size_t size() const noexcept;
    std::size_t capacity() const noexcept;

    double* data() noexcept;
    const double* data() const noexcept;

    void fill(double value);

    void swap(Storage& other) noexcept;
};

}

#endif
