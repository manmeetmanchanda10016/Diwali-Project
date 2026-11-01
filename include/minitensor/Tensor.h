#ifndef MINITENSOR_TENSOR_H
#define MINITENSOR_TENSOR_H

#include "minitensor/Shape.h"
#include "minitensor/Storage.h"

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace minitensor {

class Tensor {
private:
    Shape shape_;
    Storage storage_;

public:
    explicit Tensor(
        const std::vector<std::size_t>& dimensions,
        double initial_value = 0.0
    );

    Tensor(
        std::initializer_list<std::size_t> dimensions,
        double initial_value = 0.0
    );

    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    std::size_t size() const noexcept;
    std::size_t rank() const noexcept;

    const Shape& shape() const noexcept;

    double* data() noexcept;
    const double* data() const noexcept;

    void fill(double value);
};

} // namespace minitensor

#endif
