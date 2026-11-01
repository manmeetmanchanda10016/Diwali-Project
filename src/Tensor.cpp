#include "minitensor/Tensor.h"

namespace minitensor {

Tensor::Tensor(
    const std::vector<std::size_t>& dimensions,
    double initial_value
)
    : shape_(dimensions),
      storage_(shape_.num_elements(), initial_value) {
}

Tensor::Tensor(
    std::initializer_list<std::size_t> dimensions,
    double initial_value
)
    : Tensor(std::vector<std::size_t>(dimensions), initial_value) {
}

std::size_t Tensor::size() const noexcept {
    return storage_.size();
}

std::size_t Tensor::rank() const noexcept {
    return shape_.rank();
}

const Shape& Tensor::shape() const noexcept {
    return shape_;
}

double* Tensor::data() noexcept {
    return storage_.data();
}

const double* Tensor::data() const noexcept {
    return storage_.data();
}

void Tensor::fill(double value) {
    storage_.fill(value);
}

} // namespace minitensor
