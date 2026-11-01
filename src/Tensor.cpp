#include "minitensor/Tensor.h"

#include <stdexcept>
#include <utility>

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
    : Tensor(
        std::vector<std::size_t>(dimensions),
        initial_value
    ) {
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

std::size_t Tensor::offset(
    const std::vector<std::size_t>& indices
) const {
    if (indices.size() != rank()) {
        throw std::invalid_argument(
            "Tensor index rank does not match tensor rank"
        );
    }

    const auto& dimensions = shape_.dimensions();
    const auto& strides = shape_.strides();

    std::size_t result = 0;

    for (std::size_t axis = 0; axis < indices.size(); ++axis) {
        if (indices[axis] >= dimensions[axis]) {
            throw std::out_of_range(
                "Tensor index is out of bounds"
            );
        }

        result += indices[axis] * strides[axis];
    }

    return result;
}

double& Tensor::at(
    const std::vector<std::size_t>& indices
) {
    return storage_.data()[offset(indices)];
}

const double& Tensor::at(
    const std::vector<std::size_t>& indices
) const {
    return storage_.data()[offset(indices)];
}

double& Tensor::at(
    std::initializer_list<std::size_t> indices
) {
    return at(std::vector<std::size_t>(indices));
}

const double& Tensor::at(
    std::initializer_list<std::size_t> indices
) const {
    return at(std::vector<std::size_t>(indices));
}

double& Tensor::operator()(
    std::initializer_list<std::size_t> indices
) {
    return at(indices);
}

const double& Tensor::operator()(
    std::initializer_list<std::size_t> indices
) const {
    return at(indices);
}

double& Tensor::operator()(
    const std::vector<std::size_t>& indices
) {
    return at(indices);
}

const double& Tensor::operator()(
    const std::vector<std::size_t>& indices
) const {
    return at(indices);
}

double& Tensor::flat_at(std::size_t index) {
    if (index >= size()) {
        throw std::out_of_range(
            "Tensor flat index is out of bounds"
        );
    }

    return storage_.data()[index];
}

const double& Tensor::flat_at(std::size_t index) const {
    if (index >= size()) {
        throw std::out_of_range(
            "Tensor flat index is out of bounds"
        );
    }

    return storage_.data()[index];
}

void Tensor::reshape(
    const std::vector<std::size_t>& dimensions
) {
    Shape new_shape(dimensions);

    if (new_shape.num_elements() != size()) {
        throw std::invalid_argument(
            "Reshape must preserve the number of elements"
        );
    }

    shape_ = std::move(new_shape);
}

void Tensor::reshape(
    std::initializer_list<std::size_t> dimensions
) {
    reshape(std::vector<std::size_t>(dimensions));
}

} // namespace minitensor