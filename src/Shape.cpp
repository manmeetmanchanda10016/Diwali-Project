#include "minitensor/Shape.h"

#include <limits>
#include <stdexcept>

namespace minitensor {

std::size_t Shape::calculate_elements(
    const std::vector<std::size_t>& dimensions
) {
    // An empty shape represents a scalar containing one element.
    if (dimensions.empty()) {
        return 1;
    }

    // Any zero dimension makes the tensor empty.
    for (const std::size_t dim : dimensions) {
        if (dim == 0) {
            return 0;
        }
    }

    std::size_t total = 1;

    for (const std::size_t dim : dimensions) {
        if (total > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::overflow_error("Tensor dimensions overflow size_t");
        }
        total *= dim;
    }

    return total;
}

Shape::Shape(const std::vector<std::size_t>& dimensions)
    : dimensions_(dimensions),
      num_elements_(calculate_elements(dimensions_)) {
}

Shape::Shape(std::initializer_list<std::size_t> dimensions)
    : Shape(std::vector<std::size_t>(dimensions)) {
}

std::size_t Shape::rank() const noexcept {
    return dimensions_.size();
}

std::size_t Shape::num_elements() const noexcept {
    return num_elements_;
}

const std::vector<std::size_t>& Shape::dimensions() const noexcept {
    return dimensions_;
}

} // namespace minitensor
