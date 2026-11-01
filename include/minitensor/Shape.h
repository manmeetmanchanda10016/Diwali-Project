#ifndef MINITENSOR_SHAPE_H
#define MINITENSOR_SHAPE_H

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace minitensor {

class Shape {
private:
    std::vector<std::size_t> dimensions_;
    std::vector<std::size_t> strides_;
    std::size_t num_elements_;

    static std::size_t calculate_elements(
        const std::vector<std::size_t>& dimensions
    );

public:
    explicit Shape(
        const std::vector<std::size_t>& dimensions
    );

    Shape(
        std::initializer_list<std::size_t> dimensions
    );

    std::size_t rank() const noexcept;
    std::size_t num_elements() const noexcept;

    const std::vector<std::size_t>& dimensions() const noexcept;

    const std::vector<std::size_t>& strides() const noexcept;
};

} // namespace minitensor

#endif