#include "minitensor/Tensor.h"

#include <cassert>
#include <stdexcept>
#include <vector>

using minitensor::Tensor;

int main() {
    Tensor tensor({2, 3, 4});

    const std::vector<std::size_t> first_strides{12, 4, 1};
    const std::vector<std::size_t> coordinates{1, 2, 3};

    assert(tensor.shape().strides() == first_strides);
    assert(tensor.size() == 24);
    assert(tensor.rank() == 3);

    tensor({1, 2, 3}) = 42.0;

    assert(tensor.flat_at(23) == 42.0);
    assert(tensor.at(coordinates) == 42.0);
    assert(tensor(coordinates) == 42.0);

    const Tensor& const_tensor = tensor;

    assert(const_tensor.at(coordinates) == 42.0);
    assert(const_tensor(coordinates) == 42.0);
    assert(const_tensor.flat_at(23) == 42.0);

    tensor.reshape({4, 6});

    const std::vector<std::size_t> reshaped_strides{6, 1};
    const std::vector<std::size_t> reshaped_coordinates{3, 5};

    assert(tensor.shape().strides() == reshaped_strides);
    assert(tensor.rank() == 2);
    assert(tensor.size() == 24);
    assert(tensor.at(reshaped_coordinates) == 42.0);
    assert(tensor(reshaped_coordinates) == 42.0);

    bool error_detected = false;

    try {
        tensor.at({4, 0});
    } catch (const std::out_of_range&) {
        error_detected = true;
    }

    assert(error_detected);

    error_detected = false;

    try {
        tensor.at({3});
    } catch (const std::invalid_argument&) {
        error_detected = true;
    }

    assert(error_detected);

    error_detected = false;

    try {
        tensor.flat_at(24);
    } catch (const std::out_of_range&) {
        error_detected = true;
    }

    assert(error_detected);

    error_detected = false;

    try {
        tensor.reshape({5, 5});
    } catch (const std::invalid_argument&) {
        error_detected = true;
    }

    assert(error_detected);

    assert(tensor.size() == 24);
    assert(tensor(reshaped_coordinates) == 42.0);

    Tensor scalar({});

    assert(scalar.rank() == 0);
    assert(scalar.size() == 1);
    assert(scalar.at({}) == 0.0);

    scalar.at({}) = 7.0;

    assert(scalar.flat_at(0) == 7.0);

    Tensor empty({2, 0, 3});

    assert(empty.size() == 0);

    error_detected = false;

    try {
        empty.at({0, 0, 0});
    } catch (const std::out_of_range&) {
        error_detected = true;
    }

    assert(error_detected);

    return 0;
}