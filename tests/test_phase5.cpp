#include "minitensor/Ops.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

using namespace minitensor;

bool nearly_equal(double a, double b) {
    return std::abs(a - b) < 1e-9;
}

void compare_tensors(const Tensor& a, const Tensor& b) {
    assert(a.shape().dimensions() == b.shape().dimensions());
    assert(a.size() == b.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        assert(nearly_equal(a.data()[i], b.data()[i]));
    }
}

int main() {
    // Nonuniform inputs help catch incorrect matrix indexing.
    Tensor a({2, 3});
    Tensor b({3, 2});

    const double a_values[] = {1, 2, 3, 4, 5, 6};
    const double b_values[] = {7, 8, 9, 10, 11, 12};

    for (std::size_t i = 0; i < a.size(); ++i) {
        a.data()[i] = a_values[i];
    }

    for (std::size_t i = 0; i < b.size(); ++i) {
        b.data()[i] = b_values[i];
    }

    const Tensor sequential = matmul(a, b);
    const Tensor parallel = matmul_parallel(a, b);

    compare_tensors(sequential, parallel);

    const double expected[] = {58, 64, 139, 154};

    for (std::size_t i = 0; i < sequential.size(); ++i) {
        assert(nearly_equal(sequential.data()[i], expected[i]));
    }

    // Test larger inputs as well.
    Tensor large_a({32, 32});
    Tensor large_b({32, 32});

    for (std::size_t i = 0; i < large_a.size(); ++i) {
        large_a.data()[i] = static_cast<double>((i % 7) + 1);
        large_b.data()[i] =
            static_cast<double>(static_cast<int>(i % 5) - 2);
    }

    compare_tensors(
        matmul(large_a, large_b),
        matmul_parallel(large_a, large_b)
    );

    return 0;
}