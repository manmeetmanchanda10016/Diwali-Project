
#include "minitensor/Ops.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace minitensor;

bool nearly_equal(double a, double b) {
    return std::abs(a - b) < 1e-9;
}

void compare_tensors(const Tensor& a, const Tensor& b) {
    assert(a.shape().dimensions() == b.shape().dimensions());
    assert(a.size() == b.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        assert(nearly_equal(a.flat_at(i), b.flat_at(i)));
    }
}

template <typename Exception, typename Function>
void expect_exception(Function function) {
    bool caught = false;

    try {
        function();
    } catch (const Exception&) {
        caught = true;
    }

    assert(caught);
}

void test_tensor_edge_cases() {
    Tensor scalar(std::vector<std::size_t>{}, 5.0);

    assert(scalar.rank() == 0);
    assert(scalar.size() == 1);
    assert(nearly_equal(scalar.flat_at(0), 5.0));

    Tensor empty({2, 0, 3});

    assert(empty.size() == 0);
    assert(empty.rank() == 3);

    expect_exception<std::out_of_range>([&]() {
        scalar.flat_at(1);
    });

    expect_exception<std::out_of_range>([]() {
        Tensor a({2, 2});
        a.at({2, 0});
    });

    expect_exception<std::invalid_argument>([]() {
        Tensor a({2, 3});
        a.reshape({4, 2});
    });

    Tensor a({2, 3}, 4.0);
    a.reshape({3, 2});

    assert(a.shape().dimensions() ==
           std::vector<std::size_t>({3, 2}));
    assert(a.size() == 6);
}

void test_elementwise_operations() {
    Tensor a({2, 3});
    Tensor b({2, 3});

    for (std::size_t i = 0; i < 6; ++i) {
        a.flat_at(i) = static_cast<double>(i + 1);
        b.flat_at(i) = static_cast<double>((i + 1) * 2);
    }

    Tensor added = add(a, b);
    Tensor subtracted = subtract(b, a);
    Tensor multiplied = multiply(a, b);

    for (std::size_t i = 0; i < 6; ++i) {
        const double x = static_cast<double>(i + 1);

        assert(nearly_equal(added.flat_at(i), 3.0 * x));
        assert(nearly_equal(subtracted.flat_at(i), x));

        // a[i] = x and b[i] = 2*x, so product = 2*x*x.
        assert(nearly_equal(
            multiplied.flat_at(i),
            2.0 * x * x
        ));
    }

    compare_tensors(a + b, added);
    compare_tensors(b - a, subtracted);
    compare_tensors(a * b, multiplied);

    expect_exception<std::invalid_argument>([]() {
        Tensor x({2, 3});
        Tensor y({3, 2});
        add(x, y);
    });
}

void test_broadcasting() {
    Tensor matrix({2, 3});
    Tensor row({3});

    for (std::size_t i = 0; i < 6; ++i) {
        matrix.flat_at(i) = static_cast<double>(i + 1);
    }

    row.flat_at(0) = 10.0;
    row.flat_at(1) = 20.0;
    row.flat_at(2) = 30.0;

    Tensor result = add_broadcast(matrix, row);

    const double expected[] = {
        11, 22, 33,
        14, 25, 36
    };

    assert(result.shape().dimensions() ==
           std::vector<std::size_t>({2, 3}));

    for (std::size_t i = 0; i < result.size(); ++i) {
        assert(nearly_equal(result.flat_at(i), expected[i]));
    }

    expect_exception<std::invalid_argument>([]() {
        Tensor x({2, 3});
        Tensor y({4});
        add_broadcast(x, y);
    });
}

void test_transpose() {
    Tensor a({2, 3});

    const double values[] = {1, 2, 3, 4, 5, 6};

    for (std::size_t i = 0; i < 6; ++i) {
        a.flat_at(i) = values[i];
    }

    Tensor result = transpose(a);

    assert(result.shape().dimensions() ==
           std::vector<std::size_t>({3, 2}));

    const double expected[] = {1, 4, 2, 5, 3, 6};

    for (std::size_t i = 0; i < 6; ++i) {
        assert(nearly_equal(result.flat_at(i), expected[i]));
    }
}

void test_reductions() {
    Tensor a({2, 3});

    const double values[] = {1, 2, 3, 4, 5, 6};

    for (std::size_t i = 0; i < 6; ++i) {
        a.flat_at(i) = values[i];
    }

    assert(nearly_equal(sum(a), 21.0));
    assert(nearly_equal(mean(a), 3.5));
    assert(nearly_equal(min(a), 1.0));
    assert(nearly_equal(max(a), 6.0));

    Tensor empty({0});

    assert(nearly_equal(sum(empty), 0.0));

    expect_exception<std::invalid_argument>([&]() {
        mean(empty);
    });

    expect_exception<std::invalid_argument>([&]() {
        min(empty);
    });

    expect_exception<std::invalid_argument>([&]() {
        max(empty);
    });
}

void test_matrix_multiplication() {
    Tensor a({2, 3});
    Tensor b({3, 2});

    const double a_values[] = {1, 2, 3, 4, 5, 6};
    const double b_values[] = {7, 8, 9, 10, 11, 12};

    for (std::size_t i = 0; i < 6; ++i) {
        a.flat_at(i) = a_values[i];
        b.flat_at(i) = b_values[i];
    }

    Tensor expected({2, 2});

    expected.flat_at(0) = 58;
    expected.flat_at(1) = 64;
    expected.flat_at(2) = 139;
    expected.flat_at(3) = 154;

    compare_tensors(matmul(a, b), expected);
    compare_tensors(matmul_parallel(a, b), expected);

    for (std::size_t block_size : {1U, 2U, 3U, 4U, 8U}) {
        compare_tensors(
            matmul_blocked(a, b, block_size),
            expected
        );

        compare_tensors(
            matmul_parallel_blocked(a, b, block_size),
            expected
        );
    }

    expect_exception<std::invalid_argument>([]() {
        Tensor x({2, 3});
        Tensor y({4, 2});
        matmul(x, y);
    });
}

void test_parallel_and_blocked_consistency() {
    Tensor a({32, 32});
    Tensor b({32, 32});

    for (std::size_t i = 0; i < a.size(); ++i) {
        a.flat_at(i) =
            static_cast<double>((i % 7) + 1);

        b.flat_at(i) =
            static_cast<double>(static_cast<int>(i % 5) - 2);
    }

    const Tensor reference = matmul(a, b);

    compare_tensors(matmul_parallel(a, b), reference);

    for (std::size_t block_size : {8U, 16U, 32U, 64U}) {
        compare_tensors(
            matmul_blocked(a, b, block_size),
            reference
        );

        compare_tensors(
            matmul_parallel_blocked(a, b, block_size),
            reference
        );
    }
}

int main() {
    test_tensor_edge_cases();
    test_elementwise_operations();
    test_broadcasting();
    test_transpose();
    test_reductions();
    test_matrix_multiplication();
    test_parallel_and_blocked_consistency();

    return 0;
}
