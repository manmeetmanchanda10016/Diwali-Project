
#include "minitensor/Ops.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace minitensor;

int main()
{
    // Test 1: Addition
    Tensor a({2, 3});
    Tensor b({2, 3});

    double values_a[] = {1, 2, 3, 4, 5, 6};
    double values_b[] = {6, 5, 4, 3, 2, 1};

    for (std::size_t i = 0; i < 6; ++i)
    {
        a.data()[i] = values_a[i];
        b.data()[i] = values_b[i];
    }

    Tensor sum = a + b;

    for (std::size_t i = 0; i < 6; ++i)
        assert(sum.data()[i] == 7.0);

    // Test 2: Subtraction
    Tensor difference = a - b;

    double expected_difference[] = {-5, -3, -1, 1, 3, 5};

    for (std::size_t i = 0; i < 6; ++i)
        assert(difference.data()[i] == expected_difference[i]);

    // Test 3: Element-wise multiplication
    Tensor product = a * b;

    double expected_product[] = {6, 10, 12, 12, 10, 6};

    for (std::size_t i = 0; i < 6; ++i)
        assert(product.data()[i] == expected_product[i]);

    // Test 4: Named addition
    Tensor named_sum = add(a, b);
    assert(named_sum.data()[0] == 7.0);

    // Test 5: Shape mismatch
    bool error_detected = false;

    try
    {
        Tensor c({3, 2});
        Tensor invalid = a + c;
    }
    catch (const std::invalid_argument&)
    {
        error_detected = true;
    }

    assert(error_detected);

    // Test 6: Matrix multiplication
    Tensor m1({2, 3});
    Tensor m2({3, 2});

    double matrix_a[] = {1, 2, 3, 4, 5, 6};
    double matrix_b[] = {7, 8, 9, 10, 11, 12};

    for (std::size_t i = 0; i < 6; ++i)
    {
        m1.data()[i] = matrix_a[i];
        m2.data()[i] = matrix_b[i];
    }

    Tensor matrix_result = matmul(m1, m2);

    double expected_matrix[] = {58, 64, 139, 154};

    for (std::size_t i = 0; i < 4; ++i)
    {
        assert(std::abs(matrix_result.data()[i] -
                        expected_matrix[i]) < 1e-9);
    }

    // Test 7: Invalid matrix dimensions
    error_detected = false;

    try
    {
        Tensor invalid = matmul(m1, m1);
    }
    catch (const std::invalid_argument&)
    {
        error_detected = true;
    }

    assert(error_detected);

    // Test 8: Non-2D matrix multiplication
    error_detected = false;

    try
    {
        Tensor one_dimensional({3});
        Tensor invalid = matmul(one_dimensional, one_dimensional);
    }
    catch (const std::invalid_argument&)
    {
        error_detected = true;
    }

    assert(error_detected);

    // Test 9: Inputs remain unchanged
    assert(a.data()[0] == 1.0);
    assert(b.data()[0] == 6.0);

    // Test 10: OpenMP matrix multiplication
    Tensor parallel_result = matmul_parallel(m1, m2);

    for (std::size_t i = 0; i < 4; ++i)
    {
        assert(std::abs(parallel_result.data()[i] -
                        matrix_result.data()[i]) < 1e-9);
    }

    // Test 11: OpenMP invalid matrix dimensions
    error_detected = false;

    try
    {
        Tensor invalid = matmul_parallel(m1, m1);
    }
    catch (const std::invalid_argument&)
    {
        error_detected = true;
    }

    assert(error_detected);

    // Test 12: OpenMP non-2D matrix multiplication
    error_detected = false;

    try
    {
        Tensor one_dimensional({3});
        Tensor invalid = matmul_parallel(
            one_dimensional,
            one_dimensional
        );
    }
    catch (const std::invalid_argument&)
    {
        error_detected = true;
    }

    assert(error_detected);

    // Test 13: Parallel multiplication preserves inputs
    assert(m1.data()[0] == 1.0);
    assert(m2.data()[0] == 7.0);

    return 0;
}
