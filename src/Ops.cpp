
#include "minitensor/Ops.h"
#include <omp.h>
#include <stdexcept>
#include <vector>

namespace minitensor {

namespace {

void validate_same_shape(const Tensor& a, const Tensor& b)
{
    if (a.shape().dimensions() != b.shape().dimensions())
    {
        throw std::invalid_argument(
            "Element-wise operations require identical tensor shapes"
        );
    }
}

}

// Element-wise addition
Tensor add(const Tensor& a, const Tensor& b)
{
    validate_same_shape(a, b);

    Tensor result(a.shape().dimensions());

    const double* a_data = a.data();
    const double* b_data = b.data();
    double* result_data = result.data();

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        result_data[i] = a_data[i] + b_data[i];
    }

    return result;
}

// Element-wise subtraction
Tensor subtract(const Tensor& a, const Tensor& b)
{
    validate_same_shape(a, b);

    Tensor result(a.shape().dimensions());

    const double* a_data = a.data();
    const double* b_data = b.data();
    double* result_data = result.data();

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        result_data[i] = a_data[i] - b_data[i];
    }

    return result;
}

// Element-wise multiplication
Tensor multiply(const Tensor& a, const Tensor& b)
{
    validate_same_shape(a, b);

    Tensor result(a.shape().dimensions());

    const double* a_data = a.data();
    const double* b_data = b.data();
    double* result_data = result.data();

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        result_data[i] = a_data[i] * b_data[i];
    }

    return result;
}

// Hybrid matrix multiplication
Tensor matmul(const Tensor& a, const Tensor& b)
{
    if (a.rank() != 2 || b.rank() != 2)
    {
        throw std::invalid_argument(
            "Matrix multiplication requires 2D tensors"
        );
    }

    const auto& a_shape = a.shape().dimensions();
    const auto& b_shape = b.shape().dimensions();

    const std::size_t rows_a = a_shape[0];
    const std::size_t cols_a = a_shape[1];
    const std::size_t rows_b = b_shape[0];
    const std::size_t cols_b = b_shape[1];

    if (cols_a != rows_b)
    {
        throw std::invalid_argument(
            "Matrix dimensions are incompatible for multiplication"
        );
    }

    constexpr long double parallel_threshold =
        256.0L * 256.0L * 256.0L;

    const long double estimated_work =
        static_cast<long double>(rows_a) *
        static_cast<long double>(cols_a) *
        static_cast<long double>(cols_b);

    // Use OpenMP for sufficiently large workloads
    if (rows_a >= 128 && estimated_work >= parallel_threshold)
    {
        return matmul_parallel(a, b);
    }

    // Normal multiplication for smaller workloads
    Tensor result({rows_a, cols_b}, 0.0);

    const double* a_data = a.data();
    const double* b_data = b.data();
    double* result_data = result.data();

    for (std::size_t i = 0; i < rows_a; ++i)
    {
        for (std::size_t k = 0; k < cols_a; ++k)
        {
            const double value = a_data[i * cols_a + k];

            for (std::size_t j = 0; j < cols_b; ++j)
            {
                result_data[i * cols_b + j] +=
                    value * b_data[k * cols_b + j];
            }
        }
    }

    return result;
}

// OpenMP parallel matrix multiplication
Tensor matmul_parallel(const Tensor& a, const Tensor& b)
{
    if (a.rank() != 2 || b.rank() != 2)
    {
        throw std::invalid_argument(
            "Matrix multiplication requires 2D tensors"
        );
    }

    const auto& a_shape = a.shape().dimensions();
    const auto& b_shape = b.shape().dimensions();

    const std::size_t rows_a = a_shape[0];
    const std::size_t cols_a = a_shape[1];
    const std::size_t rows_b = b_shape[0];
    const std::size_t cols_b = b_shape[1];

    if (cols_a != rows_b)
    {
        throw std::invalid_argument(
            "Invalid matrix dimensions"
        );
    }

    Tensor result({rows_a, cols_b}, 0.0);

    const double* A = a.data();
    const double* B = b.data();
    double* C = result.data();

    #pragma omp parallel for schedule(static)
    for (long long i = 0;
         i < static_cast<long long>(rows_a); ++i)
    {
        for (std::size_t k = 0; k < cols_a; ++k)
        {
            const double value = A[i * cols_a + k];

            for (std::size_t j = 0; j < cols_b; ++j)
            {
                C[i * cols_b + j] +=
                    value * B[k * cols_b + j];
            }
        }
    }

    return result;
}

// Addition operator
Tensor operator+(const Tensor& a, const Tensor& b)
{
    return add(a, b);
}

// Subtraction operator
Tensor operator-(const Tensor& a, const Tensor& b)
{
    return subtract(a, b);
}

// Multiplication operator
Tensor operator*(const Tensor& a, const Tensor& b)
{
    return multiply(a, b);
}

}
