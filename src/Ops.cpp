#include "minitensor/Ops.h"

#include <algorithm>
#include <cstddef>
#include <omp.h>
#include <stdexcept>
#include <vector>

namespace minitensor {

namespace detail {

// Only kernels that overwrite every output element may use this factory.
// Public tensor/storage constructors retain their initialized behavior.
struct TensorAccess
{
    static Tensor empty(const std::vector<std::size_t>& dimensions)
    {
        return Tensor(Tensor::UninitializedTag{}, dimensions);
    }
};

} // namespace detail

namespace {

void validate_same_shape(const Tensor& a, const Tensor& b)
{
    if (a.shape().dimensions() != b.shape().dimensions())
        throw std::invalid_argument(
            "Element-wise operations require identical tensor shapes");
}

std::vector<std::size_t> broadcast_shape(
    const Tensor& a, const Tensor& b)
{
    const auto& ad = a.shape().dimensions();
    const auto& bd = b.shape().dimensions();

    const std::size_t rank = std::max(ad.size(), bd.size());
    std::vector<std::size_t> result(rank);

    for (std::size_t i = 0; i < rank; ++i)
    {
        const std::size_t ai =
            i < rank - ad.size() ? 1 : ad[i - (rank - ad.size())];

        const std::size_t bi =
            i < rank - bd.size() ? 1 : bd[i - (rank - bd.size())];

        if (ai == bi)
            result[i] = ai;
        else if (ai == 1)
            result[i] = bi;
        else if (bi == 1)
            result[i] = ai;
        else
            throw std::invalid_argument(
                "Tensor shapes are not broadcast-compatible");
    }

    return result;
}

struct AddOp
{
    double operator()(double x, double y) const { return x + y; }
};

struct SubtractOp
{
    double operator()(double x, double y) const { return x - y; }
};

struct MultiplyOp
{
    double operator()(double x, double y) const { return x * y; }
};

template <typename Operation>
Tensor broadcast_operation(
    const Tensor& a,
    const Tensor& b,
    Operation operation)
{
    const auto& ad = a.shape().dimensions();
    const auto& bd = b.shape().dimensions();

    const std::vector<std::size_t> output_shape =
        broadcast_shape(a, b);

    Tensor result = detail::TensorAccess::empty(output_shape);

    const double* A = a.data();
    const double* B = b.data();
    double* C = result.data();

    const std::size_t count = result.size();

    if (count == 0)
        return result;

    if (a.size() == count && b.size() == count)
    {
        #pragma omp simd
        for (std::size_t i = 0; i < count; ++i)
            C[i] = operation(A[i], B[i]);

        return result;
    }

    if (a.size() == 1)
    {
        const double value = A[0];
        #pragma omp simd
        for (std::size_t i = 0; i < count; ++i)
            C[i] = operation(value, B[i]);
        return result;
    }
    if (b.size() == 1)
    {
        const double value = B[0];
        #pragma omp simd
        for (std::size_t i = 0; i < count; ++i)
            C[i] = operation(A[i], value);
        return result;
    }

    if (output_shape.size() == 2 &&
        ad.size() == 2 && bd.size() == 2)
    {
        const std::size_t rows = output_shape[0];
        const std::size_t cols = output_shape[1];

        const bool a_is_full = ad[0] == rows && ad[1] == cols;
        const bool b_is_full = bd[0] == rows && bd[1] == cols;

        const bool a_is_row = ad[0] == 1 && ad[1] == cols;
        const bool b_is_row = bd[0] == 1 && bd[1] == cols;

        const bool a_is_column = ad[0] == rows && ad[1] == 1;
        const bool b_is_column = bd[0] == rows && bd[1] == 1;

        if (a_is_row && b_is_full)
        {
            for (std::size_t i = 0; i < rows; ++i)
            {
                const double* row = B + i * cols;
                double* output = C + i * cols;

                #pragma omp simd
                for (std::size_t j = 0; j < cols; ++j)
                    output[j] = operation(A[j], row[j]);
            }

            return result;
        }

        if (b_is_row && a_is_full)
        {
            for (std::size_t i = 0; i < rows; ++i)
            {
                const double* row = A + i * cols;
                double* output = C + i * cols;

                #pragma omp simd
                for (std::size_t j = 0; j < cols; ++j)
                    output[j] = operation(row[j], B[j]);
            }

            return result;
        }

        if (a_is_column && b_is_full)
        {
            for (std::size_t i = 0; i < rows; ++i)
            {
                const double value = A[i];
                const double* row = B + i * cols;
                double* output = C + i * cols;

                #pragma omp simd
                for (std::size_t j = 0; j < cols; ++j)
                    output[j] = operation(value, row[j]);
            }

            return result;
        }

        if (b_is_column && a_is_full)
        {
            for (std::size_t i = 0; i < rows; ++i)
            {
                const double value = B[i];
                const double* row = A + i * cols;
                double* output = C + i * cols;

                #pragma omp simd
                for (std::size_t j = 0; j < cols; ++j)
                    output[j] = operation(row[j], value);
            }

            return result;
        }
    }

    // Normalize leading/missing axes to zero strides. Cache this once rather
    // than dividing a flat index separately for every operand and element.
    const std::size_t rank = output_shape.size();
    std::vector<std::size_t> a_strides(rank, 0), b_strides(rank, 0);
    for (std::size_t axis = 0; axis < ad.size(); ++axis)
        if (ad[axis] != 1)
            a_strides[rank - ad.size() + axis] = a.shape().strides()[axis];
    for (std::size_t axis = 0; axis < bd.size(); ++axis)
        if (bd[axis] != 1)
            b_strides[rank - bd.size() + axis] = b.shape().strides()[axis];

    std::size_t active_rank = rank;
    while (active_rank && output_shape[active_rank - 1] == 1) --active_rank;
    // Scalar/all-unit outputs were handled by the full/scalar paths above.
    std::size_t inner = output_shape[active_rank - 1];
    const std::size_t a_step = a_strides[active_rank - 1];
    const std::size_t b_step = b_strides[active_rank - 1];
    std::size_t outer_rank = active_rank - 1;
    while (outer_rank)
    {
        const std::size_t axis = outer_rank - 1;
        if (output_shape[axis] != 1 &&
            (a_strides[axis] != a_step * inner ||
             b_strides[axis] != b_step * inner))
            break;
        inner *= output_shape[axis];
        --outer_rank;
    }

    std::vector<std::size_t> coordinates(outer_rank, 0);
    std::size_t a_offset = 0, b_offset = 0;
    for (std::size_t base = 0; base < count; base += inner)
    {
        const double* a_row = A + a_offset;
        const double* b_row = B + b_offset;
        double* out = C + base;
        if (a_step == 0 && b_step == 0)
            std::fill_n(out, inner, operation(a_row[0], b_row[0]));
        else if (a_step == 0)
        {
            const double value = a_row[0];
            #pragma omp simd
            for (std::size_t j = 0; j < inner; ++j)
                out[j] = operation(value, b_row[j]);
        }
        else if (b_step == 0)
        {
            const double value = b_row[0];
            #pragma omp simd
            for (std::size_t j = 0; j < inner; ++j)
                out[j] = operation(a_row[j], value);
        }
        else
        {
            #pragma omp simd
            for (std::size_t j = 0; j < inner; ++j)
                out[j] = operation(a_row[j], b_row[j]);
        }
        // Advance an outer-coordinate odometer once per contiguous run.
        for (std::size_t axis = outer_rank; axis > 0; --axis)
        {
            const std::size_t d = axis - 1;
            ++coordinates[d];
            a_offset += a_strides[d]; b_offset += b_strides[d];
            if (coordinates[d] < output_shape[d]) break;
            coordinates[d] = 0;
            a_offset -= a_strides[d] * output_shape[d];
            b_offset -= b_strides[d] * output_shape[d];
        }
    }

    return result;
}

} // anonymous namespace

// --------------------------------------------------
// Element-wise operations
// --------------------------------------------------

Tensor add(const Tensor& a, const Tensor& b)
{
    validate_same_shape(a, b);

    Tensor result = detail::TensorAccess::empty(a.shape().dimensions());

    const double* __restrict__ A = a.data();
    const double* __restrict__ B = b.data();
    double* __restrict__ C = result.data();

    const std::size_t count = a.size();

    #pragma omp simd
    for (std::size_t i = 0; i < count; ++i)
        C[i] = A[i] + B[i];

    return result;
}

Tensor subtract(const Tensor& a, const Tensor& b)
{
    validate_same_shape(a, b);

    Tensor result = detail::TensorAccess::empty(a.shape().dimensions());

    const double* __restrict__ A = a.data();
    const double* __restrict__ B = b.data();
    double* __restrict__ C = result.data();

    const std::size_t count = a.size();

    #pragma omp simd
    for (std::size_t i = 0; i < count; ++i)
        C[i] = A[i] - B[i];

    return result;
}

Tensor multiply(const Tensor& a, const Tensor& b)
{
    validate_same_shape(a, b);

    Tensor result = detail::TensorAccess::empty(a.shape().dimensions());

    const double* __restrict__ A = a.data();
    const double* __restrict__ B = b.data();
    double* __restrict__ C = result.data();

    const std::size_t count = a.size();

    #pragma omp simd
    for (std::size_t i = 0; i < count; ++i)
        C[i] = A[i] * B[i];

    return result;
}

// --------------------------------------------------
// Parallel SIMD operations
// --------------------------------------------------

Tensor add_parallel(const Tensor& a, const Tensor& b)
{
    // Avoid creating a team for short, bandwidth-bound loops.
    if (a.size() < 65536 || omp_get_max_threads() == 1)
        return add(a, b);

    validate_same_shape(a, b);

    Tensor result = detail::TensorAccess::empty(a.shape().dimensions());

    const double* __restrict__ A = a.data();
    const double* __restrict__ B = b.data();
    double* __restrict__ C = result.data();

    const std::size_t count = a.size();

    #pragma omp parallel for simd schedule(static)
    for (long long i = 0; i < static_cast<long long>(count); ++i)
        C[i] = A[i] + B[i];

    return result;
}

Tensor subtract_parallel(const Tensor& a, const Tensor& b)
{
    // Avoid creating a team for short, bandwidth-bound loops.
    if (a.size() < 65536 || omp_get_max_threads() == 1)
        return subtract(a, b);

    validate_same_shape(a, b);

    Tensor result = detail::TensorAccess::empty(a.shape().dimensions());

    const double* __restrict__ A = a.data();
    const double* __restrict__ B = b.data();
    double* __restrict__ C = result.data();

    const std::size_t count = a.size();

    #pragma omp parallel for simd schedule(static)
    for (long long i = 0; i < static_cast<long long>(count); ++i)
        C[i] = A[i] - B[i];

    return result;
}

Tensor multiply_parallel(const Tensor& a, const Tensor& b)
{
    // Avoid creating a team for short, bandwidth-bound loops.
    if (a.size() < 65536 || omp_get_max_threads() == 1)
        return multiply(a, b);

    validate_same_shape(a, b);

    Tensor result = detail::TensorAccess::empty(a.shape().dimensions());

    const double* __restrict__ A = a.data();
    const double* __restrict__ B = b.data();
    double* __restrict__ C = result.data();

    const std::size_t count = a.size();

    #pragma omp parallel for simd schedule(static)
    for (long long i = 0; i < static_cast<long long>(count); ++i)
        C[i] = A[i] * B[i];

    return result;
}

// --------------------------------------------------
// Broadcasting operations
// --------------------------------------------------

Tensor add_broadcast(const Tensor& a, const Tensor& b)
{
    return broadcast_operation(a, b, AddOp{});
}

Tensor subtract_broadcast(const Tensor& a, const Tensor& b)
{
    return broadcast_operation(a, b, SubtractOp{});
}

Tensor multiply_broadcast(const Tensor& a, const Tensor& b)
{
    return broadcast_operation(a, b, MultiplyOp{});
}

// --------------------------------------------------
// Automatic matrix multiplication dispatch
// --------------------------------------------------

Tensor matmul(const Tensor& a, const Tensor& b)
{
    if (a.rank() != 2 || b.rank() != 2)
        throw std::invalid_argument(
            "Matrix multiplication requires 2D tensors");

    const auto& as = a.shape().dimensions();
    const auto& bs = b.shape().dimensions();

    const std::size_t rows_a = as[0];
    const std::size_t cols_a = as[1];
    const std::size_t rows_b = bs[0];
    const std::size_t cols_b = bs[1];

    if (cols_a != rows_b)
        throw std::invalid_argument(
            "Matrix dimensions are incompatible for multiplication");

    const long double work =
        static_cast<long double>(rows_a) *
        static_cast<long double>(cols_a) *
        static_cast<long double>(cols_b);

    constexpr long double threshold = 64.0L * 64.0L * 64.0L;

    if (work <= threshold)
    {
        Tensor result({rows_a, cols_b}, 0.0);

        const double* A = a.data();
        const double* B = b.data();
        double* C = result.data();

        for (std::size_t i = 0; i < rows_a; ++i)
        {
            for (std::size_t k = 0; k < cols_a; ++k)
            {
                const double value = A[i * cols_a + k];

                for (std::size_t j = 0; j < cols_b; ++j)
                    C[i * cols_b + j] +=
                        value * B[k * cols_b + j];
            }
        }

        return result;
    }

    return matmul_parallel(a, b);
}

// --------------------------------------------------
// OpenMP matrix multiplication
// --------------------------------------------------

Tensor matmul_parallel(const Tensor& a, const Tensor& b)
{
    if (a.rank() != 2 || b.rank() != 2)
        throw std::invalid_argument(
            "Matrix multiplication requires 2D tensors");

    const auto& as = a.shape().dimensions();
    const auto& bs = b.shape().dimensions();

    const std::size_t rows_a = as[0];
    const std::size_t cols_a = as[1];
    const std::size_t rows_b = bs[0];
    const std::size_t cols_b = bs[1];

    if (cols_a != rows_b)
        throw std::invalid_argument("Invalid matrix dimensions");

    Tensor result({rows_a, cols_b}, 0.0);

    const double* A = a.data();
    const double* B = b.data();
    double* C = result.data();

    // Empty products need no pointer arithmetic or parallel region.
    if (rows_a == 0 || cols_a == 0 || cols_b == 0)
        return result;

    // Reuse each B value across four rows. Each worker owns complete output
    // rows, so no synchronization is needed while accumulating over k.
    // Keep one-row tasks when grouping would leave requested workers idle.
    const std::size_t row_group =
        rows_a / 4 >= static_cast<std::size_t>(omp_get_max_threads()) ? 4 : 1;
    const std::size_t groups = rows_a / row_group + (rows_a % row_group != 0);

    #pragma omp parallel for schedule(static)
    for (long long group = 0; group < static_cast<long long>(groups); ++group)
    {
        const std::size_t i = static_cast<std::size_t>(group) * row_group;
        if (row_group == 4 && rows_a - i >= row_group)
        {
            double* c0 = C + i * cols_b;
            double* c1 = c0 + cols_b;
            double* c2 = c1 + cols_b;
            double* c3 = c2 + cols_b;
            for (std::size_t k = 0; k < cols_a; ++k)
            {
                const double a0 = A[i * cols_a + k];
                const double a1 = A[(i + 1) * cols_a + k];
                const double a2 = A[(i + 2) * cols_a + k];
                const double a3 = A[(i + 3) * cols_a + k];
                const double* b_row = B + k * cols_b;
                #pragma omp simd
                for (std::size_t j = 0; j < cols_b; ++j)
                {
                    const double value = b_row[j];
                    c0[j] += a0 * value;
                    c1[j] += a1 * value;
                    c2[j] += a2 * value;
                    c3[j] += a3 * value;
                }
            }
        }
        else
        {
            // Single-row tasks or the final one to three rows.
            const std::size_t end = i + std::min(row_group, rows_a - i);
            for (std::size_t row = i; row < end; ++row)
                for (std::size_t k = 0; k < cols_a; ++k)
                {
                    const double value = A[row * cols_a + k];
                    #pragma omp simd
                    for (std::size_t j = 0; j < cols_b; ++j)
                        C[row * cols_b + j] += value * B[k * cols_b + j];
                }
        }
    }

    return result;
}

// --------------------------------------------------
// Cache-blocked matrix multiplication
// --------------------------------------------------

Tensor matmul_blocked(
    const Tensor& a,
    const Tensor& b,
    std::size_t block_size)
{
    if (a.rank() != 2 || b.rank() != 2)
        throw std::invalid_argument(
            "Matrix multiplication requires 2D tensors");

    const auto& as = a.shape().dimensions();
    const auto& bs = b.shape().dimensions();

    const std::size_t rows_a = as[0];
    const std::size_t cols_a = as[1];
    const std::size_t rows_b = bs[0];
    const std::size_t cols_b = bs[1];

    if (cols_a != rows_b)
        throw std::invalid_argument(
            "Matrix dimensions are incompatible for multiplication");

    if (block_size == 0)
        throw std::invalid_argument("Block size cannot be zero");

    Tensor result({rows_a, cols_b}, 0.0);

    const double* A = a.data();
    const double* B = b.data();
    double* C = result.data();

    for (std::size_t ii = 0; ii < rows_a; ii += block_size)
    {
        for (std::size_t jj = 0; jj < cols_b; jj += block_size)
        {
            for (std::size_t kk = 0; kk < cols_a; kk += block_size)
            {
                const std::size_t i_end =
                    ii + std::min(block_size, rows_a - ii);
                const std::size_t j_end =
                    jj + std::min(block_size, cols_b - jj);
                const std::size_t k_end =
                    kk + std::min(block_size, cols_a - kk);

                for (std::size_t i = ii; i < i_end; ++i)
                {
                    for (std::size_t k = kk; k < k_end; ++k)
                    {
                        const double value = A[i * cols_a + k];

                        for (std::size_t j = jj; j < j_end; ++j)
                            C[i * cols_b + j] +=
                                value * B[k * cols_b + j];
                    }
                }
            }
        }
    }

    return result;
}

// --------------------------------------------------
// Parallel cache-blocked matrix multiplication
// --------------------------------------------------

Tensor matmul_parallel_blocked(
    const Tensor& a,
    const Tensor& b,
    std::size_t block_size)
{
    if (a.rank() != 2 || b.rank() != 2)
        throw std::invalid_argument(
            "Matrix multiplication requires 2D tensors");

    const auto& as = a.shape().dimensions();
    const auto& bs = b.shape().dimensions();

    const std::size_t M = as[0];
    const std::size_t K = as[1];
    const std::size_t N = bs[1];

    if (K != bs[0])
        throw std::invalid_argument(
            "Matrix dimensions are incompatible for multiplication");

    if (block_size == 0)
        throw std::invalid_argument("Block size cannot be zero");

    Tensor result({M, N}, 0.0);

    const double* A = a.data();
    const double* B = b.data();
    double* C = result.data();

    const std::size_t row_blocks =
        M / block_size + (M % block_size != 0);

    const std::size_t col_blocks =
        N / block_size + (N % block_size != 0);

    // Parallelize independent output tiles across both dimensions.
    #pragma omp parallel for collapse(2) schedule(static)
    for (long long bi = 0;
         bi < static_cast<long long>(row_blocks); ++bi)
    {
        for (long long bj = 0;
             bj < static_cast<long long>(col_blocks); ++bj)
        {
            const std::size_t ii =
                static_cast<std::size_t>(bi) * block_size;

            const std::size_t jj =
                static_cast<std::size_t>(bj) * block_size;

            const std::size_t i_end =
                ii + std::min(block_size, M - ii);

            const std::size_t j_end =
                jj + std::min(block_size, N - jj);

            for (std::size_t kk = 0; kk < K; kk += block_size)
            {
                const std::size_t k_end =
                    kk + std::min(block_size, K - kk);

                for (std::size_t i = ii; i < i_end; ++i)
                {
                    for (std::size_t k = kk; k < k_end; ++k)
                    {
                        const double value = A[i * K + k];

                        #pragma omp simd
                        for (std::size_t j = jj; j < j_end; ++j)
                        {
                            C[i * N + j] +=
                                value * B[k * N + j];
                        }
                    }
                }
            }
        }
    }

    return result;
}

// --------------------------------------------------
// Cache-blocked transpose
// --------------------------------------------------

Tensor transpose(const Tensor& a)
{
    if (a.rank() != 2)
        throw std::invalid_argument("Transpose requires a 2D tensor");

    const auto& dims = a.shape().dimensions();

    const std::size_t rows = dims[0];
    const std::size_t cols = dims[1];

    Tensor result = detail::TensorAccess::empty({cols, rows});

    const double* input = a.data();
    double* output = result.data();

    constexpr std::size_t BLOCK_SIZE = 16;

    for (std::size_t ii = 0; ii < rows; ii += BLOCK_SIZE)
    {
        for (std::size_t jj = 0; jj < cols; jj += BLOCK_SIZE)
        {
            const std::size_t i_end =
                ii + std::min(BLOCK_SIZE, rows - ii);
            const std::size_t j_end =
                jj + std::min(BLOCK_SIZE, cols - jj);

            // Write adjacent output elements; tiles bound the strided reads.
            for (std::size_t j = jj; j < j_end; ++j)
            {
                for (std::size_t i = ii; i < i_end; ++i)
                    output[j * rows + i] = input[i * cols + j];
            }
        }
    }

    return result;
}

// --------------------------------------------------
// Arithmetic operators
// --------------------------------------------------

Tensor operator+(const Tensor& a, const Tensor& b)
{
    return add(a, b);
}

Tensor operator-(const Tensor& a, const Tensor& b)
{
    return subtract(a, b);
}

Tensor operator*(const Tensor& a, const Tensor& b)
{
    return multiply(a, b);
}

// --------------------------------------------------
// Reductions
// --------------------------------------------------

double sum(const Tensor& a)
{
    double result = 0.0;
    const double* input = a.data();
    const std::size_t count = a.size();

    if (count >= 262144 && omp_get_max_threads() > 1)
    {
        #pragma omp parallel for simd schedule(static) reduction(+:result)
        for (long long i = 0; i < static_cast<long long>(count); ++i)
            result += input[i];
    }
    else
    {
        #pragma omp simd reduction(+:result)
        for (std::size_t i = 0; i < count; ++i)
            result += input[i];
    }
    return result;
}

double mean(const Tensor& a)
{
    if (a.size() == 0)
        throw std::invalid_argument(
            "Mean of an empty tensor is undefined");

    return sum(a) / static_cast<double>(a.size());
}

namespace {

template <typename Compare>
double extreme(const Tensor& a, Compare better)
{
    const std::size_t count = a.size();
    const double* input = a.data();
    double result = input[0];
    if (count < 262144 || omp_get_max_threads() == 1)
    {
        for (std::size_t i = 1; i < count; ++i)
            if (better(input[i], result)) result = input[i];
        return result;
    }

    // Scan ordered contiguous chunks; combine in chunk order. Unlike an
    // unordered SIMD min/max reduction, this retains first-NaN and signed-zero
    // behavior of the original strict-comparison loop.
    std::vector<double> partial(omp_get_max_threads(), input[0]);
    #pragma omp parallel
    {
        const std::size_t id = static_cast<std::size_t>(omp_get_thread_num());
        const std::size_t team = static_cast<std::size_t>(omp_get_num_threads());
        const std::size_t chunk = count / team, extra = count % team;
        const std::size_t begin = chunk * id + std::min(id, extra);
        const std::size_t end = begin + chunk + (id < extra);
        double local = input[0];
        for (std::size_t i = begin; i < end; ++i)
            if (better(input[i], local)) local = input[i];
        partial[id] = local;
    }
    for (const double value : partial)
        if (better(value, result)) result = value;
    return result;
}

} // namespace

double min(const Tensor& a)
{
    if (a.size() == 0)
        throw std::invalid_argument("Minimum of an empty tensor is undefined");
    return extreme(a, [](double x, double y) { return x < y; });
}

double max(const Tensor& a)
{
    if (a.size() == 0)
        throw std::invalid_argument("Maximum of an empty tensor is undefined");
    return extreme(a, [](double x, double y) { return x > y; });
}

} // namespace minitensor
