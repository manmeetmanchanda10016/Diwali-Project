
#include "minitensor/Ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace minitensor;
using Clock = std::chrono::steady_clock;

volatile double benchmark_sink = 0.0;

struct AddOp {
    double operator()(double x, double y) const {
        return x + y;
    }
};

struct SubtractOp {
    double operator()(double x, double y) const {
        return x - y;
    }
};

struct MultiplyOp {
    double operator()(double x, double y) const {
        return x * y;
    }
};

// Direct SIMD reference for row broadcasting.
template <typename Operation>
Tensor direct_row_reference(
    const Tensor& a,
    const Tensor& b,
    Operation op)
{
    const std::size_t rows = a.shape().dimensions()[0];
    const std::size_t cols = a.shape().dimensions()[1];

    Tensor result({rows, cols});

    const double* A = a.data();
    const double* B = b.data();
    double* C = result.data();

    for (std::size_t i = 0; i < rows; ++i) {
        const double* row = A + i * cols;
        double* output = C + i * cols;

        #pragma omp simd
        for (std::size_t j = 0; j < cols; ++j) {
            output[j] = op(row[j], B[j]);
        }
    }

    return result;
}

// Direct SIMD reference for column broadcasting.
template <typename Operation>
Tensor direct_column_reference(
    const Tensor& a,
    const Tensor& b,
    Operation op)
{
    const std::size_t rows = a.shape().dimensions()[0];
    const std::size_t cols = a.shape().dimensions()[1];

    Tensor result({rows, cols});

    const double* A = a.data();
    const double* B = b.data();
    double* C = result.data();

    for (std::size_t i = 0; i < rows; ++i) {
        const double value = B[i];
        const double* row = A + i * cols;
        double* output = C + i * cols;

        #pragma omp simd
        for (std::size_t j = 0; j < cols; ++j) {
            output[j] = op(row[j], value);
        }
    }

    return result;
}

// Check numerical correctness.
bool correct(const Tensor& a, const Tensor& b)
{
    if (a.shape().dimensions() != b.shape().dimensions()) {
        return false;
    }

    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a.data()[i] - b.data()[i]) > 1e-9) {
            return false;
        }
    }

    return true;
}

// Prevent benchmark results from being optimized away.
void consume(const Tensor& result)
{
    if (result.size() > 0) {
        benchmark_sink += result.data()[0];
        benchmark_sink += result.data()[result.size() - 1];
    }
}

// Calculate median timing.
double median(std::vector<double>& values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

// Benchmark one implementation against a reference.
template <typename Current, typename Reference>
void run_case(
    const std::string& name,
    const Tensor& a,
    const Tensor& b,
    Current current,
    Reference reference)
{
    Tensor expected = current(a, b);
    Tensor actual = reference(a, b);

    if (!correct(expected, actual)) {
        std::cout << "\n" << name << ": CORRECTNESS FAILED\n";
        return;
    }

    constexpr int warmups = 2;
    constexpr int repetitions = 9;

    std::vector<double> current_times;
    std::vector<double> reference_times;

    for (int i = 0; i < warmups; ++i) {
        Tensor x = current(a, b);
        Tensor y = reference(a, b);

        consume(x);
        consume(y);
    }

    for (int i = 0; i < repetitions; ++i) {

        auto measure = [](auto&& function) {
            const auto start = Clock::now();

            Tensor result = function();

            const auto end = Clock::now();

            consume(result);

            return std::chrono::duration<double, std::milli>(
                end - start
            ).count();
        };

        // Alternate order to reduce measurement-order bias.
        if (i % 2 == 0) {
            current_times.push_back(
                measure([&]() {
                    return current(a, b);
                })
            );

            reference_times.push_back(
                measure([&]() {
                    return reference(a, b);
                })
            );
        }
        else {
            reference_times.push_back(
                measure([&]() {
                    return reference(a, b);
                })
            );

            current_times.push_back(
                measure([&]() {
                    return current(a, b);
                })
            );
        }
    }

    const double current_ms = median(current_times);
    const double reference_ms = median(reference_times);

    std::cout << "\n" << name << "\n";

    std::cout << "Library:          "
              << current_ms << " ms\n";

    std::cout << "Direct reference: "
              << reference_ms << " ms\n";

    std::cout << "Speedup:          "
              << current_ms / reference_ms << "x\n";
}

// Run same-shape, row and column broadcasting tests.
template <typename Operation>
void run_operation(
    const std::string& operation_name,
    const Tensor& full,
    const Tensor& row,
    const Tensor& column,
    Operation op,
    const std::function<Tensor(
        const Tensor&, const Tensor&)>& library)
{
    std::cout << "\n========== "
              << operation_name
              << " ==========\n";

    // Same shape.
    run_case(
        "Same shape",
        full,
        full,
        library,
        [&](const Tensor& a, const Tensor& b) {
            Tensor result(a.shape().dimensions());

            const double* A = a.data();
            const double* B = b.data();
            double* C = result.data();

            #pragma omp simd
            for (std::size_t i = 0; i < a.size(); ++i) {
                C[i] = op(A[i], B[i]);
            }

            return result;
        }
    );

    // Row broadcasting.
    run_case(
        "Row broadcasting",
        full,
        row,
        library,
        [&](const Tensor& a, const Tensor& b) {
            return direct_row_reference(a, b, op);
        }
    );

    // Column broadcasting.
    run_case(
        "Column broadcasting",
        full,
        column,
        library,
        [&](const Tensor& a, const Tensor& b) {
            return direct_column_reference(a, b, op);
        }
    );
}

int main()
{
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "MINITENSOR BROADCASTING BENCHMARK V3\n";
    std::cout << "====================================\n";
    std::cout << "Same inputs, allocation and timing methodology\n";

    constexpr std::size_t rows = 512;
    constexpr std::size_t cols = 512;

    Tensor full({rows, cols});
    Tensor row({1, cols});
    Tensor column({rows, 1});

    // Initialize full matrix.
    for (std::size_t i = 0; i < full.size(); ++i) {
        full.data()[i] =
            static_cast<double>(i % 100) * 0.1;
    }

    // Initialize row tensor.
    for (std::size_t i = 0; i < cols; ++i) {
        row.data()[i] =
            static_cast<double>(i % 50) * 0.2;
    }

    // Initialize column tensor.
    for (std::size_t i = 0; i < rows; ++i) {
        column.data()[i] =
            static_cast<double>(i % 30) * 0.3;
    }

    // Addition.
    run_operation(
        "ADD",
        full,
        row,
        column,
        AddOp{},
        [](const Tensor& a, const Tensor& b) {
            return add_broadcast(a, b);
        }
    );

    // Subtraction.
    run_operation(
        "SUBTRACT",
        full,
        row,
        column,
        SubtractOp{},
        [](const Tensor& a, const Tensor& b) {
            return subtract_broadcast(a, b);
        }
    );

    // Multiplication.
    run_operation(
        "MULTIPLY",
        full,
        row,
        column,
        MultiplyOp{},
        [](const Tensor& a, const Tensor& b) {
            return multiply_broadcast(a, b);
        }
    );

    std::cout << "\n====================================\n";
    std::cout << "Benchmark completed.\n";
    std::cout << "Checksum: " << benchmark_sink << "\n";

    return 0;
}
