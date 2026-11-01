
#include "minitensor/Ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace minitensor;
using Clock = std::chrono::steady_clock;

// Original sequential i-k-j matrix multiplication
Tensor normal_matmul(const Tensor& a, const Tensor& b)
{
    if (a.rank() != 2 || b.rank() != 2)
    {
        throw std::invalid_argument(
            "Matrix multiplication requires 2D tensors"
        );
    }

    const auto& a_shape = a.shape().dimensions();
    const auto& b_shape = b.shape().dimensions();

    const std::size_t rows = a_shape[0];
    const std::size_t inner = a_shape[1];
    const std::size_t cols = b_shape[1];

    if (inner != b_shape[0])
    {
        throw std::invalid_argument(
            "Incompatible matrix dimensions"
        );
    }

    Tensor result({rows, cols}, 0.0);

    const double* A = a.data();
    const double* B = b.data();
    double* C = result.data();

    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t k = 0; k < inner; ++k)
        {
            const double value = A[i * inner + k];

            for (std::size_t j = 0; j < cols; ++j)
            {
                C[i * cols + j] +=
                    value * B[k * cols + j];
            }
        }
    }

    return result;
}

// Generate nonuniform matrices
Tensor create_matrix(
    std::size_t rows,
    std::size_t cols,
    double seed)
{
    Tensor result({rows, cols});

    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            result.flat_at(i * cols + j) =
                std::fmod(seed + i * 0.17 + j * 0.31, 10.0);
        }
    }

    return result;
}

// Compare two tensor results
bool compare_results(
    const Tensor& a,
    const Tensor& b,
    double tolerance = 1e-8)
{
    if (a.shape().dimensions() != b.shape().dimensions())
    {
        return false;
    }

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (std::abs(a.flat_at(i) - b.flat_at(i)) > tolerance)
        {
            return false;
        }
    }

    return true;
}

// Measure median execution time
template <typename Function>
double measure(
    Function function,
    int warmups = 3,
    int measurements = 9)
{
    for (int i = 0; i < warmups; ++i)
    {
        Tensor result = function();

        volatile double check =
            result.size() > 0 ? result.flat_at(0) : 0.0;

        (void)check;
    }

    std::vector<double> times;
    times.reserve(measurements);

    for (int i = 0; i < measurements; ++i)
    {
        const auto start = Clock::now();

        Tensor result = function();

        const auto end = Clock::now();

        volatile double check =
            result.size() > 0 ? result.flat_at(0) : 0.0;

        (void)check;

        const double elapsed =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        times.push_back(elapsed);
    }

    std::sort(times.begin(), times.end());

    return times[times.size() / 2];
}

int main()
{
    const std::vector<std::size_t> sizes = {
        64, 128, 256, 512, 1024
    };

    const std::vector<std::size_t> block_sizes = {
        16, 32, 64, 128
    };

    constexpr int warmups = 3;
    constexpr int measurements = 9;

    double global_checksum = 0.0;

    std::cout << "\nMATRIX MULTIPLICATION BENCHMARK\n";
    std::cout << "============================================\n";

    std::cout << "Warmups: " << warmups
              << " | Measurements: " << measurements
              << " | Statistic: Median\n";

    for (const std::size_t n : sizes)
    {
        std::cout << "\n--------------------------------------------\n";
        std::cout << "Matrix size: " << n << " x " << n << "\n";
        std::cout << "--------------------------------------------\n";

        Tensor a = create_matrix(n, n, 1.0);
        Tensor b = create_matrix(n, n, 2.0);

        // Sequential reference
        Tensor reference = normal_matmul(a, b);

        // Correctness checks
        if (!compare_results(reference, matmul(a, b)) ||
            !compare_results(reference, matmul_parallel(a, b)))
        {
            throw std::runtime_error(
                "Normal, OpenMP or hybrid correctness check failed"
            );
        }

        const double normal_time = measure(
            [&]() { return normal_matmul(a, b); },
            warmups,
            measurements
        );

        const double parallel_time = measure(
            [&]() { return matmul_parallel(a, b); },
            warmups,
            measurements
        );

        const double hybrid_time = measure(
            [&]() { return matmul(a, b); },
            warmups,
            measurements
        );

        double best_blocked_time = 0.0;
        double best_parallel_blocked_time = 0.0;

        std::size_t best_blocked_size = 0;
        std::size_t best_parallel_blocked_size = 0;

        std::cout << "\nBlocked experiments:\n";

        for (const std::size_t block_size : block_sizes)
        {
            const double blocked_time = measure(
                [&]()
                {
                    return matmul_blocked(a, b, block_size);
                },
                warmups,
                measurements
            );

            const double parallel_blocked_time = measure(
                [&]()
                {
                    return matmul_parallel_blocked(
                        a, b, block_size
                    );
                },
                warmups,
                measurements
            );

            Tensor blocked_result =
                matmul_blocked(a, b, block_size);

            Tensor parallel_blocked_result =
                matmul_parallel_blocked(a, b, block_size);

            if (!compare_results(reference, blocked_result) ||
                !compare_results(reference, parallel_blocked_result))
            {
                throw std::runtime_error(
                    "Blocked multiplication correctness check failed"
                );
            }

            std::cout << "Block " << std::setw(3) << block_size
                      << " | Blocked: "
                      << std::fixed << std::setprecision(3)
                      << blocked_time << " ms"
                      << " | Parallel-blocked: "
                      << parallel_blocked_time << " ms\n";

            if (best_blocked_size == 0 ||
                blocked_time < best_blocked_time)
            {
                best_blocked_time = blocked_time;
                best_blocked_size = block_size;
            }

            if (best_parallel_blocked_size == 0 ||
                parallel_blocked_time < best_parallel_blocked_time)
            {
                best_parallel_blocked_time = parallel_blocked_time;
                best_parallel_blocked_size = block_size;
            }
        }

        std::cout << "\nPerformance Results:\n";

        auto print_result = [](
            const std::string& name,
            double time)
        {
            std::cout << std::left << std::setw(25)
                      << name
                      << std::right << std::fixed
                      << std::setprecision(3)
                      << time << " ms\n";
        };

        print_result("Normal sequential", normal_time);
        print_result("OpenMP", parallel_time);
        print_result("Hybrid", hybrid_time);
        print_result("Best blocked", best_blocked_time);
        print_result(
            "Best parallel-blocked",
            best_parallel_blocked_time
        );

        std::cout << "\nBest block sizes:\n";
        std::cout << "Blocked: "
                  << best_blocked_size << '\n';

        std::cout << "Parallel-blocked: "
                  << best_parallel_blocked_size << '\n';

        std::cout << "\nSpeedup compared with normal:\n";

        std::cout << "OpenMP: "
                  << normal_time / parallel_time << "x\n";

        std::cout << "Hybrid: "
                  << normal_time / hybrid_time << "x\n";

        std::cout << "Best blocked: "
                  << normal_time / best_blocked_time << "x\n";

        std::cout << "Best parallel-blocked: "
                  << normal_time / best_parallel_blocked_time << "x\n";

        for (std::size_t i = 0; i < reference.size(); ++i)
        {
            global_checksum += reference.flat_at(i);
        }
    }

    std::cout << "\n============================================\n";

    std::cout << "Global checksum: "
              << std::fixed << std::setprecision(3)
              << global_checksum << '\n';

    std::cout << "All correctness checks passed.\n";
    std::cout << "Benchmark completed successfully.\n";

    return 0;
}
