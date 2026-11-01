
#include "minitensor/Ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

using namespace minitensor;
using Clock = std::chrono::steady_clock;

volatile double benchmark_sink = 0.0;

Tensor normal_matmul(const Tensor& a, const Tensor& b)
{
    if (a.rank() != 2 || b.rank() != 2)
        throw std::invalid_argument("Matrix multiplication requires 2D tensors");

    const auto& as = a.shape().dimensions();
    const auto& bs = b.shape().dimensions();

    const std::size_t rows = as[0];
    const std::size_t inner = as[1];
    const std::size_t cols = bs[1];

    if (inner != bs[0])
        throw std::invalid_argument("Incompatible matrix dimensions");

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
                C[i * cols + j] += value * B[k * cols + j];
        }
    }

    return result;
}

Tensor create_matrix(std::size_t rows, std::size_t cols, double seed)
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

bool compare_results(
    const Tensor& a,
    const Tensor& b,
    double tolerance = 1e-8)
{
    if (a.shape().dimensions() != b.shape().dimensions())
        return false;

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (std::abs(a.flat_at(i) - b.flat_at(i)) > tolerance)
            return false;
    }

    return true;
}

template <typename Function>
double measure(Function function, int warmups = 2, int measurements = 7)
{
    for (int i = 0; i < warmups; ++i)
    {
        Tensor result = function();

        if (result.size() > 0)
            benchmark_sink += result.flat_at(result.size() - 1);
    }

    std::vector<double> times;
    times.reserve(measurements);

    for (int i = 0; i < measurements; ++i)
    {
        const auto start = Clock::now();

        Tensor result = function();

        const auto end = Clock::now();

        if (result.size() > 0)
            benchmark_sink += result.flat_at(result.size() - 1);

        times.push_back(
            std::chrono::duration<double, std::milli>(end - start).count()
        );
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

void print_result(const std::string& name, double time)
{
    std::cout << std::left << std::setw(28) << name
              << std::right << std::fixed << std::setprecision(3)
              << time << " ms\n";
}

void print_speedup(
    const std::string& name,
    double baseline,
    double time)
{
    std::cout << std::left << std::setw(28) << name
              << std::right << std::fixed << std::setprecision(3)
              << baseline / time << "x\n";
}

int main()
{
    const std::vector<std::size_t> sizes = {
        64, 128, 256, 512, 1024
    };

    const std::vector<std::size_t> block_sizes = {
        32, 64, 128, 256
    };

    const std::vector<int> thread_counts = {
        1, 2, 4, 8, 16
    };

    constexpr int warmups = 2;
    constexpr int measurements = 7;

    omp_set_dynamic(0);

    std::cout << "\nMINITENSOR MATRIX MULTIPLICATION BENCHMARK\n";
    std::cout << "============================================\n";
    std::cout << "Warmups: " << warmups
              << " | Measurements: " << measurements << '\n';

    for (const std::size_t n : sizes)
    {
        std::cout << "\n============================================\n";
        std::cout << "MATRIX SIZE: " << n << " x " << n << '\n';
        std::cout << "============================================\n";

        Tensor a = create_matrix(n, n, 1.0);
        Tensor b = create_matrix(n, n, 2.0);

        Tensor reference = normal_matmul(a, b);

        const double normal_time = measure(
            [&]() { return normal_matmul(a, b); },
            warmups, measurements
        );

        print_result("Normal sequential", normal_time);

        double best_blocked_time = 1e100;
        std::size_t best_block_size = 0;

        std::cout << "\nSequential blocked results:\n";

        for (const std::size_t block : block_sizes)
        {
            Tensor result = matmul_blocked(a, b, block);

            if (!compare_results(reference, result))
                throw std::runtime_error("Blocked correctness check failed");

            const double time = measure(
                [&]() { return matmul_blocked(a, b, block); },
                warmups, measurements
            );

            print_result("Block " + std::to_string(block), time);

            if (time < best_blocked_time)
            {
                best_blocked_time = time;
                best_block_size = block;
            }
        }

        std::cout << "Best sequential block size: "
                  << best_block_size << '\n';

        for (const int threads : thread_counts)
        {
            omp_set_num_threads(threads);

            std::cout << "\n---------- THREADS: "
                      << threads << " ----------\n";

            Tensor parallel_result = matmul_parallel(a, b);
            Tensor hybrid_result = matmul(a, b);

            if (!compare_results(reference, parallel_result) ||
                !compare_results(reference, hybrid_result))
            {
                throw std::runtime_error(
                    "Parallel or hybrid correctness check failed"
                );
            }

            const double parallel_time = measure(
                [&]() { return matmul_parallel(a, b); },
                warmups, measurements
            );

            const double hybrid_time = measure(
                [&]() { return matmul(a, b); },
                warmups, measurements
            );

            print_result("OpenMP", parallel_time);
            print_result("Hybrid", hybrid_time);

            double best_parallel_blocked_time = 1e100;
            std::size_t best_parallel_block_size = 0;

            for (const std::size_t block : block_sizes)
            {
                Tensor result = matmul_parallel_blocked(a, b, block);

                if (!compare_results(reference, result))
                {
                    throw std::runtime_error(
                        "Parallel-blocked correctness check failed"
                    );
                }

                const double time = measure(
                    [&]() {
                        return matmul_parallel_blocked(a, b, block);
                    },
                    warmups, measurements
                );

                if (time < best_parallel_blocked_time)
                {
                    best_parallel_blocked_time = time;
                    best_parallel_block_size = block;
                }
            }

            print_result(
                "Best parallel-blocked",
                best_parallel_blocked_time
            );

            std::cout << "\nSpeedup vs sequential:\n";

            print_speedup("OpenMP", normal_time, parallel_time);
            print_speedup("Hybrid", normal_time, hybrid_time);
            print_speedup(
                "Best sequential blocked",
                normal_time,
                best_blocked_time
            );
            print_speedup(
                "Best parallel-blocked",
                normal_time,
                best_parallel_blocked_time
            );

            std::cout << "Best parallel block size: "
                      << best_parallel_block_size << '\n';
        }

        for (std::size_t i = 0; i < reference.size(); ++i)
            benchmark_sink += reference.flat_at(i);
    }

    std::cout << "\n============================================\n";
    std::cout << "Benchmark completed successfully.\n";
    std::cout << "Prevent-optimization checksum: "
              << std::fixed << std::setprecision(3)
              << benchmark_sink << '\n';

    return 0;
}
