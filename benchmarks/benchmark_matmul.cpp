
#include "minitensor/Ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

using namespace minitensor;
using Clock = std::chrono::steady_clock;

volatile double benchmark_sink = 0.0;

constexpr int WARMUPS = 3;
constexpr int MEASUREMENTS = 10;

struct BenchmarkResult
{
    double average_ms;
    double median_ms;
    double gflops;
};

Tensor normal_matmul(const Tensor& a, const Tensor& b)
{
    if (a.rank() != 2 || b.rank() != 2)
        throw std::invalid_argument(
            "Matrix multiplication requires 2D tensors"
        );

    const auto& as = a.shape().dimensions();
    const auto& bs = b.shape().dimensions();

    const std::size_t rows = as[0];
    const std::size_t inner = as[1];
    const std::size_t cols = bs[1];

    if (inner != bs[0])
        throw std::invalid_argument(
            "Incompatible matrix dimensions"
        );

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
                std::fmod(
                    seed + i * 0.17 + j * 0.31,
                    10.0
                );
        }
    }

    return result;
}

bool compare_results(
    const Tensor& a,
    const Tensor& b,
    double tolerance = 1e-8)
{
    if (a.shape().dimensions() !=
        b.shape().dimensions())
    {
        return false;
    }

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (std::abs(
                a.flat_at(i) - b.flat_at(i)
            ) > tolerance)
        {
            return false;
        }
    }

    return true;
}

template <typename Function>
BenchmarkResult measure(
    Function function,
    std::size_t matrix_size)
{
    for (int i = 0; i < WARMUPS; ++i)
    {
        double checksum = 0.0;

        {
            Tensor result = function();

            if (result.size() > 0)
            {
                checksum =
                    result.flat_at(result.size() - 1);
            }
        }

        benchmark_sink =
            benchmark_sink + checksum;
    }

    std::vector<double> times;
    times.reserve(MEASUREMENTS);

    double total_time = 0.0;

    for (int i = 0; i < MEASUREMENTS; ++i)
    {
        double checksum = 0.0;

        const auto start = Clock::now();

        {
            Tensor result = function();

            if (result.size() > 0)
            {
                checksum =
                    result.flat_at(result.size() - 1);
            }
        }

        const auto end = Clock::now();

        const double elapsed =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        times.push_back(elapsed);
        total_time += elapsed;

        benchmark_sink =
            benchmark_sink + checksum;
    }

    std::sort(times.begin(), times.end());

    const double average =
        total_time / MEASUREMENTS;

    const double median =
        (times[MEASUREMENTS / 2 - 1] +
         times[MEASUREMENTS / 2]) / 2.0;

    const double operations =
        2.0 * matrix_size * matrix_size * matrix_size;

    const double gflops =
        operations / (median * 1e6);

    return {
        average,
        median,
        gflops
    };
}

void print_result(
    const std::string& name,
    const BenchmarkResult& result)
{
    std::cout << std::left << std::setw(28)
              << name
              << std::right
              << std::fixed
              << std::setprecision(4)
              << "Avg: " << result.average_ms
              << " ms | Median: " << result.median_ms
              << " ms | GFLOPS: " << result.gflops
              << '\n';
}

void save_csv(
    std::ofstream& csv,
    std::size_t matrix_size,
    const std::string& algorithm,
    int threads,
    std::size_t block_size,
    const BenchmarkResult& result,
    bool correct)
{
    csv << matrix_size << ','
        << algorithm << ','
        << threads << ','
        << block_size << ','
        << WARMUPS << ','
        << MEASUREMENTS << ','
        << std::fixed << std::setprecision(6)
        << result.average_ms << ','
        << result.median_ms << ','
        << result.gflops << ','
        << (correct ? "PASS" : "FAIL")
        << '\n';
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
        1, 2, 4, 8
    };

    omp_set_dynamic(0);

    std::ofstream csv("matmul_benchmark.csv");

    if (!csv.is_open())
    {
        std::cerr << "Could not create CSV file.\n";
        return 1;
    }

    csv << "MatrixSize,Algorithm,Threads,BlockSize,"
        << "Warmups,Measurements,Average_ms,"
        << "Median_ms,GFLOPS,Correctness\n";

    std::cout
        << "\nMINITENSOR MATRIX MULTIPLICATION BENCHMARK\n";

    std::cout
        << "============================================\n";

    std::cout << "Warmups: " << WARMUPS
              << " | Measurements: "
              << MEASUREMENTS << '\n';

    for (const std::size_t n : sizes)
    {
        std::cout
            << "\n============================================\n";

        std::cout << "MATRIX SIZE: "
                  << n << " x " << n << '\n';

        std::cout
            << "============================================\n";

        Tensor a = create_matrix(n, n, 1.0);
        Tensor b = create_matrix(n, n, 2.0);

        Tensor reference = normal_matmul(a, b);

        // Normal sequential baseline

        omp_set_num_threads(1);

        BenchmarkResult normal_result =
            measure(
                [&]() {
                    return normal_matmul(a, b);
                },
                n
            );

        print_result(
            "Normal sequential",
            normal_result
        );

        save_csv(
            csv, n, "NormalSequential",
            1, 0, normal_result, true
        );

        // Sequential blocked multiplication

        std::cout << "\nSequential blocked:\n";

        for (const std::size_t block : block_sizes)
        {
            Tensor result =
                matmul_blocked(a, b, block);

            const bool correct =
                compare_results(reference, result);

            if (!correct)
                throw std::runtime_error(
                    "Sequential blocked correctness failed"
                );

            BenchmarkResult benchmark =
                measure(
                    [&]() {
                        return matmul_blocked(
                            a, b, block
                        );
                    },
                    n
                );

            print_result(
                "Block " + std::to_string(block),
                benchmark
            );

            save_csv(
                csv, n, "SequentialBlocked",
                1, block, benchmark, correct
            );
        }

        // Parallel and hybrid multiplication

        for (const int threads : thread_counts)
        {
            omp_set_num_threads(threads);

            std::cout
                << "\n---------- THREADS: "
                << threads
                << " ----------\n";

            // OpenMP multiplication

            Tensor parallel_result =
                matmul_parallel(a, b);

            bool parallel_correct =
                compare_results(
                    reference,
                    parallel_result
                );

            if (!parallel_correct)
                throw std::runtime_error(
                    "OpenMP correctness failed"
                );

            BenchmarkResult parallel_benchmark =
                measure(
                    [&]() {
                        return matmul_parallel(a, b);
                    },
                    n
                );

            print_result(
                "OpenMP",
                parallel_benchmark
            );

            save_csv(
                csv, n, "OpenMP",
                threads, 0,
                parallel_benchmark,
                parallel_correct
            );

            // Hybrid multiplication

            Tensor hybrid_result =
                matmul(a, b);

            bool hybrid_correct =
                compare_results(
                    reference,
                    hybrid_result
                );

            if (!hybrid_correct)
                throw std::runtime_error(
                    "Hybrid correctness failed"
                );

            BenchmarkResult hybrid_benchmark =
                measure(
                    [&]() {
                        return matmul(a, b);
                    },
                    n
                );

            print_result(
                "Hybrid",
                hybrid_benchmark
            );

            save_csv(
                csv, n, "Hybrid",
                threads, 0,
                hybrid_benchmark,
                hybrid_correct
            );

            // Parallel blocked multiplication

            for (const std::size_t block : block_sizes)
            {
                Tensor blocked_result =
                    matmul_parallel_blocked(
                        a, b, block
                    );

                bool blocked_correct =
                    compare_results(
                        reference,
                        blocked_result
                    );

                if (!blocked_correct)
                    throw std::runtime_error(
                        "Parallel blocked correctness failed"
                    );

                BenchmarkResult blocked_benchmark =
                    measure(
                        [&]() {
                            return matmul_parallel_blocked(
                                a, b, block
                            );
                        },
                        n
                    );

                print_result(
                    "Parallel blocked " +
                    std::to_string(block),
                    blocked_benchmark
                );

                save_csv(
                    csv, n, "ParallelBlocked",
                    threads, block,
                    blocked_benchmark,
                    blocked_correct
                );
            }
        }

        // Prevent optimization of reference calculations.

        double checksum = 0.0;

        for (std::size_t i = 0;
             i < reference.size(); ++i)
        {
            checksum += reference.flat_at(i);
        }

        benchmark_sink =
            benchmark_sink + checksum;
    }

    csv.close();

    std::cout
        << "\n============================================\n";

    std::cout
        << "Benchmark completed successfully.\n";

    std::cout
        << "CSV saved: matmul_benchmark.csv\n";

    std::cout
        << "Checksum: "
        << std::fixed
        << std::setprecision(3)
        << benchmark_sink
        << '\n';

    return 0;
}
