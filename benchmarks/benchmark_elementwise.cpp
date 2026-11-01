
#include "minitensor/Ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

using namespace minitensor;
using Clock = std::chrono::high_resolution_clock;

volatile double benchmark_sink = 0.0;

void scalar_add(const Tensor& a, const Tensor& b, Tensor& result) {
    for (std::size_t i = 0; i < a.size(); ++i)
        result.data()[i] = a.data()[i] + b.data()[i];
}

void scalar_subtract(const Tensor& a, const Tensor& b, Tensor& result) {
    for (std::size_t i = 0; i < a.size(); ++i)
        result.data()[i] = a.data()[i] - b.data()[i];
}

void scalar_multiply(const Tensor& a, const Tensor& b, Tensor& result) {
    for (std::size_t i = 0; i < a.size(); ++i)
        result.data()[i] = a.data()[i] * b.data()[i];
}

template <typename Function>
double measure(Function function, int repetitions = 5) {
    std::vector<double> times;

    for (int i = 0; i < 2; ++i) {
        Tensor result = function();
        benchmark_sink += result.data()[result.size() - 1];
    }

    for (int i = 0; i < repetitions; ++i) {
        auto start = Clock::now();

        Tensor result = function();

        auto end = Clock::now();

        benchmark_sink += result.data()[result.size() - 1];

        double elapsed =
            std::chrono::duration<double, std::milli>(end - start).count();

        times.push_back(elapsed);
    }

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

bool verify(const Tensor& expected, const Tensor& actual) {
    if (expected.size() != actual.size())
        return false;

    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(expected.data()[i] - actual.data()[i]) > 1e-10)
            return false;
    }

    return true;
}

template <typename ScalarFunction,
          typename SimdFunction,
          typename ParallelFunction>
void benchmark_operation(
    const std::string& name,
    const Tensor& a,
    const Tensor& b,
    ScalarFunction scalar,
    SimdFunction simd,
    ParallelFunction parallel
) {
    Tensor scalar_result({a.size()});
    scalar(a, b, scalar_result);

    double scalar_time = measure([&]() {
        Tensor result({a.size()});
        scalar(a, b, result);
        return result;
    });

    double simd_time = measure([&]() {
        return simd(a, b);
    });

    Tensor simd_result = simd(a, b);

    std::cout << "\n" << name << "\n";
    std::cout << "Scalar: " << scalar_time << " ms\n";

    std::cout << "SIMD:   " << simd_time << " ms"
              << " | Speedup: " << scalar_time / simd_time << "x"
              << " | " << (verify(scalar_result, simd_result) ? "PASS" : "FAIL")
              << "\n";

    const int thread_counts[] = {1, 2, 4, 8, 16};

    for (int threads : thread_counts) {
        omp_set_num_threads(threads);

        double parallel_time = measure([&]() {
            return parallel(a, b);
        });

        Tensor parallel_result = parallel(a, b);

        std::cout << "Parallel (" << std::setw(2) << threads << " threads): "
                  << std::setw(10) << parallel_time << " ms"
                  << " | Speedup: " << std::fixed << std::setprecision(3)
                  << scalar_time / parallel_time << "x"
                  << " | "
                  << (verify(scalar_result, parallel_result) ? "PASS" : "FAIL")
                  << "\n";
    }
}

void run_benchmark(std::size_t size) {
    std::cout << "\n========================================\n";
    std::cout << "Tensor size: " << size << " elements\n";
    std::cout << "========================================\n";

    Tensor a({size});
    Tensor b({size});

    for (std::size_t i = 0; i < size; ++i) {
        a.data()[i] = (i % 1000) * 0.25;
        b.data()[i] = (i % 500) * 0.15;
    }

    benchmark_operation(
        "ADDITION",
        a, b,
        scalar_add,
        add,
        add_parallel
    );

    benchmark_operation(
        "SUBTRACTION",
        a, b,
        scalar_subtract,
        subtract,
        subtract_parallel
    );

    benchmark_operation(
        "MULTIPLICATION",
        a, b,
        scalar_multiply,
        multiply,
        multiply_parallel
    );
}

int main() {
    std::cout << "MINITENSOR ELEMENTWISE BENCHMARK\n";
    std::cout << "Scalar vs SIMD vs Parallel SIMD\n";

    omp_set_dynamic(0);

    run_benchmark(1000000);
    run_benchmark(4000000);
    run_benchmark(16000000);

    std::cout << "\nBenchmark completed.\n";
    std::cout << "Prevent-optimization checksum: "
              << benchmark_sink << "\n";

    return 0;
}
