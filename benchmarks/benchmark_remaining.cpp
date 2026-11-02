
#include "minitensor/Tensor.h"
#include "minitensor/Ops.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

volatile double result_sink = 0.0;

template <typename Function>
double benchmark_ms(Function function, int repetitions = 7, int warmups = 2) {
    for (int i = 0; i < warmups; ++i) {
        result_sink = result_sink + function();
    }

    std::vector<double> times;
    times.reserve(repetitions);

    for (int i = 0; i < repetitions; ++i) {
        const auto start = Clock::now();

        const double result = function();

        const auto finish = Clock::now();

        result_sink = result_sink + result;

        std::chrono::duration<double, std::milli> elapsed =
            finish - start;

        times.push_back(elapsed.count());
    }

    std::sort(times.begin(), times.end());

    return times[times.size() / 2];
}

void print_result(
    const std::string& name,
    const std::string& size,
    double ms
) {
    std::cout << std::left << std::setw(30) << name
              << std::setw(18) << size
              << std::right << std::fixed << std::setprecision(4)
              << ms << " ms\n";
}

void benchmark_construction() {
    std::cout << "\n=== Tensor Construction / Allocation ===\n";

    for (std::size_t count : {1024ULL, 100000ULL, 1000000ULL}) {

        const std::string label =
            std::to_string(count) + " elements";

        const double time = benchmark_ms([count]() {

            minitensor::Tensor tensor({count}, 1.5);

            return tensor.size() == 0
                ? 0.0
                : tensor.flat_at(count - 1);

        }, 9, 2);

        print_result("Tensor construction", label, time);
    }
}

void benchmark_indexing() {
    constexpr std::size_t rows = 512;
    constexpr std::size_t cols = 512;

    minitensor::Tensor tensor({rows, cols}, 1.0);

    std::cout << "\n=== Indexing (512 x 512) ===\n";

    const double flat_time = benchmark_ms([&]() {

        double total = 0.0;

        for (std::size_t i = 0; i < tensor.size(); ++i) {
            total += tensor.flat_at(i);
        }

        return total;

    }, 7, 2);

    const double multidimensional_time = benchmark_ms([&]() {

        double total = 0.0;

        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                total += tensor.at({i, j});
            }
        }

        return total;

    }, 7, 2);

    print_result("flat_at() read", "262144 accesses", flat_time);

    print_result(
        "at({row, col}) read",
        "262144 accesses",
        multidimensional_time
    );
}

void benchmark_reshape() {
    std::cout << "\n=== Reshape ===\n";

    minitensor::Tensor tensor({512, 512}, 2.0);

    const double time = benchmark_ms([&]() {

        tensor.reshape({256, 1024});

        const double value =
            static_cast<double>(tensor.size());

        tensor.reshape({512, 512});

        return value;

    }, 9, 2);

    print_result(
        "reshape (two shape changes)",
        "262144 elements",
        time
    );
}

void benchmark_fill() {
    std::cout << "\n=== Fill ===\n";

    for (std::size_t count : {1024ULL, 100000ULL, 1000000ULL}) {

        minitensor::Tensor tensor({count}, 0.0);

        const std::string label =
            std::to_string(count) + " elements";

        const double time = benchmark_ms([&]() {

            tensor.fill(3.25);

            return tensor.flat_at(count - 1);

        }, 7, 2);

        print_result("fill()", label, time);
    }
}

void benchmark_reductions() {
    constexpr std::size_t count = 1000000;

    minitensor::Tensor tensor({count}, 0.0);

    for (std::size_t i = 0; i < count; ++i) {
        tensor.flat_at(i) =
            static_cast<double>((i % 1000) + 1);
    }

    std::cout << "\n=== Reductions (1,000,000 elements) ===\n";

    print_result(
        "sum()",
        "1000000 elements",
        benchmark_ms([&]() {
            return minitensor::sum(tensor);
        })
    );

    print_result(
        "mean()",
        "1000000 elements",
        benchmark_ms([&]() {
            return minitensor::mean(tensor);
        })
    );

    print_result(
        "min()",
        "1000000 elements",
        benchmark_ms([&]() {
            return minitensor::min(tensor);
        })
    );

    print_result(
        "max()",
        "1000000 elements",
        benchmark_ms([&]() {
            return minitensor::max(tensor);
        })
    );
}

}

int main() {
    std::cout << "MiniTensor Remaining Operations Benchmark\n";

    std::cout
        << "Median of repeated runs; Release build recommended.\n";

    benchmark_construction();

    benchmark_indexing();

    benchmark_reshape();

    benchmark_fill();

    benchmark_reductions();

    std::cout << "\nBenchmark completed.\n";

    return 0;
}
