
#include "minitensor/Ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace minitensor;
using Clock = std::chrono::steady_clock;

volatile double benchmark_sink = 0.0;

Tensor normal_transpose(const Tensor& a)
{
    const std::size_t rows = a.shape().dimensions()[0];
    const std::size_t cols = a.shape().dimensions()[1];

    Tensor result({cols, rows});

    const double* input = a.data();
    double* output = result.data();

    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            output[j * rows + i] = input[i * cols + j];
        }
    }

    return result;
}

Tensor blocked_transpose(const Tensor& a, std::size_t block_size)
{
    const std::size_t rows = a.shape().dimensions()[0];
    const std::size_t cols = a.shape().dimensions()[1];

    Tensor result({cols, rows});

    const double* input = a.data();
    double* output = result.data();

    for (std::size_t ii = 0; ii < rows; ii += block_size)
    {
        for (std::size_t jj = 0; jj < cols; jj += block_size)
        {
            const std::size_t i_end =
                ii + std::min(block_size, rows - ii);

            const std::size_t j_end =
                jj + std::min(block_size, cols - jj);

            for (std::size_t i = ii; i < i_end; ++i)
            {
                for (std::size_t j = jj; j < j_end; ++j)
                {
                    output[j * rows + i] =
                        input[i * cols + j];
                }
            }
        }
    }

    return result;
}

template <typename Function>
double measure(Function function, int warmups = 2, int measurements = 7)
{
    for (int i = 0; i < warmups; ++i)
    {
        Tensor result = function();
        benchmark_sink += result.data()[result.size() / 2];
    }

    std::vector<double> times;

    for (int i = 0; i < measurements; ++i)
    {
        const auto start = Clock::now();

        Tensor result = function();

        const auto end = Clock::now();

        benchmark_sink += result.data()[result.size() / 2];

        const double elapsed =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        times.push_back(elapsed);
    }

    std::sort(times.begin(), times.end());

    return times[times.size() / 2];
}

bool verify(const Tensor& a, const Tensor& b)
{
    if (a.shape().dimensions() != b.shape().dimensions())
    {
        return false;
    }

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (std::abs(a.data()[i] - b.data()[i]) > 1e-10)
        {
            return false;
        }
    }

    return true;
}

Tensor create_matrix(std::size_t rows, std::size_t cols)
{
    Tensor result({rows, cols});

    for (std::size_t i = 0; i < result.size(); ++i)
    {
        result.data()[i] =
            static_cast<double>(i % 1000) * 0.25;
    }

    return result;
}

void run_benchmark(std::size_t size)
{
    std::cout << "\n========================================\n";
    std::cout << "MATRIX SIZE: " << size << " x " << size << '\n';
    std::cout << "========================================\n";

    Tensor input = create_matrix(size, size);

    Tensor reference = normal_transpose(input);
    Tensor library_result = transpose(input);

    if (!verify(reference, library_result))
    {
        std::cout << "ERROR: Library transpose is incorrect!\n";
        return;
    }

    std::cout << "\nNormal transpose:\n";

    const double normal_time = measure(
        [&]() { return normal_transpose(input); }
    );

    std::cout << "Normal: "
              << normal_time << " ms\n";

    std::cout << "\nLibrary transpose:\n";

    const double library_time = measure(
        [&]() { return transpose(input); }
    );

    std::cout << "Library: "
              << library_time << " ms\n";

    std::cout << "\nBlocked transpose results:\n";

    for (std::size_t block_size : {16, 32, 64, 128})
    {
        Tensor blocked_result =
            blocked_transpose(input, block_size);

        if (!verify(reference, blocked_result))
        {
            std::cout << "Block " << block_size
                      << ": INCORRECT RESULT\n";
            continue;
        }

        const double blocked_time = measure(
            [&]() {
                return blocked_transpose(input, block_size);
            }
        );

        std::cout << "Block " << std::setw(3)
                  << block_size << ": "
                  << blocked_time << " ms";

        std::cout << " | Speedup: "
                  << normal_time / blocked_time
                  << "x\n";
    }

    std::cout << "\nLibrary speedup: "
              << normal_time / library_time
              << "x\n";
}

int main()
{
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "MINITENSOR TRANSPOSE BENCHMARK\n";
    std::cout << "========================================\n";

    for (std::size_t size : {256, 512, 1024, 2048})
    {
        run_benchmark(size);
    }

    std::cout << "\n========================================\n";
    std::cout << "Benchmark completed successfully.\n";
    std::cout << "Checksum: " << benchmark_sink << '\n';

    return 0;
}
