
#include "minitensor/Ops.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <stdexcept>

using namespace minitensor;
using namespace std;
using namespace chrono;

// Normal matrix multiplication (baseline)
Tensor normal_matmul(const Tensor& a, const Tensor& b)
{
    const size_t rows = a.shape().dimensions()[0];
    const size_t inner = a.shape().dimensions()[1];
    const size_t cols = b.shape().dimensions()[1];

    Tensor result({rows, cols}, 0.0);

    const double* A = a.data();
    const double* B = b.data();
    double* C = result.data();

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t k = 0; k < inner; ++k)
        {
            double value = A[i * inner + k];

            for (size_t j = 0; j < cols; ++j)
            {
                C[i * cols + j] +=
                    value * B[k * cols + j];
            }
        }
    }

    return result;
}

// Blocked matrix multiplication
Tensor blocked_matmul(const Tensor& a, const Tensor& b)
{
    const size_t rows = a.shape().dimensions()[0];
    const size_t inner = a.shape().dimensions()[1];
    const size_t cols = b.shape().dimensions()[1];

    constexpr size_t BLOCK_SIZE = 32;

    Tensor result({rows, cols}, 0.0);

    const double* A = a.data();
    const double* B = b.data();
    double* C = result.data();

    for (size_t ii = 0; ii < rows; ii += BLOCK_SIZE)
    {
        for (size_t kk = 0; kk < inner; kk += BLOCK_SIZE)
        {
            for (size_t jj = 0; jj < cols; jj += BLOCK_SIZE)
            {
                const size_t i_end = min(ii + BLOCK_SIZE, rows);
                const size_t k_end = min(kk + BLOCK_SIZE, inner);
                const size_t j_end = min(jj + BLOCK_SIZE, cols);

                for (size_t i = ii; i < i_end; ++i)
                {
                    for (size_t k = kk; k < k_end; ++k)
                    {
                        const double value = A[i * inner + k];

                        for (size_t j = jj; j < j_end; ++j)
                        {
                            C[i * cols + j] +=
                                value * B[k * cols + j];
                        }
                    }
                }
            }
        }
    }

    return result;
}

// Benchmark function
template <typename Function>
double benchmark(Function function, double& checksum)
{
    vector<double> times;

    // Warm-up
    for (int i = 0; i < 3; ++i)
    {
        Tensor result = function();

        checksum += result.data()[0];
        checksum += result.data()[result.size() - 1];
    }

    // Measurements
    for (int i = 0; i < 7; ++i)
    {
        auto start = steady_clock::now();

        Tensor result = function();

        auto end = steady_clock::now();

        double elapsed =
            duration<double, milli>(end - start).count();

        times.push_back(elapsed);

        checksum += result.data()[0];
        checksum += result.data()[result.size() - 1];
    }

    sort(times.begin(), times.end());

    return times[times.size() / 2];
}

// Verify multiplication results
void verify_result(const Tensor& result, size_t n)
{
    const double expected = 2.0 * static_cast<double>(n);

    for (size_t i = 0; i < result.size(); ++i)
    {
        if (abs(result.data()[i] - expected) > 1e-9)
        {
            throw runtime_error(
                "Matrix multiplication produced an incorrect result"
            );
        }
    }
}

int main()
{
    cout << fixed << setprecision(3);

    vector<size_t> sizes = {
        64, 128, 256, 512, 1024
    };

    double global_checksum = 0.0;

    cout << "\nMATRIX MULTIPLICATION BENCHMARK\n";
    cout << "============================================\n";
    cout << "Warmups: 3 | Measurements: 7 | Statistic: Median\n";

    for (size_t n : sizes)
    {
        Tensor a({n, n});
        Tensor b({n, n});

        for (size_t i = 0; i < n * n; ++i)
        {
            a.data()[i] = 1.0;
            b.data()[i] = 2.0;
        }

        cout << "\n--------------------------------------------\n";
        cout << "Matrix size: " << n << " x " << n << '\n';
        cout << "--------------------------------------------\n";

        // Verify all implementations before benchmarking
        verify_result(normal_matmul(a, b), n);
        verify_result(blocked_matmul(a, b), n);
        verify_result(matmul_parallel(a, b), n);
        verify_result(matmul(a, b), n);

        double checksum = 0.0;

        double normal_time = benchmark(
            [&]() { return normal_matmul(a, b); },
            checksum
        );

        double blocked_time = benchmark(
            [&]() { return blocked_matmul(a, b); },
            checksum
        );

        double parallel_time = benchmark(
            [&]() { return matmul_parallel(a, b); },
            checksum
        );

        // This calls the actual library hybrid implementation
        double hybrid_time = benchmark(
            [&]() { return matmul(a, b); },
            checksum
        );

        cout << "\nNormal:          " << normal_time << " ms\n";
        cout << "Blocked:         " << blocked_time << " ms\n";
        cout << "OpenMP:          " << parallel_time << " ms\n";
        cout << "Library Hybrid:  " << hybrid_time << " ms\n";

        cout << "\nSpeedup compared with normal:\n";

        cout << "Blocked:         "
             << normal_time / blocked_time << "x\n";

        cout << "OpenMP:          "
             << normal_time / parallel_time << "x\n";

        cout << "Library Hybrid:  "
             << normal_time / hybrid_time << "x\n";

        global_checksum += checksum;
    }

    cout << "\n============================================\n";
    cout << "Global checksum: " << global_checksum << '\n';

    return 0;
}
