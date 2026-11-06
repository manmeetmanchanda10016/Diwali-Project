#include "minitensor/Ops.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <vector>

using namespace minitensor;
using Clock = std::chrono::steady_clock;

int main()
{
    try
    {
        omp_set_dynamic(0);
        volatile double sink = 0;
        constexpr int warmups = 3, measurements = 10;
        std::cout << "Rows,Inner,Cols,Threads,Warmups,Measurements,Average_ms,Median_ms,GFLOPS,Correctness";
        for (int i = 0; i < measurements; ++i) std::cout << ",Sample" << i << "_ms";
        std::cout << '\n' << std::fixed << std::setprecision(6);
        const std::array<std::array<std::size_t, 3>, 5> shapes{{
            {{64, 64, 64}}, {{257, 259, 261}}, {{512, 512, 512}},
            {{1024, 1024, 1024}}, {{4, 1024, 512}}
        }};
        for (const auto& shape : shapes)
        {
            const auto m = shape[0], k = shape[1], n = shape[2];
            Tensor a({m, k}), b({k, n});
            for (std::size_t i = 0; i < a.size(); ++i)
                a.flat_at(i) = (static_cast<int>(i % 17) - 8) / 9.0;
            for (std::size_t i = 0; i < b.size(); ++i)
                b.flat_at(i) = (static_cast<int>(i % 23) - 11) / 12.0;
            // This is an unchanged library variant. The independent wider
            // reference and randomized checks live in MatmulReferenceTests.
            const Tensor expected = matmul_blocked(a, b, 128);
            for (int threads : {1, 2, 4, 8})
            {
                omp_set_num_threads(threads);
                const Tensor actual = matmul_parallel(a, b);
                for (std::size_t i = 0; i < actual.size(); ++i)
                    if (!std::isfinite(actual.flat_at(i)) ||
                        std::abs(actual.flat_at(i) - expected.flat_at(i)) >
                            1e-10 + 1e-10 * std::abs(expected.flat_at(i)))
                        throw std::runtime_error("Focused matmul correctness failure");
                for (int i = 0; i < warmups; ++i)
                {
                    const Tensor out = matmul_parallel(a, b);
                    sink += out.flat_at(out.size() - 1);
                }
                std::vector<double> samples;
                double total = 0;
                for (int i = 0; i < measurements; ++i)
                {
                    double value = 0;
                    const auto begin = Clock::now();
                    {
                        const Tensor out = matmul_parallel(a, b);
                        value = out.flat_at(out.size() - 1);
                    } // Include output allocation, initialization and destruction.
                    const double ms = std::chrono::duration<double, std::milli>(
                        Clock::now() - begin).count();
                    samples.push_back(ms);
                    total += ms;
                    sink += value;
                }
                auto sorted = samples;
                std::sort(sorted.begin(), sorted.end());
                const double median = (sorted[4] + sorted[5]) / 2;
                std::cout << m << ',' << k << ',' << n << ',' << threads << ','
                          << warmups << ',' << measurements << ',' << total / measurements
                          << ',' << median << ',' << (2.0 * m * k * n / (median * 1e6))
                          << ",PASS";
                for (const double ms : samples) std::cout << ',' << ms;
                std::cout << std::endl;
            }
        }
        std::cerr << "Checksum: " << sink << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
