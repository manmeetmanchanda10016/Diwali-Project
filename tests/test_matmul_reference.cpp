#include "minitensor/Ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <omp.h>
#include <random>
#include <stdexcept>

using namespace minitensor;

namespace {

// Deliberately independent i-j-k dot products with wider accumulation.
Tensor reference(const Tensor& a, const Tensor& b)
{
    const auto m = a.shape().dimensions()[0];
    const auto k = a.shape().dimensions()[1];
    const auto n = b.shape().dimensions()[1];
    Tensor out({m, n});
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
        {
            long double value = 0;
            for (std::size_t p = 0; p < k; ++p)
                value += static_cast<long double>(a.at({i, p})) *
                         static_cast<long double>(b.at({p, j}));
            out.at({i, j}) = static_cast<double>(value);
        }
    return out;
}

void check(const Tensor& actual, const Tensor& expected)
{
    if (actual.shape().dimensions() != expected.shape().dimensions())
        throw std::runtime_error("Incorrect output shape");
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        const double x = actual.flat_at(i);
        const double y = expected.flat_at(i);
        if (!std::isfinite(x) || std::abs(x - y) > 1e-10 + 1e-10 * std::abs(y))
            throw std::runtime_error("Matrix result differs from independent reference");
    }
}

template <typename Function>
void invalid(Function function)
{
    try { function(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Expected invalid_argument");
}

} // namespace

int main()
{
    try
    {
        omp_set_dynamic(0);
        std::mt19937 rng(20261106);
        std::uniform_real_distribution<double> values(-2.0, 2.0);
        std::size_t shapes = 0;
        std::size_t checks = 0;
        const auto exercise = [&](std::size_t m, std::size_t k, std::size_t n)
        {
            Tensor a({m, k}), b({k, n});
            for (std::size_t i = 0; i < a.size(); ++i) a.flat_at(i) = values(rng);
            for (std::size_t i = 0; i < b.size(); ++i) b.flat_at(i) = values(rng);
            const Tensor a_before = a, b_before = b;
            const Tensor expected = reference(a, b);
            for (int threads : {1, 2, 4})
            {
                omp_set_num_threads(threads);
                check(matmul(a, b), expected);
                check(matmul_parallel(a, b), expected);
                checks += 2;
                for (std::size_t tile : {std::size_t(1), std::size_t(7),
                        std::size_t(32), std::size_t(128),
                        std::numeric_limits<std::size_t>::max()})
                {
                    check(matmul_blocked(a, b, tile), expected);
                    check(matmul_parallel_blocked(a, b, tile), expected);
                    checks += 2;
                }
            }
            check(a, a_before);
            check(b, b_before);
            ++shapes;
        };

        // Empty dimensions, singleton axes, signed random inputs, fringe tiles.
        for (std::size_t m : {0U, 1U, 3U, 7U, 17U})
            for (std::size_t k : {0U, 1U, 2U, 9U, 33U})
                for (std::size_t n : {0U, 1U, 5U, 8U, 31U})
                    exercise(m, k, n);
        const std::array<std::array<std::size_t, 3>, 8> boundaries{{
            {{31, 33, 35}}, {{63, 65, 67}}, {{65, 63, 65}}, {{127, 3, 129}},
            {{129, 5, 127}}, {{3, 129, 131}}, {{1, 513, 515}}, {{65, 65, 65}}
        }};
        for (const auto& dims : boundaries) exercise(dims[0], dims[1], dims[2]);

        // Both inputs may be the same tensor; operations must not modify it.
        Tensor same({9, 9});
        for (std::size_t i = 0; i < same.size(); ++i) same.flat_at(i) = values(rng);
        const Tensor before = same;
        const Tensor square = reference(same, same);
        check(matmul(same, same), square);
        check(matmul_parallel(same, same), square);
        check(matmul_blocked(same, same, 7), square);
        check(matmul_parallel_blocked(same, same, 7), square);
        check(same, before);
        for (const auto function : {matmul, matmul_parallel})
        {
            invalid([&] { function(Tensor({3}), Tensor({3, 2})); });
            invalid([&] { function(Tensor({2, 3}), Tensor({4, 2})); });
        }
        for (const auto function : {matmul_blocked, matmul_parallel_blocked})
        {
            invalid([&] { function(Tensor({3}), Tensor({3, 2}), 7); });
            invalid([&] { function(Tensor({2, 3}), Tensor({4, 2}), 7); });
            invalid([&] { function(Tensor({2, 3}), Tensor({3, 2}), 0); });
        }
        std::cout << shapes << " shapes, " << checks
                  << " variant/thread/tile checks passed; self-input and error cases passed\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
