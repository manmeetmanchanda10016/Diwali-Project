
#include "minitensor/Ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace minitensor;
using Clock = std::chrono::steady_clock;

volatile double benchmark_sink = 0.0;

// --------------------------------------------------
// Precomputed broadcasting metadata
// --------------------------------------------------

struct BroadcastInfo
{
    std::vector<std::size_t> output_strides;
    std::vector<std::size_t> a_strides;
    std::vector<std::size_t> b_strides;
    std::size_t count;
};

BroadcastInfo prepare_broadcast(
    const Tensor& a,
    const Tensor& b)
{
    const auto& a_dims = a.shape().dimensions();
    const auto& b_dims = b.shape().dimensions();

    const std::size_t rank =
        std::max(a_dims.size(), b_dims.size());

    std::vector<std::size_t> output_dims(rank);

    for (std::size_t i = 0; i < rank; ++i)
    {
        const std::size_t ad =
            i < rank - a_dims.size()
                ? 1
                : a_dims[i - (rank - a_dims.size())];

        const std::size_t bd =
            i < rank - b_dims.size()
                ? 1
                : b_dims[i - (rank - b_dims.size())];

        if (ad != bd && ad != 1 && bd != 1)
        {
            throw std::invalid_argument(
                "Incompatible broadcast shapes"
            );
        }

        output_dims[i] = std::max(ad, bd);
    }

    BroadcastInfo info;

    info.count = 1;

    for (std::size_t i = 0; i < rank; ++i)
    {
        info.count *= output_dims[i];
    }

    info.output_strides.resize(rank);
    info.a_strides.assign(rank, 0);
    info.b_strides.assign(rank, 0);

    std::size_t stride = 1;

    for (std::size_t i = rank; i-- > 0;)
    {
        info.output_strides[i] = stride;
        stride *= output_dims[i];
    }

    std::size_t a_stride = 1;
    std::size_t b_stride = 1;

    for (std::size_t i = rank; i-- > 0;)
    {
        if (i >= rank - a_dims.size())
        {
            const std::size_t dim =
                a_dims[i - (rank - a_dims.size())];

            if (dim != 1)
            {
                info.a_strides[i] = a_stride;
            }

            a_stride *= dim;
        }

        if (i >= rank - b_dims.size())
        {
            const std::size_t dim =
                b_dims[i - (rank - b_dims.size())];

            if (dim != 1)
            {
                info.b_strides[i] = b_stride;
            }

            b_stride *= dim;
        }
    }

    return info;
}

// --------------------------------------------------
// Optimized broadcasting
// --------------------------------------------------

template <typename Operation>
Tensor optimized_broadcast(
    const Tensor& a,
    const Tensor& b,
    Operation operation)
{
    const BroadcastInfo info = prepare_broadcast(a, b);

    const auto& a_dims = a.shape().dimensions();
    const auto& b_dims = b.shape().dimensions();

    const std::size_t rank = info.output_strides.size();

    std::vector<std::size_t> output_dims(rank);

    for (std::size_t i = 0; i < rank; ++i)
    {
        const std::size_t ad =
            i < rank - a_dims.size()
                ? 1
                : a_dims[i - (rank - a_dims.size())];

        const std::size_t bd =
            i < rank - b_dims.size()
                ? 1
                : b_dims[i - (rank - b_dims.size())];

        output_dims[i] = std::max(ad, bd);
    }

    Tensor result(output_dims);

    const double* A = a.data();
    const double* B = b.data();
    double* C = result.data();

    for (std::size_t flat = 0; flat < info.count; ++flat)
    {
        std::size_t remainder = flat;

        std::size_t offset_a = 0;
        std::size_t offset_b = 0;

        for (std::size_t d = 0; d < rank; ++d)
        {
            const std::size_t coordinate =
                remainder / info.output_strides[d];

            remainder %= info.output_strides[d];

            offset_a += coordinate * info.a_strides[d];
            offset_b += coordinate * info.b_strides[d];
        }

        C[flat] = operation(A[offset_a], B[offset_b]);
    }

    return result;
}

// --------------------------------------------------
// Timing
// --------------------------------------------------

template <typename Function>
double measure(
    Function function,
    int warmups = 2,
    int measurements = 7)
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

// --------------------------------------------------
// Verification
// --------------------------------------------------

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

// --------------------------------------------------
// Test data
// --------------------------------------------------

Tensor create_tensor(
    const std::vector<std::size_t>& dimensions)
{
    Tensor result(dimensions);

    for (std::size_t i = 0; i < result.size(); ++i)
    {
        result.data()[i] =
            static_cast<double>(i % 1000) * 0.25;
    }

    return result;
}

// --------------------------------------------------
// Benchmark operations
// --------------------------------------------------

void run_case(
    const std::string& name,
    const Tensor& a,
    const Tensor& b)
{
    std::cout << "\n" << name << '\n';

    auto run_operation = [&](int operation, bool optimized)
    {
        auto function = [&]() -> Tensor
        {
            if (optimized)
            {
                if (operation == 0)
                    return optimized_broadcast(
                        a, b,
                        [](double x, double y) { return x + y; }
                    );

                if (operation == 1)
                    return optimized_broadcast(
                        a, b,
                        [](double x, double y) { return x - y; }
                    );

                return optimized_broadcast(
                    a, b,
                    [](double x, double y) { return x * y; }
                );
            }

            if (operation == 0)
                return add_broadcast(a, b);

            if (operation == 1)
                return subtract_broadcast(a, b);

            return multiply_broadcast(a, b);
        };

        Tensor reference = function();

        Tensor expected =
            operation == 0 ? add_broadcast(a, b) :
            operation == 1 ? subtract_broadcast(a, b) :
                             multiply_broadcast(a, b);

        if (!verify(reference, expected))
        {
            throw std::runtime_error(
                "Broadcasting verification failed"
            );
        }

        return measure(function);
    };

    const char* names[] = {"Add", "Subtract", "Multiply"};

    for (int operation = 0; operation < 3; ++operation)
    {
        const double current_time =
            run_operation(operation, false);

        const double optimized_time =
            run_operation(operation, true);

        std::cout << names[operation] << '\n';

        std::cout << "Current: "
                  << current_time << " ms\n";

        std::cout << "Optimized: "
                  << optimized_time << " ms\n";

        std::cout << "Speedup: "
                  << current_time / optimized_time
                  << "x\n";
    }
}

int main()
{
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "MINITENSOR BROADCASTING BENCHMARK\n";
    std::cout << "========================================\n";

    {
        Tensor a = create_tensor({256, 256});
        Tensor b = create_tensor({256, 1});

        run_case("Case 1: (256,256) + (256,1)", a, b);
    }

    {
        Tensor a = create_tensor({512, 512});
        Tensor b = create_tensor({1, 512});

        run_case("Case 2: (512,512) + (1,512)", a, b);
    }

    {
        Tensor a = create_tensor({256, 256, 16});
        Tensor b = create_tensor({1, 1, 16});

        run_case("Case 3: (256,256,16) + (1,1,16)", a, b);
    }

    {
        Tensor a = create_tensor({512, 512});
        Tensor b = create_tensor({512, 512});

        run_case("Case 4: (512,512) + (512,512)", a, b);
    }

    std::cout << "\n========================================\n";
    std::cout << "Benchmark completed successfully.\n";
    std::cout << "Checksum: " << benchmark_sink << '\n';

    return 0;
}
