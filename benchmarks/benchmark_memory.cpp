
#include "minitensor/Tensor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <new>
#include <optional>
#include <string>
#include <vector>

using namespace minitensor;
using Clock = std::chrono::steady_clock;

constexpr int WARMUPS = 3;
constexpr int MEASUREMENTS = 10;
constexpr std::size_t ALIGNMENT = 64;
constexpr std::size_t BATCH_SIZE = 16;

volatile double benchmark_sink = 0.0;

struct BenchmarkResult
{
    double average_ms;
    double median_ms;
};

template <typename Function>
BenchmarkResult measure(Function function)
{
    for (int i = 0; i < WARMUPS; ++i)
        function();

    std::vector<double> times;
    times.reserve(MEASUREMENTS);

    double total = 0.0;

    for (int i = 0; i < MEASUREMENTS; ++i)
    {
        const auto start = Clock::now();

        function();

        const auto end = Clock::now();

        const double elapsed =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        times.push_back(elapsed);
        total += elapsed;
    }

    std::sort(times.begin(), times.end());

    const double average = total / MEASUREMENTS;

    const double median =
        (times[MEASUREMENTS / 2 - 1] +
         times[MEASUREMENTS / 2]) / 2.0;

    return {average, median};
}

void print_result(
    const std::string& name,
    const BenchmarkResult& result)
{
    std::cout << std::left << std::setw(25)
              << name
              << std::right << std::fixed
              << std::setprecision(6)
              << "Average: " << result.average_ms
              << " ms | Median: " << result.median_ms
              << " ms\n";
}

void save_csv(
    std::ofstream& csv,
    std::size_t elements,
    const std::string& operation,
    const BenchmarkResult& result)
{
    csv << elements << ','
        << elements * sizeof(double) << ','
        << operation << ','
        << WARMUPS << ','
        << MEASUREMENTS << ','
        << std::fixed << std::setprecision(8)
        << result.average_ms << ','
        << result.median_ms << '\n';
}

void* aligned_allocate(std::size_t bytes)
{
    return ::operator new(
        bytes,
        std::align_val_t{ALIGNMENT}
    );
}

void aligned_deallocate(void* pointer)
{
    ::operator delete(
        pointer,
        std::align_val_t{ALIGNMENT}
    );
}

BenchmarkResult benchmark_allocation(
    std::size_t bytes)
{
    std::vector<void*> pointers(BATCH_SIZE, nullptr);

    return measure([&]()
    {
        for (std::size_t i = 0; i < BATCH_SIZE; ++i)
            pointers[i] = aligned_allocate(bytes);

        // Cleanup is intentionally outside the timed operation.
        for (void*& pointer : pointers)
        {
            aligned_deallocate(pointer);
            pointer = nullptr;
        }
    });
}

BenchmarkResult benchmark_initialization(
    std::size_t elements)
{
    const std::size_t bytes =
        elements * sizeof(double);

    std::vector<double*> buffers(
        BATCH_SIZE,
        nullptr
    );

    for (double*& pointer : buffers)
    {
        pointer = static_cast<double*>(
            aligned_allocate(bytes)
        );
    }

    const BenchmarkResult result = measure([&]()
    {
        double checksum = 0.0;

        for (double* pointer : buffers)
        {
            std::fill_n(pointer, elements, 1.25);
            checksum += pointer[elements - 1];
        }

        benchmark_sink =
            benchmark_sink + checksum;
    });

    for (double* pointer : buffers)
        aligned_deallocate(pointer);

    return result;
}

BenchmarkResult benchmark_deallocation(
    std::size_t bytes)
{
    return measure([&]()
    {
        std::vector<void*> pointers(
            BATCH_SIZE,
            nullptr
        );

        // Prepare memory before starting the timer.
        for (void*& pointer : pointers)
            pointer = aligned_allocate(bytes);

        const auto start = Clock::now();

        for (void*& pointer : pointers)
        {
            aligned_deallocate(pointer);
            pointer = nullptr;
        }

        const auto end = Clock::now();

        benchmark_sink =
            benchmark_sink +
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();
    });
}

BenchmarkResult benchmark_tensor_construction(
    std::size_t elements)
{
    const std::vector<std::size_t> dimensions = {
        elements
    };

    std::optional<Tensor> tensor;

    return measure([&]()
    {
        const auto start = Clock::now();

        tensor.emplace(dimensions, 1.25);

        const auto end = Clock::now();

        benchmark_sink =
            benchmark_sink +
            tensor->data()[elements - 1];

        // Tensor destruction is outside this timing.
        tensor.reset();

        benchmark_sink =
            benchmark_sink +
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();
    });
}

BenchmarkResult benchmark_tensor_destruction(
    std::size_t elements)
{
    const std::vector<std::size_t> dimensions = {
        elements
    };

    std::optional<Tensor> tensor;

    return measure([&]()
    {
        tensor.emplace(dimensions, 1.25);

        benchmark_sink =
            benchmark_sink +
            tensor->data()[elements - 1];

        const auto start = Clock::now();

        tensor.reset();

        const auto end = Clock::now();

        benchmark_sink =
            benchmark_sink +
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();
    });
}

int main()
{
    const std::vector<std::size_t> sizes = {
        1024,
        16384,
        65536,
        262144
    };

    std::ofstream csv("memory_benchmark.csv");

    if (!csv.is_open())
    {
        std::cerr << "Could not create CSV file.\n";
        return 1;
    }

    csv << "Elements,Bytes,Operation,Warmups,"
        << "Measurements,Average_ms,Median_ms\n";

    std::cout
        << "\nMINITENSOR MEMORY BENCHMARK\n";

    std::cout
        << "============================================\n";

    std::cout << "Warmups: " << WARMUPS
              << " | Measurements: "
              << MEASUREMENTS << '\n';

    std::cout << "Alignment: "
              << ALIGNMENT << " bytes\n";

    std::cout << "Allocation batch: "
              << BATCH_SIZE << '\n';

    for (const std::size_t elements : sizes)
    {
        const std::size_t bytes =
            elements * sizeof(double);

        std::cout
            << "\n============================================\n";

        std::cout << "ELEMENTS: " << elements << '\n';

        std::cout << "MEMORY: "
                  << bytes << " bytes\n";

        std::cout
            << "============================================\n";

        // Raw aligned allocation

        BenchmarkResult allocation =
            benchmark_allocation(bytes);

        print_result("Raw allocation", allocation);

        save_csv(
            csv, elements,
            "RawAllocation",
            allocation
        );

        // Initialization

        BenchmarkResult initialization =
            benchmark_initialization(elements);

        print_result("Initialization", initialization);

        save_csv(
            csv, elements,
            "Initialization",
            initialization
        );

        // Raw aligned deallocation

        BenchmarkResult deallocation =
            benchmark_deallocation(bytes);

        print_result("Raw deallocation", deallocation);

        save_csv(
            csv, elements,
            "RawDeallocation",
            deallocation
        );

        // Actual Tensor construction

        BenchmarkResult construction =
            benchmark_tensor_construction(elements);

        print_result("Tensor construction", construction);

        save_csv(
            csv, elements,
            "TensorConstruction",
            construction
        );

        // Actual Tensor destruction

        BenchmarkResult destruction =
            benchmark_tensor_destruction(elements);

        print_result("Tensor destruction", destruction);

        save_csv(
            csv, elements,
            "TensorDestruction",
            destruction
        );
    }

    csv.close();

    std::cout
        << "\n============================================\n";

    std::cout
        << "Memory benchmark completed successfully.\n";

    std::cout
        << "CSV saved: memory_benchmark.csv\n";

    std::cout
        << "Checksum: "
        << std::fixed << std::setprecision(6)
        << benchmark_sink << '\n';

    return 0;
}
