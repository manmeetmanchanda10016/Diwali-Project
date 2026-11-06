
#include "minitensor/Tensor.h"

#include <algorithm>
#include <chrono>
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
    std::vector<double> times;
    times.reserve(MEASUREMENTS);

    for (int i = 0; i < WARMUPS; ++i)
        function();

    for (int i = 0; i < MEASUREMENTS; ++i)
        times.push_back(function());

    std::sort(times.begin(), times.end());

    double total = 0.0;

    for (double time : times)
        total += time;

    const double average = total / MEASUREMENTS;

    const double median =
        (times[MEASUREMENTS / 2 - 1] +
         times[MEASUREMENTS / 2]) / 2.0;

    return {average, median};
}

double elapsed_ms(
    Clock::time_point start,
    Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
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
    std::size_t batch_size,
    const BenchmarkResult& result)
{
    const double average_per_operation_us =
        result.average_ms * 1000.0 / batch_size;

    const double median_per_operation_us =
        result.median_ms * 1000.0 / batch_size;

    csv << elements << ','
        << elements * sizeof(double) << ','
        << operation << ','
        << batch_size << ','
        << WARMUPS << ','
        << MEASUREMENTS << ','
        << std::fixed << std::setprecision(8)
        << result.average_ms << ','
        << result.median_ms << ','
        << average_per_operation_us << ','
        << median_per_operation_us << '\n';
}

BenchmarkResult benchmark_allocation(std::size_t bytes)
{
    std::vector<void*> pointers(BATCH_SIZE, nullptr);

    return measure([&]()
    {
        const auto start = Clock::now();

        for (std::size_t i = 0; i < BATCH_SIZE; ++i)
            pointers[i] = aligned_allocate(bytes);

        const auto end = Clock::now();

        const double time = elapsed_ms(start, end);

        for (void*& pointer : pointers)
        {
            aligned_deallocate(pointer);
            pointer = nullptr;
        }

        return time;
    });
}

BenchmarkResult benchmark_deallocation(std::size_t bytes)
{
    std::vector<void*> pointers(BATCH_SIZE, nullptr);

    return measure([&]()
    {
        // Prepare allocations before timing.
        for (void*& pointer : pointers)
            pointer = aligned_allocate(bytes);

        const auto start = Clock::now();

        for (void*& pointer : pointers)
        {
            aligned_deallocate(pointer);
            pointer = nullptr;
        }

        const auto end = Clock::now();

        return elapsed_ms(start, end);
    });
}

BenchmarkResult benchmark_initialization(std::size_t elements)
{
    const std::size_t bytes =
        elements * sizeof(double);

    double* buffer = static_cast<double*>(
        aligned_allocate(bytes)
    );

    const BenchmarkResult result = measure([&]()
    {
        const auto start = Clock::now();

        std::fill_n(buffer, elements, 1.25);

        const auto end = Clock::now();

        const double time = elapsed_ms(start, end);

        // Read every initialized element after timing.
        // This prevents initialization work from being optimized away.
        double checksum = 0.0;

        for (std::size_t i = 0; i < elements; ++i)
            checksum += buffer[i];

        benchmark_sink = benchmark_sink + checksum;

        return time;
    });

    aligned_deallocate(buffer);

    return result;
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

        const double time = elapsed_ms(start, end);

        benchmark_sink =
            benchmark_sink + tensor->data()[elements - 1];

        // Destruction happens outside the timed interval.
        tensor.reset();

        return time;
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
        // Construction happens before timing.
        tensor.emplace(dimensions, 1.25);

        benchmark_sink =
            benchmark_sink + tensor->data()[elements - 1];

        const auto start = Clock::now();

        tensor.reset();

        const auto end = Clock::now();

        return elapsed_ms(start, end);
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

    csv << "Elements,Bytes,Operation,BatchSize,"
        << "Warmups,Measurements,"
        << "AverageBatch_ms,MedianBatch_ms,"
        << "AveragePerOperation_us,MedianPerOperation_us\n";

    std::cout << "\nMINITENSOR MEMORY BENCHMARK\n";
    std::cout << "============================================\n";
    std::cout << "Warmups: " << WARMUPS
              << " | Measurements: " << MEASUREMENTS << '\n';
    std::cout << "Alignment: " << ALIGNMENT << " bytes\n";
    std::cout << "Raw allocation batch: " << BATCH_SIZE << '\n';

    for (const std::size_t elements : sizes)
    {
        const std::size_t bytes =
            elements * sizeof(double);

        std::cout << "\n============================================\n";
        std::cout << "ELEMENTS: " << elements << '\n';
        std::cout << "MEMORY: " << bytes << " bytes\n";
        std::cout << "============================================\n";

        const BenchmarkResult allocation =
            benchmark_allocation(bytes);

        print_result("Raw allocation", allocation);

        save_csv(
            csv, elements, "RawAllocation",
            BATCH_SIZE, allocation
        );

        const BenchmarkResult initialization =
            benchmark_initialization(elements);

        print_result("Initialization", initialization);

        save_csv(
            csv, elements, "Initialization",
            1, initialization
        );

        const BenchmarkResult deallocation =
            benchmark_deallocation(bytes);

        print_result("Raw deallocation", deallocation);

        save_csv(
            csv, elements, "RawDeallocation",
            BATCH_SIZE, deallocation
        );

        const BenchmarkResult construction =
            benchmark_tensor_construction(elements);

        print_result("Tensor construction", construction);

        save_csv(
            csv, elements, "TensorConstruction",
            1, construction
        );

        const BenchmarkResult destruction =
            benchmark_tensor_destruction(elements);

        print_result("Tensor destruction", destruction);

        save_csv(
            csv, elements, "TensorDestruction",
            1, destruction
        );
    }

    csv.close();

    std::cout << "\n============================================\n";
    std::cout << "Memory benchmark completed successfully.\n";
    std::cout << "CSV saved: memory_benchmark.csv\n";
    std::cout << "Checksum: " << benchmark_sink << '\n';

    return 0;
}
