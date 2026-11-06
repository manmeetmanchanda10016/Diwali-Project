
#include "minitensor/Tensor.h"
#include "minitensor/Storage.h"
#include "minitensor/Shape.h"

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
constexpr double INITIAL_VALUE = 1.25;

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

double consume_checksum(
    const double* buffer,
    std::size_t elements)
{
    double checksum = 0.0;

    for (std::size_t i = 0; i < elements; ++i)
        checksum += buffer[i];

    benchmark_sink = benchmark_sink + checksum;

    return checksum;
}

void print_result(
    const std::string& name,
    const BenchmarkResult& result)
{
    std::cout << std::left << std::setw(34)
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
        << 1 << ','
        << WARMUPS << ','
        << MEASUREMENTS << ','
        << std::fixed << std::setprecision(8)
        << result.average_ms << ','
        << result.median_ms << ','
        << result.average_ms * 1000.0 << ','
        << result.median_ms * 1000.0 << '\n';
}

BenchmarkResult benchmark_raw_allocation(std::size_t bytes)
{
    return measure([&]()
    {
        const auto start = Clock::now();

        void* pointer = aligned_allocate(bytes);

        const auto end = Clock::now();

        const double time = elapsed_ms(start, end);

        aligned_deallocate(pointer);

        return time;
    });
}

BenchmarkResult benchmark_raw_allocation_initialization(
    std::size_t elements)
{
    const std::size_t bytes = elements * sizeof(double);

    return measure([&]()
    {
        double* buffer = static_cast<double*>(
            aligned_allocate(bytes)
        );

        const auto start = Clock::now();

        std::fill_n(buffer, elements, INITIAL_VALUE);

        const auto end = Clock::now();

        const double time = elapsed_ms(start, end);

        consume_checksum(buffer, elements);

        aligned_deallocate(buffer);

        return time;
    });
}

BenchmarkResult benchmark_existing_buffer_initialization(
    std::size_t elements)
{
    const std::size_t bytes = elements * sizeof(double);

    double* buffer = static_cast<double*>(
        aligned_allocate(bytes)
    );

    std::fill_n(buffer, elements, 0.0);

    const BenchmarkResult result = measure([&]()
    {
        const auto start = Clock::now();

        std::fill_n(buffer, elements, INITIAL_VALUE);

        const auto end = Clock::now();

        const double time = elapsed_ms(start, end);

        consume_checksum(buffer, elements);

        return time;
    });

    aligned_deallocate(buffer);

    return result;
}

BenchmarkResult benchmark_raw_deallocation(
    std::size_t elements)
{
    const std::size_t bytes = elements * sizeof(double);

    return measure([&]()
    {
        // Prepare and initialize memory before timing.
        double* buffer = static_cast<double*>(
            aligned_allocate(bytes)
        );

        std::fill_n(buffer, elements, INITIAL_VALUE);

        consume_checksum(buffer, elements);

        // Only deallocation is measured.
        const auto start = Clock::now();

        aligned_deallocate(buffer);

        const auto end = Clock::now();

        return elapsed_ms(start, end);
    });
}

BenchmarkResult benchmark_shape_construction(
    std::size_t elements)
{
    const std::vector<std::size_t> dimensions = {
        elements
    };

    std::optional<Shape> shape;

    return measure([&]()
    {
        const auto start = Clock::now();

        shape.emplace(dimensions);

        const auto end = Clock::now();

        const double time = elapsed_ms(start, end);

        shape.reset();

        return time;
    });
}

BenchmarkResult benchmark_storage_construction(
    std::size_t elements)
{
    std::optional<Storage> storage;

    return measure([&]()
    {
        const auto start = Clock::now();

        storage.emplace(elements, INITIAL_VALUE);

        const auto end = Clock::now();

        const double time = elapsed_ms(start, end);

        consume_checksum(storage->data(), elements);

        storage.reset();

        return time;
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

        tensor.emplace(dimensions, INITIAL_VALUE);

        const auto end = Clock::now();

        const double time = elapsed_ms(start, end);

        consume_checksum(tensor->data(), elements);

        tensor.reset();

        return time;
    });
}

BenchmarkResult benchmark_storage_destruction(
    std::size_t elements)
{
    std::optional<Storage> storage;

    return measure([&]()
    {
        // Construction and initialization happen before timing.
        storage.emplace(elements, INITIAL_VALUE);

        consume_checksum(storage->data(), elements);

        const auto start = Clock::now();

        storage.reset();

        const auto end = Clock::now();

        return elapsed_ms(start, end);
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
        // Construction and initialization happen before timing.
        tensor.emplace(dimensions, INITIAL_VALUE);

        consume_checksum(tensor->data(), elements);

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
        262144,
        1048576
    };

    std::ofstream csv("memory_benchmark.csv");

    if (!csv.is_open())
    {
        std::cerr << "Could not create CSV file.\n";
        return 1;
    }

    csv << "Elements,Bytes,Operation,BatchSize,"
        << "Warmups,Measurements,"
        << "Average_ms,Median_ms,"
        << "Average_us,Median_us\n";

    std::cout << "\nMINITENSOR MEMORY BENCHMARK\n";
    std::cout << "============================================\n";
    std::cout << "Warmups: " << WARMUPS
              << " | Measurements: " << MEASUREMENTS << '\n';
    std::cout << "Alignment: " << ALIGNMENT << " bytes\n";

    for (const std::size_t elements : sizes)
    {
        const std::size_t bytes =
            elements * sizeof(double);

        std::cout << "\n============================================\n";
        std::cout << "ELEMENTS: " << elements << '\n';
        std::cout << "MEMORY: " << bytes << " bytes\n";
        std::cout << "============================================\n";

        const BenchmarkResult raw_allocation =
            benchmark_raw_allocation(bytes);

        print_result("Raw allocation", raw_allocation);
        save_csv(csv, elements, "RawAllocation", raw_allocation);

        const BenchmarkResult raw_init =
            benchmark_raw_allocation_initialization(elements);

        print_result("Raw allocation + initialization", raw_init);
        save_csv(csv, elements, "RawAllocationInitialization", raw_init);

        const BenchmarkResult existing_init =
            benchmark_existing_buffer_initialization(elements);

        print_result("Existing buffer initialization", existing_init);
        save_csv(csv, elements, "ExistingBufferInitialization", existing_init);

        const BenchmarkResult raw_deallocation =
            benchmark_raw_deallocation(elements);

        print_result("Raw deallocation", raw_deallocation);
        save_csv(csv, elements, "RawDeallocation", raw_deallocation);

        const BenchmarkResult shape =
            benchmark_shape_construction(elements);

        print_result("Shape construction", shape);
        save_csv(csv, elements, "ShapeConstruction", shape);

        const BenchmarkResult storage =
            benchmark_storage_construction(elements);

        print_result("Storage construction", storage);
        save_csv(csv, elements, "StorageConstruction", storage);

        const BenchmarkResult tensor =
            benchmark_tensor_construction(elements);

        print_result("Tensor construction", tensor);
        save_csv(csv, elements, "TensorConstruction", tensor);

        const BenchmarkResult storage_destruction =
            benchmark_storage_destruction(elements);

        print_result("Storage destruction", storage_destruction);
        save_csv(csv, elements, "StorageDestruction", storage_destruction);

        const BenchmarkResult tensor_destruction =
            benchmark_tensor_destruction(elements);

        print_result("Tensor destruction", tensor_destruction);
        save_csv(csv, elements, "TensorDestruction", tensor_destruction);
    }

    csv.close();

    std::cout << "\n============================================\n";
    std::cout << "Memory benchmark completed successfully.\n";
    std::cout << "CSV saved: memory_benchmark.csv\n";
    std::cout << "Checksum: " << benchmark_sink << '\n';

    return 0;
}
