
#include "minitensor/Tensor.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace minitensor;
using namespace std;
using namespace chrono;

constexpr int WARMUP = 3;
constexpr int ITERATIONS = 7;

double global_checksum = 0.0;

template <typename Function>
double benchmark(Function operation)
{
    for (int i = 0; i < WARMUP; i++)
    {
        global_checksum += operation();
    }

    vector<double> timings;
    timings.reserve(ITERATIONS);

    for (int i = 0; i < ITERATIONS; i++)
    {
        auto start = steady_clock::now();

        double result = operation();

        auto end = steady_clock::now();

        double elapsed =
            duration<double, milli>(end - start).count();

        timings.push_back(elapsed);

        // Checksum update is outside the measured interval.
        global_checksum += result;
    }

    sort(timings.begin(), timings.end());

    return timings[ITERATIONS / 2];
}

int main()
{
    vector<size_t> sizes = {
        1000000,
        10000000,
        50000000
    };

    cout << fixed << setprecision(3);

    cout << "\n====================================\n";
    cout << "       MINITENSOR BENCHMARK V3\n";
    cout << "====================================\n";

    cout << "Warmup iterations: " << WARMUP << '\n';
    cout << "Measured iterations: " << ITERATIONS << '\n';

    for (size_t n : sizes)
    {
        cout << "\n------------------------------------\n";
        cout << "Tensor elements: " << n << '\n';
        cout << "------------------------------------\n";

        Tensor tensor({n}, 1.0);

        // 1. Construction + initialization + destruction
        double creationTime = benchmark([&]()
        {
            Tensor temp({n}, 1.0);

            return temp.data()[n - 1];
        });

        // 2. Fill operation
        double fillTime = benchmark([&]()
        {
            tensor.fill(5.0);

            return tensor.data()[n - 1];
        });

        // 3. Sequential traversal
        double traversalTime = benchmark([&]()
        {
            double sum = 0.0;

            const double* ptr = tensor.data();

            for (size_t i = 0; i < n; i++)
            {
                sum += ptr[i];
            }

            return sum;
        });

        // 4. Copy construction + allocation + destruction
        double copyTime = benchmark([&]()
        {
            Tensor copy = tensor;

            return copy.data()[n - 1];
        });

        // 5. Memory usage
        double memoryMiB =
            (static_cast<double>(n) * sizeof(double)) /
            (1024.0 * 1024.0);

        cout << "Creation time:   "
             << creationTime << " ms\n";

        cout << "Fill time:       "
             << fillTime << " ms\n";

        cout << "Traversal time:  "
             << traversalTime << " ms\n";

        cout << "Copy time:       "
             << copyTime << " ms\n";

        cout << "Memory usage:    "
             << memoryMiB << " MiB\n";
    }

    cout << "\nGlobal checksum: "
         << global_checksum << '\n';

    cout << "\nBenchmark completed!\n";

    return 0;
}
