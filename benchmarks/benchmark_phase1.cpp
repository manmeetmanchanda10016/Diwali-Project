
#include "minitensor/Tensor.h"

#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace minitensor;
using namespace std;
using namespace chrono;

int main() {

    vector<size_t> sizes = {
        1000000,
        10000000,
        50000000
    };

    cout << fixed << setprecision(3);

    cout << "\nPHASE 1 PERFORMANCE BENCHMARK\n";
    cout << "-------------------------------------\n";

    for (size_t n : sizes) {

        cout << "\nTensor size: " << n << " elements\n";

        auto start = high_resolution_clock::now();

        Tensor tensor({n}, 1.0);

        auto end = high_resolution_clock::now();

        double creationTime =
            duration<double, milli>(end - start).count();

        cout << "Creation time: "
             << creationTime << " ms\n";

        start = high_resolution_clock::now();

        tensor.fill(5.0);

        end = high_resolution_clock::now();

        double fillTime =
            duration<double, milli>(end - start).count();

        cout << "Fill time: "
             << fillTime << " ms\n";

        volatile double sum = 0.0;

        start = high_resolution_clock::now();

        for (size_t i = 0; i < tensor.size(); i++) {
            sum += tensor.data()[i];
        }

        end = high_resolution_clock::now();

        double traversalTime =
            duration<double, milli>(end - start).count();

        cout << "Traversal time: "
             << traversalTime << " ms\n";

        cout << "Checksum: " << sum << '\n';

        cout << "Memory used: "
             << (n * sizeof(double)) / (1024.0 * 1024.0)
             << " MB\n";
    }

    return 0;
}
