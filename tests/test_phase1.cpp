#include "minitensor/Tensor.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using minitensor::Tensor;

int main()
{
    // Test 1: Tensor creation
    Tensor tensor({2, 3}, 5.0);

    assert(tensor.rank() == 2);
    assert(tensor.size() == 6);

    // Test 2: Shape verification
    assert(tensor.shape().dimensions() == std::vector<std::size_t>({2, 3}));

    // Test 3: Fill operation
    tensor.fill(10.0);

    for (std::size_t i = 0; i < tensor.size(); i++)
    {
        assert(tensor.data()[i] == 10.0);
    }

    // Test 4: Data modification
    tensor.data()[0] = 99.0;
    assert(tensor.data()[0] == 99.0);

    // Test 5: Copy constructor
    Tensor copiedTensor = tensor;

    assert(copiedTensor.size() == tensor.size());
    assert(copiedTensor.data() != tensor.data());
    assert(copiedTensor.data()[0] == 99.0);

    // Test 6: Copy independence
    copiedTensor.data()[0] = 50.0;
    assert(tensor.data()[0] == 99.0);

    // Test 7: Move constructor
    Tensor movedTensor = std::move(copiedTensor);

    assert(movedTensor.size() == 6);
    assert(movedTensor.data()[0] == 50.0);

    // Test 8: Empty tensor
    Tensor emptyTensor({0, 3});

    assert(emptyTensor.size() == 0);

    // Test 9: Scalar tensor
    Tensor scalarTensor({});

    assert(scalarTensor.rank() == 0);
    assert(scalarTensor.size() == 1);

    // Test 10: Overflow detection
    bool overflowDetected = false;

    try
    {
        Tensor largeTensor({
            std::numeric_limits<std::size_t>::max(),
            2
        });
    }
    catch (const std::overflow_error&)
    {
        overflowDetected = true;
    }

    assert(overflowDetected);

    // Test 11: 64-byte memory alignment
    Tensor alignedTensor({1000}, 1.0);

    assert(
        reinterpret_cast<std::uintptr_t>(alignedTensor.data()) % 64 == 0
    );

    // Test 12: Contiguous memory
    assert(
        reinterpret_cast<std::uintptr_t>(alignedTensor.data() + 1) -
        reinterpret_cast<std::uintptr_t>(alignedTensor.data()) == sizeof(double)
    );

    // Test 13: Copy retains alignment and independent storage
    Tensor alignedCopy = alignedTensor;

    assert(
        reinterpret_cast<std::uintptr_t>(alignedCopy.data()) % 64 == 0
    );

    assert(alignedCopy.data() != alignedTensor.data());

    // Test 14: Move retains alignment
    Tensor alignedMoved = std::move(alignedCopy);

    assert(
        reinterpret_cast<std::uintptr_t>(alignedMoved.data()) % 64 == 0
    );

    std::cout << "All Phase 1 tests passed!\n";

    return 0;
}