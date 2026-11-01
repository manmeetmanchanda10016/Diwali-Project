# MiniTensor

**A lightweight, high-performance multidimensional tensor library written in C++17.**

MiniTensor is a custom tensor library built from scratch in C++. It provides multidimensional tensor storage, indexing, mathematical operations, broadcasting, matrix multiplication, reductions, and performance optimizations using SIMD and OpenMP.

The project focuses on efficient memory management, contiguous storage, optimized numerical operations, and a clean C++ interface.

---

## Features

### 1. Tensor Representation and Memory Management
- Multidimensional tensor creation.
- Contiguous memory allocation.
- Custom 64-byte aligned storage.
- Deep copy and move semantics.
- Efficient memory management.
- Shape and element-count tracking.
- Dimension overflow detection.

### 2. Multidimensional Indexing
- Row-major memory layout.
- Multidimensional indexing.
- Element access using `at()` and `operator()`.
- Flat indexing using `flat_at()`.
- Tensor reshaping.

### 3. Mathematical Operations
- Elementwise addition.
- Elementwise subtraction.
- Elementwise multiplication.
- Matrix multiplication.
- Sequential and OpenMP-parallel operations.

### 4. Advanced Tensor Operations
- Matrix transpose.
- Broadcasting.
- Sum reduction.
- Mean reduction.
- Minimum reduction.
- Maximum reduction.

### 5. Performance Optimization
- SIMD vectorization.
- OpenMP parallel computing.
- Blocked matrix multiplication.
- Parallel blocked matrix multiplication.
- Blocked matrix transpose.
- Optimized broadcasting paths.
- Native CPU optimization.
- Link Time Optimization (LTO).

### 6. Testing and Application
- Six test suites covering Phases 1–6.
- Tensor Playground interactive application.
- Performance benchmarks for different operations.
- Correctness testing across tensor shapes and operations.

---

## Project Structure

```text
MiniTensor/
│
├── include/
│   └── minitensor/
│       ├── Shape.h
│       ├── Storage.h
│       ├── Tensor.h
│       └── Ops.h
│
├── src/
│   ├── Shape.cpp
│   ├── Storage.cpp
│   ├── Tensor.cpp
│   └── Ops.cpp
│
├── tests/
│   ├── test_phase1.cpp
│   ├── test_phase2.cpp
│   ├── test_phase3.cpp
│   ├── test_phase4.cpp
│   ├── test_phase5.cpp
│   └── test_phase6.cpp
│
├── benchmarks/
│   ├── benchmark_phase1.cpp
│   ├── benchmark_v2.cpp
│   ├── benchmark_matmul.cpp
│   ├── benchmark_elementwise.cpp
│   ├── benchmark_transpose.cpp
│   ├── benchmark_broadcast.cpp
│   ├── benchmark_broadcast_v2.cpp
│   └── benchmark_broadcast_v3.cpp
│
├── examples/
│   ├── main.cpp
│   └── tensor_application.cpp
│
├── cmake/
│   └── MiniTensorConfig.cmake.in
│
├── CMakeLists.txt
└── README.md
```

---

## Requirements

- C++17 compatible compiler.
- CMake 3.16 or newer.
- GCC or Clang (MSVC may also be supported).
- OpenMP development support.

Tested development environment:

- Windows 11
- MSYS2 GCC 16.1.0
- CMake
- VS Code
- AMD Ryzen AI 7 350 processor

---

## Building the Project

### Windows (MinGW)

Clone the repository:

```bash
git clone https://github.com/manmeetmanchanda10016/Diwali-Project.git
cd MiniTensor_Phase1
```

Configure the Release build:

```powershell
cmake -S . -B build-release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DMINITENSOR_ENABLE_NATIVE=ON -DMINITENSOR_ENABLE_LTO=ON
```

Build:

```powershell
cmake --build build-release -j
```

### Linux (GCC)

Install the required build tools and OpenMP support.

```bash
sudo apt update
sudo apt install build-essential cmake libgomp1
```

Configure:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

Build:

```bash
cmake --build build -j
```

---

## Running Tests

MiniTensor uses CTest for automated testing.

```bash
ctest --test-dir build-release --output-on-failure
```

The six test suites cover:

| Test suite | Coverage |
|---|---|
| Phase 1 | Tensor representation and storage |
| Phase 2 | Indexing and reshaping |
| Phase 3 | Mathematical operations |
| Phase 4 | Broadcasting and reductions |
| Phase 5 | Performance-oriented operations |
| Phase 6 | Edge cases and application correctness |

---

## Running the Examples

Run the basic example:

```powershell
.\build-release\basic_example.exe
```

Run the interactive Tensor Playground:

```powershell
.\build-release\tensor_application.exe
```

The Tensor Playground provides menu-driven access to:

- Tensor creation and management.
- Element access and modification.
- Arithmetic operations.
- Broadcasting.
- Matrix multiplication.
- Transpose.
- Tensor reductions.

Run interactive applications in a terminal that supports standard input.

---

## Using MiniTensor

Include the required headers:

```cpp
#include "minitensor/Tensor.h"
#include "minitensor/Ops.h"

#include <iostream>

using namespace minitensor;

int main()
{
    Tensor a({2, 3});
    Tensor b({2, 3});

    a.fill(5.0);
    b.fill(2.0);

    Tensor result = add(a, b);

    std::cout << "Result:\n";

    for (std::size_t i = 0; i < result.size(); i++)
    {
        std::cout << result.flat_at(i) << " ";
    }

    return 0;
}
```

Expected output:

```text
Result:
7 7 7 7 7 7
```

### Matrix Multiplication

```cpp
Tensor a({2, 3});
Tensor b({3, 2});

Tensor result = matmul(a, b);
```

### Broadcasting

```cpp
Tensor a({2, 3});
Tensor b({3});

Tensor result = add_broadcast(a, b);
```

### Transpose

```cpp
Tensor a({2, 3});

Tensor result = transpose(a);
```

### Reductions

```cpp
double total = sum(a);
double average = mean(a);
double minimum = min(a);
double maximum = max(a);
```

---

## Installation

MiniTensor supports CMake installation.

After configuring and building:

```powershell
cmake --install build-release --prefix install
```

The installation directory contains:

```text
install/
├── include/
│   └── minitensor/
│       ├── Shape.h
│       ├── Storage.h
│       ├── Tensor.h
│       └── Ops.h
│
└── lib/
    ├── libminitensor.a
    └── cmake/
        └── MiniTensor/
```

The exported CMake package provides the target:

```cmake
MiniTensor::minitensor
```

---

## Performance Optimization

MiniTensor incorporates several optimization techniques:

| Optimization | Purpose |
|---|---|
| 64-byte aligned memory | Improve memory alignment |
| SIMD | Process multiple elements per instruction |
| OpenMP | Parallelize computational workloads |
| Blocked matrix multiplication | Improve cache utilization |
| Blocked transpose | Improve memory access locality |
| Native CPU optimization | Enable processor-specific instructions |
| Link Time Optimization | Optimize across translation units |

### Benchmark Highlights

Measurements from the development system (AMD Ryzen AI 7 350):

| Operation | Input size | Observed result |
|---|---:|---|
| Matrix multiplication | 128 × 128 | Up to 2.63× speedup |
| Matrix multiplication | 256 × 256 | Up to 3.36× speedup |
| Matrix multiplication | 1024 × 1024 | Up to 3.80× speedup |
| Transpose | 256 × 256 | Up to 3.62× speedup |
| Transpose | 1024 × 1024 | Up to 2.33× speedup |
| Transpose | 2048 × 2048 | Up to 2.58× speedup |

Speedups compare the optimized implementation against its corresponding sequential baseline in the development benchmarks. Results depend on hardware, compiler, and workload.

---

## Development Phases

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Tensor representation and memory management | Completed |
| Phase 2 | Multidimensional indexing and memory layout | Completed |
| Phase 3 | Mathematical operations | Completed |
| Phase 4 | Advanced tensor operations | Completed |
| Phase 5 | Performance optimization and parallel computing | Completed |
| Phase 6 | Testing and application development | Completed |
| Phase 7 | Packaging, documentation and public release | In progress |

---

## Future Improvements

Potential future enhancements include:

- Additional tensor operations.
- More advanced SIMD optimizations.
- Improved parallel scheduling.
- Expanded benchmarking across different processors.
- Additional examples and documentation.
- Linux and cross-platform build verification.

---

## Contributing

Contributions, suggestions, and performance improvements are welcome.

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Run the test suite.
5. Submit a pull request.

---

## License

A license has not yet been selected for this project. Check the repository for licensing information before reusing or distributing the code.

---

## Author

**MiniTensor — A C++17 Multidimensional Tensor Library**

GitHub Repository:

https://github.com/manmeetmanchanda10016/Diwali-Project
