# MiniTensor — A Lightweight C++ Tensor Library

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Release](https://img.shields.io/badge/Release-v0.1.0-green.svg)](https://github.com/manmeetmanchanda10016/Diwali-Project/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MiniTensor is a lightweight, high-performance tensor library developed in C++17 as part of the **Diwali 2026 Challenge**.

The project implements multidimensional tensor storage, indexing, mathematical operations, broadcasting, matrix multiplication, reductions, memory management, and performance optimizations.

The library uses contiguous memory storage, cache-friendly algorithms, SIMD optimization, and OpenMP parallelism to improve execution performance.

**Repository:** https://github.com/manmeetmanchanda10016/Diwali-Project

---

## Project Objectives

- Implement a reusable multidimensional tensor library in C++.
- Provide efficient contiguous memory storage.
- Support multidimensional indexing and reshaping.
- Implement mathematical and matrix operations.
- Support broadcasting and reduction operations.
- Optimize execution using SIMD, blocking, and OpenMP.
- Provide testing, benchmarking, documentation, and CMake packaging.

---

## Features

### Tensor Management
- Multidimensional tensor creation.
- Contiguous memory allocation.
- 64-byte aligned memory storage.
- Shape and dimension management.
- Deep copy and move semantics.
- Tensor initialization and filling.

### Indexing and Reshaping
- Row-major memory layout.
- Multidimensional indexing using `at()`.
- Element access using `operator()`.
- Flat indexing using `flat_at()`.
- Tensor reshaping without copying the underlying elements.

### Mathematical Operations
- Elementwise addition.
- Elementwise subtraction.
- Elementwise multiplication.
- Parallel elementwise operations.
- Matrix multiplication.
- Parallel matrix multiplication.
- Blocked matrix multiplication.
- Parallel blocked matrix multiplication.

### Advanced Operations
- Tensor transpose.
- Broadcasting addition.
- Broadcasting subtraction.
- Broadcasting multiplication.
- Sum reduction.
- Mean reduction.
- Minimum reduction.
- Maximum reduction.

### Performance Optimization
- Cache-friendly blocked matrix multiplication.
- OpenMP parallelization.
- SIMD elementwise operations.
- OpenMP SIMD.
- Blocked transpose.
- Hybrid matrix multiplication dispatch.
- Native CPU optimization.
- Link-time optimization (LTO).

### Testing and Packaging
- Six test suites.
- Performance benchmarking.
- Interactive tensor application.
- CMake installation and export.
- Reusable `find_package(MiniTensor)` support.
- MIT License.

---

## Project Development — Seven Phases

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Tensor Representation and Memory Management | Completed |
| Phase 2 | Indexing and Reshaping | Completed |
| Phase 3 | Mathematical Operations | Completed |
| Phase 4 | Advanced Tensor Operations | Completed |
| Phase 5 | Performance Optimization | Completed |
| Phase 6 | Testing and Application | Completed |
| Phase 7 | Packaging, Documentation and Release | Completed |

---

## Project Structure

```text
MiniTensor_Phase1/
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
│   ├── benchmark_broadcast_v3.cpp
│   └── benchmark_remaining.cpp
│
├── examples/
│   ├── main.cpp
│   └── tensor_application.cpp
│
├── cmake/
│   └── MiniTensorConfig.cmake.in
│
├── CMakeLists.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Requirements

- C++17 compatible compiler.
- CMake 3.16 or later.
- GCC or Clang.
- OpenMP support.
- Git.

The project was developed and benchmarked using:

| Component | Configuration |
|---|---|
| Processor | AMD Ryzen AI 7 350 |
| Physical Cores | 8 |
| Logical Processors | 16 |
| Compiler | GCC 16.1.0 |
| Language Standard | C++17 |
| Build Configuration | Release |
| Parallelization | OpenMP |

---

## Build Instructions

Clone the repository:

```bash
git clone https://github.com/manmeetmanchanda10016/Diwali-Project.git

cd Diwali-Project
```

Configure the Release build:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
```

Build the project:

```bash
cmake --build build-release -j
```

For the tested MSYS2 MinGW environment with native CPU optimization and LTO:

```powershell
cmake -S . -B build-release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DMINITENSOR_ENABLE_NATIVE=ON -DMINITENSOR_ENABLE_LTO=ON

cmake --build build-release -j
```

---

## Running Tests

Run all six test suites using CTest:

```bash
ctest --test-dir build-release --output-on-failure
```

The test suites cover tensor representation, indexing, mathematical operations, advanced operations, optimization, and application functionality.

---

## Usage Example

```cpp
#include "minitensor/Tensor.h"
#include "minitensor/Ops.h"

#include <iostream>

int main() {

    minitensor::Tensor A({2, 2}, 3.0);
    minitensor::Tensor B({2, 2}, 5.0);

    minitensor::Tensor C = minitensor::add(A, B);

    for (std::size_t i = 0; i < C.size(); ++i) {
        std::cout << C.flat_at(i) << " ";
    }

    return 0;
}
```

Output:

```text
8 8 8 8
```

---

## Interactive Tensor Application

MiniTensor also includes an interactive console application demonstrating its operations.

The application provides the following menu:

1. Tensor Management
2. Arithmetic Operations
3. Broadcasting
4. Matrix Multiplication
5. Transpose
6. Reduction Operations
0. Exit

It allows users to experiment with tensor creation, indexing, mathematical operations, and other supported functionality.

---

# Performance Benchmarks

Performance was evaluated using Release builds on an AMD Ryzen AI 7 350 processor.

The benchmark results below are measurements from the local development environment. Execution times may vary depending on hardware, compiler, build configuration, and workload.

## 1. Matrix Multiplication

| Matrix Size | Sequential | Best Parallel/Hybrid | Speedup |
|---|---:|---:|---:|
| 64 × 64 | 0.023 ms | 0.026 ms | — |
| 128 × 128 | 0.177 ms | 0.068 ms | 2.60× |
| 256 × 256 | 1.444 ms | 0.430 ms | 3.36× |
| 512 × 512 | 12.494 ms | 3.476 ms | 3.59× |
| 1024 × 1024 | 95.876 ms | 25.204 ms | 3.80× |

The results demonstrate improved performance from parallel and hybrid matrix multiplication for larger matrices.

## 2. Transpose Optimization

Comparison between standard transpose and blocked transpose using a block size of 16.

| Matrix Size | Standard Transpose | Blocked Transpose | Speedup |
|---|---:|---:|---:|
| 256 × 256 | 0.518 ms | 0.143 ms | 3.62× |
| 512 × 512 | 2.269 ms | 0.985 ms | 2.30× |
| 1024 × 1024 | 9.736 ms | 4.171 ms | 2.33× |
| 2048 × 2048 | 40.869 ms | 15.870 ms | 2.58× |

Blocked transpose improves cache locality and reduces execution time.

## 3. Elementwise Operations

Performance comparison at 16 million elements.

| Operation | Scalar | SIMD | Parallel |
|---|---:|---:|---:|
| Addition | 100.311 ms | 42.135 ms | 40.131 ms (2 threads) |
| Subtraction | 100.462 ms | 33.138 ms | 40.265 ms (2 threads) |
| Multiplication | 101.531 ms | 41.696 ms | 40.206 ms (4 threads) |

SIMD and parallel execution improve the performance of large elementwise operations.

## 4. Broadcasting

Broadcasting was evaluated using optimized implementations and a fair SIMD reference benchmark.

The V3 benchmark showed performance ratios approximately between **0.88× and 1.03×** relative to the direct SIMD reference, depending on the operation and workload.

These results indicate that the optimized broadcasting implementation performs close to the direct reference for the tested workloads.

## 5. Tensor Construction and Allocation

| Tensor Size | Execution Time |
|---|---:|
| 1,024 elements | 0.0005 ms |
| 100,000 elements | 0.0053 ms |
| 1,000,000 elements | 1.9329 ms |

## 6. Indexing Performance

Tested with a 512 × 512 tensor, containing 262,144 elements.

| Indexing Method | Number of Accesses | Execution Time |
|---|---:|---:|
| Flat indexing (`flat_at()`) | 262,144 | 0.5793 ms |
| Multidimensional indexing (`at()`) | 262,144 | 8.8345 ms |

Flat indexing has lower access overhead because it directly accesses elements using their linear index.

## 7. Reshape Performance

| Operation | Tensor Size | Execution Time |
|---|---:|---:|
| Two reshape operations | 262,144 elements | 0.0002 ms |

The benchmark performs two shape changes while preserving the underlying tensor elements.

## 8. Fill Operation

| Tensor Size | Execution Time |
|---|---:|
| 1,024 elements | 0.0001 ms |
| 100,000 elements | 0.0081 ms |
| 1,000,000 elements | 0.0955 ms |

## 9. Reduction Operations

Tested with one million elements.

| Operation | Execution Time |
|---|---:|
| Sum | 0.1102 ms |
| Mean | 0.1104 ms |
| Minimum | 0.5575 ms |
| Maximum | 0.5595 ms |

All operation-level timings represent median measurements from repeated benchmark runs.

---

# Functional Testing Results

All six test suites were executed successfully using CTest.

| Test Suite | Execution Time | Result |
|---|---:|---|
| Phase 1 | 0.02 s | Passed |
| Phase 2 | 0.02 s | Passed |
| Phase 3 | 0.08 s | Passed |
| Phase 4 | 0.01 s | Passed |
| Phase 5 | 0.02 s | Passed |
| Phase 6 | 0.02 s | Passed |
| **Total** | **0.19 s** | **6/6 Passed** |

**Test summary:**
- Total test suites: 6
- Passed: 6
- Failed: 0
- Total CTest execution time: 0.19 seconds

These are functional test-suite timings and are reported separately from operation-level performance benchmarks.

---

## Installation and CMake Package Usage

MiniTensor supports installation and CMake package export.

Install the library:

```bash
cmake --install build-release --prefix install
```

A separate CMake project can discover the installed package using:

```cmake
find_package(MiniTensor REQUIRED)

add_executable(my_application main.cpp)

target_link_libraries(my_application
    PRIVATE MiniTensor::minitensor
)
```

The installed package was tested using a separate consumer project.

---

## Release

**Current Release: v0.1.0 — Initial Release**

The release includes the completed seven development phases, source code, tests, benchmarks, documentation, and CMake packaging.

Release page:

https://github.com/manmeetmanchanda10016/Diwali-Project/releases/tag/v0.1.0

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

Developed as part of the **Diwali 2026 Challenge**.

**Team Leader:** MANMEET SINGH MANCHANDA

**Project:** MiniTensor — Lightweight C++ Tensor Library

**GitHub Repository:** https://github.com/manmeetmanchanda10016/Diwali-Project

---

**MiniTensor v0.1.0 — All Seven Development Phases Completed.**