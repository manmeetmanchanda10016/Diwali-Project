# Mini Tensor Library (C++)

A lightweight C++ tensor library developed as part of the **Diwali 2026 Challenge**. The project focuses on efficient multidimensional tensor storage, mathematical operations, memory management, and performance optimization using OpenMP.

## Project Overview

The objective is to build a simplified tensor library from scratch in C++, inspired by the fundamental functionality of libraries such as NumPy and PyTorch.

The library supports multidimensional tensors, contiguous memory storage, tensor indexing, arithmetic operations, broadcasting, matrix multiplication, and parallel computation.

## Features Implemented

### Phase 1 – Tensor Representation and Memory Management
- Multidimensional tensor creation.
- Contiguous memory allocation.
- 64-byte aligned memory storage.
- Deep copy and move semantics.
- Memory management and element initialization.
- Scalar and empty tensor support.

### Phase 2 – Multidimensional Indexing and Memory Layout
- Row-major memory layout.
- Stride calculation.
- Multidimensional element access.
- Flat indexing.
- Tensor reshaping.

### Phase 3 – Mathematical Operations
- Element-wise addition.
- Element-wise subtraction.
- Element-wise multiplication.
- Matrix multiplication.
- OpenMP-based parallel matrix multiplication.
- Hybrid sequential and parallel execution.

### Phase 4 – Advanced Tensor Operations
- Tensor transpose.
- Broadcasting addition.
- Broadcasting subtraction.
- Broadcasting multiplication.
- Sum reduction.
- Mean reduction.
- Minimum reduction.
- Maximum reduction.

### Phase 5 – Performance Optimization and Parallel Computing
Several matrix multiplication implementations were developed and benchmarked:

1. **Normal Sequential:** Uses an optimized `i-k-j` loop order.
2. **OpenMP:** Parallelizes matrix multiplication across rows.
3. **Hybrid:** Selects sequential or parallel execution based on workload.
4. **Blocked Matrix Multiplication:** Divides matrices into smaller blocks to improve cache utilization.
5. **Parallel-Blocked Matrix Multiplication:** Combines cache blocking with OpenMP parallelization.

## Performance Benchmarks

Matrix multiplication implementations were compared using different matrix sizes and block sizes.

### Latest Benchmark Results

| Matrix Size | Normal Sequential | OpenMP | Hybrid | Parallel-Blocked (Best) |
|---|---:|---:|---:|---:|
| 64 × 64 | 0.040 ms | 0.079 ms | 0.040 ms | 0.097 ms |
| 128 × 128 | 0.502 ms | 0.264 ms | 0.669 ms | 0.303 ms |
| 256 × 256 | 3.491 ms | 0.947 ms | 0.997 ms | 0.931 ms |
| 512 × 512 | 26.154 ms | 6.468 ms | 6.031 ms | 6.658 ms |
| 1024 × 1024 | 211.100 ms | 40.512 ms | 45.874 ms | 37.092 ms |

### Best Performance Achieved

For a 1024 × 1024 matrix:

| Metric | Result |
|---|---:|
| Normal sequential runtime | 211.100 ms |
| Parallel-blocked runtime | 37.092 ms |
| Execution time saved | 174.008 ms |
| Runtime reduction | 82.43% |
| Speedup | 5.691× |

### Block Size Experiments

Different block sizes were tested to identify their impact on performance.

| Matrix Size | Best Sequential Block Size | Best Parallel Block Size |
|---|---:|---:|
| 64 × 64 | 64 | 16 |
| 128 × 128 | 128 | 32 |
| 256 × 256 | 64 | 32 |
| 512 × 512 | 64 | 32 |
| 1024 × 1024 | 64 | 64 |

**Observation:** Parallel-blocked multiplication achieved the best measured runtime for the 256 × 256 and 1024 × 1024 matrices. Hybrid multiplication performed best at 512 × 512. The optimal block size varied with matrix dimensions.

*Benchmark results are machine-dependent and may vary with CPU, compiler, OpenMP thread count, and system load.*

All benchmark implementations produced matching results within the configured numerical tolerance. The global checksum was `30621314642.990`.

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
│   └── test_phase5.cpp
│
├── benchmarks/
│   ├── benchmark_phase1.cpp
│   ├── benchmark_v2.cpp
│   └── benchmark_matmul.cpp
│
├── examples/
│   └── main.cpp
│
├── CMakeLists.txt
└── README.md
```

## Technologies Used

- **Language:** C++
- **Build System:** CMake
- **Compiler:** GCC
- **Parallel Computing:** OpenMP
- **Version Control:** Git and GitHub
- **Development Environment:** Visual Studio Code

## Build Instructions

### Prerequisites

- C++17 compatible compiler.
- CMake.
- OpenMP support.
- Git.

### 1. Clone the Repository

```bash
git clone https://github.com/manmeetmanchanda10016/Diwali-Project.git
cd Diwali-Project
```

### 2. Configure the Project

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
```

### 3. Build the Library

```bash
cmake --build build-release -j
```

### 4. Run Tests

```bash
ctest --test-dir build-release --output-on-failure
```

### 5. Run the Matrix Multiplication Benchmark

```bash
./build-release/benchmark_matmul
```

On Windows PowerShell:

```powershell
.\build-release\benchmark_matmul.exe
```

## Testing

Five test suites have been implemented:

| Test Suite | Coverage |
|---|---|
| Phase 1 | Tensor creation and memory management |
| Phase 2 | Indexing, strides and reshaping |
| Phase 3 | Mathematical operations and matrix multiplication |
| Phase 4 | Broadcasting, transpose and reductions |
| Phase 5 | Matrix multiplication correctness and parallel consistency |

**Latest validation:** All 5 test suites passed successfully.

## Development Roadmap

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Tensor representation and memory management | Completed |
| Phase 2 | Multidimensional indexing and memory layout | Completed |
| Phase 3 | Mathematical operations | Completed |
| Phase 4 | Advanced tensor operations | Completed |
| Phase 5 | Performance optimization and parallel computing | Completed |
| Phase 6 | Testing and application development | Upcoming |
| Phase 7 | Packaging, documentation and public release | Upcoming |

## Future Improvements

- Additional edge-case and stress testing.
- Practical tensor-based applications.
- Cross-platform validation on Linux.
- Further performance optimization.
- Improved API documentation.
- Packaging and public release.

## Contributors

Developed as a team project for the **Diwali 2026 Challenge**.

---

**Project Goal:** Build a reusable, efficient, and lightweight C++ tensor library while exploring memory management, data structures, algorithm optimization, and parallel computing.
