# Mini Tensor Library (C++)

A lightweight, high-performance multidimensional tensor library developed in C++ as part of the Diwali 2026 coding challenge.

The project focuses on efficient memory management, multidimensional indexing, mathematical operations, broadcasting, and parallel matrix multiplication using OpenMP.

## Features

### Phase 1 – Tensor Representation and Memory Management
- Multidimensional tensor creation.
- Contiguous memory allocation.
- 64-byte aligned memory allocation.
- Shape and element-count management.
- Deep copy and move semantics.
- Scalar and empty tensor support.
- Memory safety and overflow checking.

### Phase 2 – Indexing and Memory Layout
- Row-major memory layout.
- Multidimensional indexing.
- Element access using `at()`.
- Flat indexing using `flat_at()`.
- Tensor reshaping without changing the underlying data.

### Phase 3 – Mathematical Operations
- Element-wise addition.
- Element-wise subtraction.
- Element-wise multiplication.
- Matrix multiplication.
- Operator overloading (`+`, `-`, `*`).
- OpenMP parallel matrix multiplication.
- Hybrid matrix multiplication that selects sequential or parallel execution based on estimated workload.

### Phase 4 – Advanced Tensor Operations
- Matrix transpose.
- Broadcasting for addition, subtraction, and multiplication.
- Tensor reduction operations:
  - `sum()`
  - `mean()`
  - `min()`
  - `max()`
- Empty tensor and incompatible-shape error handling.

## Performance

Matrix multiplication benchmark results from the Release build:

| Matrix Size | Sequential | OpenMP | Hybrid |
|---|---:|---:|---:|
| 256 × 256 | 5.731 ms | 0.890 ms | 0.939 ms |
| 512 × 512 | 33.008 ms | 5.886 ms | 6.013 ms |
| 1024 × 1024 | 245.567 ms | 35.667 ms | 34.696 ms |

The hybrid implementation achieved approximately:
- 6.10× speedup at 256 × 256.
- 5.49× speedup at 512 × 512.
- 7.08× speedup at 1024 × 1024.

*Benchmark results depend on hardware, compiler settings, and system load.*

## Technologies Used

- C++
- CMake
- GCC
- OpenMP
- Git and GitHub

## Project Structure

```text
MiniTensor_Phase1/
├── include/
│   └── minitensor/
│       ├── Shape.h
│       ├── Storage.h
│       ├── Tensor.h
│       └── Ops.h
├── src/
│   ├── Shape.cpp
│   ├── Storage.cpp
│   ├── Tensor.cpp
│   └── Ops.cpp
├── tests/
│   ├── test_phase1.cpp
│   ├── test_phase2.cpp
│   ├── test_phase3.cpp
│   └── test_phase4.cpp
├── benchmarks/
│   ├── benchmark_phase1.cpp
│   ├── benchmark_v2.cpp
│   └── benchmark_matmul.cpp
├── examples/
│   └── main.cpp
├── CMakeLists.txt
└── README.md
```

## Build Instructions

### Requirements
- C++ compiler with C++17 support.
- CMake.
- OpenMP support.

### Build using CMake

```bash
cmake -S . -B build
cmake --build build -j
```

### Run tests

```bash
ctest --test-dir build --output-on-failure
```

### Release build

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
ctest --test-dir build-release --output-on-failure
```

On Windows with MinGW, specify the generator:

```powershell
cmake -S . -B build-release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
```

## Testing

The project currently has four test suites:

- Phase1Tests
- Phase2Tests
- Phase3Tests
- Phase4Tests

All four test suites currently pass.

## Project Status

| Phase | Status |
|---|---|
| Phase 1 – Tensor and Memory Management | Completed |
| Phase 2 – Indexing and Reshape | Completed |
| Phase 3 – Mathematical Operations and OpenMP | Completed |
| Phase 4 – Advanced Tensor Operations | Completed |

## Objective

The goal is to build a compact tensor library from scratch while exploring low-level memory management, numerical computing, algorithm optimization, and parallel execution in C++.
