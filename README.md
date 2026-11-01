# Mini Tensor Library (C++)

A C++17 multidimensional tensor library developed for the Diwali 2026 coding challenge. It provides contiguous, 64-byte aligned storage; multidimensional indexing; tensor operations; broadcasting; reductions; and OpenMP matrix multiplication.

## Features

### Phase 1 – Tensor Representation and Memory Management

- Create tensors with arbitrary-rank shapes.
- Store values in contiguous, 64-byte aligned memory.
- Track tensor shape and element count with overflow checking.
- Support deep copy and move operations.
- Support scalar and empty tensors.

### Phase 2 – Indexing and Memory Layout

- Use row-major memory layout.
- Access elements with multidimensional `at()` and `operator()`.
- Access contiguous elements with `flat_at()`.
- Reshape tensors without changing their data.

### Phase 3 – Mathematical Operations

- Element-wise addition, subtraction, and multiplication.
- Matrix multiplication.
- OpenMP parallel matrix multiplication.
- Hybrid matrix multiplication that selects sequential or parallel execution based on estimated workload.

### Phase 4 – Advanced Tensor Operations

- Matrix transpose.
- Broadcasting for addition, subtraction, and multiplication.
- Reductions: `sum()`, `mean()`, `min()`, and `max()`.
- Error handling for empty reductions and incompatible shapes.

### Phase 5 – Performance and Parallel Computing

- Compare baseline, blocked, OpenMP, and library hybrid matrix multiplication.
- Run benchmarks in Release mode with warmups and repeated measurements.
- Check sequential and parallel multiplication results against expected values.

## Performance

One Release-build benchmark sample on Windows with MinGW:

| Matrix Size | Baseline | OpenMP | Library Hybrid | Hybrid Speedup |
|---|---:|---:|---:|---:|
| 256 × 256 | 2.660 ms | 0.643 ms | 0.561 ms | 4.74× |
| 512 × 512 | 20.453 ms | 3.962 ms | 3.949 ms | 5.18× |
| 1024 × 1024 | 173.768 ms | 35.499 ms | 31.633 ms | 5.49× |

The blocked implementation did not improve performance in this sample. OpenMP overhead can make parallel execution slower for smaller matrices. Timings depend on hardware, compiler, thread count, and system load; these results are not a universal performance guarantee.

## Requirements

- C++17 compiler
- CMake 3.16 or later
- OpenMP-capable compiler and runtime

## Build

### Windows with MinGW

```powershell
cmake -S . -B build-release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j