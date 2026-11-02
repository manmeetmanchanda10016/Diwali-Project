# MiniTensor — A C++ Mini Tensor Library

A lightweight, reusable multidimensional tensor library developed in C++ with a focus on memory management, mathematical operations, CPU performance optimization, and efficient tensor computation.

**Project:** Diwali 2026 Challenge  
**Team Leader:** MANMEET SINGH MANCHANDA  
**Team Size:** 4  
**Development Period:** 01–05 November 2026  
**Release:** v0.1.0 — Initial Release

[GitHub Repository](https://github.com/manmeetmanchanda10016/Diwali-Project)

---

## 1. Project Overview

MiniTensor is a custom C++ tensor library designed to create, store, access, and manipulate multidimensional numerical data.

The library implements fundamental tensor operations and incorporates CPU optimization techniques such as:

- Contiguous memory allocation
- 64-byte aligned storage
- SIMD optimization
- OpenMP parallel execution
- Cache blocking
- Hybrid matrix multiplication
- Efficient multidimensional indexing

The project was developed in seven phases, progressing from basic tensor representation to advanced operations, performance optimization, testing, and CMake packaging.

An additional GPU experiment was conducted using PyTorch and an NVIDIA Tesla T4 GPU in Google Colab to study the performance benefits of CUDA acceleration.

**Important:** MiniTensor currently supports CPU execution. CUDA/GPU execution is not implemented in the library. The Tesla T4 experiment was conducted separately using PyTorch.

---

## 2. Features

### Tensor Representation and Memory Management
- Multidimensional tensor representation.
- Shape and dimension management.
- Contiguous memory storage.
- Custom 64-byte aligned memory allocation.
- Deep copy and move semantics.
- Memory management and overflow checks.
- Scalar and zero-sized tensor handling.

### Indexing and Reshaping
- Row-major memory layout.
- Precomputed strides.
- Flat indexing using `flat_at()`.
- Multidimensional indexing using `at()`.
- Tensor reshaping without copying tensor elements.

### Mathematical Operations
- Elementwise addition.
- Elementwise subtraction.
- Elementwise multiplication.
- Matrix multiplication.
- Parallel matrix multiplication using OpenMP.

### Advanced Operations
- Matrix transpose.
- Broadcasting.
- Broadcasting addition, subtraction, and multiplication.
- Sum reduction.
- Mean reduction.
- Minimum reduction.
- Maximum reduction.

### Performance Optimization
- SIMD-based elementwise operations.
- OpenMP parallel execution.
- Parallel SIMD execution.
- Blocked matrix multiplication.
- Parallel blocked matrix multiplication.
- Hybrid matrix multiplication.
- Blocked transpose.
- Native CPU optimization.
- Link-Time Optimization (LTO).

### Testing and Packaging
- Six test suites.
- Example applications.
- Interactive tensor playground.
- Performance benchmark programs.
- CMake installation and exported targets.
- Installable package configuration.

---

## 3. Project Structure

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

## 4. Development Phases

### Phase 1 — Tensor Representation and Memory Management

Implemented:
- Shape class for storing tensor dimensions.
- Cached element count.
- Storage class for contiguous memory allocation.
- 64-byte aligned memory allocation.
- Tensor class combining shape and storage.
- Deep copy and move semantics.
- Scalar and empty tensor handling.
- Overflow detection.

### Phase 2 — Indexing and Reshaping

Implemented:
- Row-major stride calculation.
- Multidimensional indexing.
- Flat indexing.
- `at()` and `flat_at()` access methods.
- Tensor reshaping.

### Phase 3 — Mathematical Operations

Implemented:
- Elementwise addition.
- Elementwise subtraction.
- Elementwise multiplication.
- Matrix multiplication.
- OpenMP parallel matrix multiplication.
- Hybrid matrix multiplication.

### Phase 4 — Advanced Tensor Operations

Implemented:
- Matrix transpose.
- Broadcasting.
- Broadcasting arithmetic operations.
- Sum and mean reductions.
- Minimum and maximum reductions.
- Empty tensor reduction handling.

### Phase 5 — Performance Optimization

Implemented:
- Blocked matrix multiplication.
- Parallel blocked matrix multiplication.
- SIMD elementwise operations.
- OpenMP SIMD.
- Blocked transpose.
- Hybrid execution.
- Native CPU optimization.
- Link-Time Optimization.

### Phase 6 — Testing and Application

Implemented:
- Six independent test suites.
- Tensor examples.
- Interactive tensor playground.
- Benchmark programs for different operations.
- Correctness verification.

### Phase 7 — Packaging and Documentation

Implemented:
- CMake installation rules.
- Exported CMake library target.
- Package configuration.
- README documentation.
- MIT License.
- GitHub repository and v0.1.0 release.

---

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| C++ | Core library implementation |
| CMake | Build system and packaging |
| GCC | C++ compilation |
| OpenMP | CPU parallel execution |
| SIMD | Vectorized CPU operations |
| Git | Version control |
| GitHub | Project hosting and release |
| Google Colab | Additional benchmarking environment |
| PyTorch | External tensor library comparison |
| CUDA | GPU experiment through PyTorch |

---

## 6. Performance Benchmarks

Performance was evaluated using separate benchmark runs on the developer's laptop and Google Colab.

The results below are reported separately because hardware, compiler versions, CPU allocation, and build configurations differ.

### 6.1 Developer Laptop — Matrix Multiplication

**Hardware:** AMD Ryzen AI 7 350  
**CPU:** 8 physical cores / 16 logical processors  
**Compiler:** MSYS2 GCC 16.1.0  
**Build:** Release with native optimization and LTO.

| Matrix Size | Sequential | Best Parallel/Hybrid | Speedup |
|---|---:|---:|---:|
| 64×64 | 0.023 ms | 0.026 ms (OpenMP, 1 thread) | — |
| 128×128 | 0.177 ms | 0.068 ms | 2.60× |
| 256×256 | 1.444 ms | 0.430 ms | 3.36× |
| 512×512 | 12.494 ms | 3.476 ms | 3.59× |
| 1024×1024 | 95.876 ms | 25.204 ms | 3.80× |

The best parallel configurations were selected from the tested thread counts. Parallel execution provided greater benefits as matrix size increased.

### 6.2 Developer Laptop — Blocked Transpose

| Matrix Size | Normal Transpose | Blocked 16×16 | Speedup |
|---|---:|---:|---:|
| 256×256 | 0.518 ms | 0.143 ms | 3.62× |
| 512×512 | 2.269 ms | 0.985 ms | 2.30× |
| 1024×1024 | 9.736 ms | 4.171 ms | 2.33× |
| 2048×2048 | 40.869 ms | 15.870 ms | 2.58× |

Blocking improves cache locality by processing smaller matrix regions.

### 6.3 Developer Laptop — Elementwise Operations

Tensor size: 16 million elements.

| Operation | Scalar | SIMD | Best Parallel |
|---|---:|---:|---:|
| Addition | 100.311 ms | 42.135 ms | 40.131 ms |
| Subtraction | 100.462 ms | 33.138 ms | 40.265 ms |
| Multiplication | 101.531 ms | 41.696 ms | 40.206 ms |

The results demonstrate the benefits of SIMD and parallel execution for large elementwise workloads.

---

## 7. Google Colab — MiniTensor CPU Benchmarks

Additional benchmarking was performed using Google Colab.

**Compiler:** GNU 13.3.0  
**OpenMP:** 4.5  
**LTO:** Enabled  
**Native CPU optimization:** Disabled

### 7.1 Matrix Multiplication

| Matrix Size | Sequential | Best Parallel/Hybrid | Speedup |
|---|---:|---:|---:|
| 64×64 | 0.173 ms | 0.168 ms | 1.03× |
| 128×128 | 1.408 ms | 1.140 ms | 1.24× |
| 256×256 | 6.112 ms | 5.551 ms | 1.10× |
| 512×512 | 50.952 ms | 44.856 ms | 1.14× |
| 1024×1024 | 636.381 ms | 367.324 ms | 1.73× |

The best configuration varied by matrix size. The Colab results show approximately 1.73× improvement at 1024×1024.

### 7.2 Elementwise Operations

Best observed SIMD or parallel configuration:

| Tensor Size | Operation | Scalar | Best Optimized | Speedup |
|---|---|---:|---:|---:|
| 1M | Addition | 1.402 ms | 1.277 ms | 1.10× |
| 1M | Subtraction | 1.335 ms | 1.332 ms | 1.00× |
| 1M | Multiplication | 1.450 ms | 1.298 ms | 1.12× |
| 4M | Addition | 8.264 ms | 7.333 ms | 1.13× |
| 4M | Subtraction | 8.263 ms | 7.639 ms | 1.08× |
| 4M | Multiplication | 8.516 ms | 7.336 ms | 1.16× |
| 16M | Addition | 87.326 ms | 84.221 ms | 1.04× |
| 16M | Subtraction | 87.818 ms | 84.611 ms | 1.04× |
| 16M | Multiplication | 88.441 ms | 83.732 ms | 1.06× |

All reported elementwise configurations passed the benchmark correctness checks.

The gains were modest in Colab, and increasing the thread count did not always improve execution time. This can happen when operations are limited by memory bandwidth or parallel execution overhead.

### 7.3 Blocked Transpose

| Matrix Size | Normal | Best Block Size | Best Blocked Time | Speedup |
|---|---:|---:|---:|---:|
| 256×256 | 0.1917 ms | 16 | 0.0830 ms | 2.31× |
| 512×512 | 1.2851 ms | 64 | 0.8340 ms | 1.54× |
| 1024×1024 | 5.3095 ms | 64 | 3.5020 ms | 1.52× |
| 2048×2048 | 58.6203 ms | 16 | 29.8970 ms | 1.96× |

The optimal block size varied depending on matrix dimensions.

### 7.4 Additional Operations

| Operation | Input | Execution Time |
|---|---|---:|
| Construction | 1,024 elements | 0.0005 ms |
| Construction | 100,000 elements | 0.0213 ms |
| Construction | 1M elements | 0.3852 ms |
| Flat indexing | 262,144 elements | 0.3334 ms |
| Multidimensional indexing | 262,144 accesses | 5.3204 ms |
| Two reshape changes | 262,144 elements | 0.0002 ms |
| Fill | 1M elements | 0.3818 ms |
| Sum | 1M elements | 0.6387 ms |
| Mean | 1M elements | 0.6367 ms |
| Minimum | 1M elements | 1.3341 ms |
| Maximum | 1M elements | 1.2918 ms |

---

## 8. CUDA Experiment — PyTorch CPU vs NVIDIA Tesla T4

An additional GPU experiment was conducted in Google Colab to study the performance difference between CPU and GPU tensor computation.

**Environment:**
- PyTorch: 2.11.0+cu130
- GPU: NVIDIA Tesla T4
- CPU threads: 1
- CUDA execution through PyTorch.

The experiment measured matrix multiplication and elementwise operations on CPU and CUDA.

**Important:** MiniTensor does not currently implement CUDA. This experiment was performed separately using PyTorch to evaluate GPU acceleration.

### 8.1 Matrix Multiplication

| Matrix Size | PyTorch CPU | PyTorch CUDA | GPU Speedup |
|---|---:|---:|---:|
| 64×64 | 0.0083 ms | 0.0315 ms | 0.26× |
| 128×128 | 0.0345 ms | 0.0353 ms | 0.98× |
| 256×256 | 0.2908 ms | 0.0687 ms | 4.23× |
| 512×512 | 2.4581 ms | 0.1590 ms | 15.46× |
| 1024×1024 | 23.8251 ms | 0.9082 ms | 26.24× |

The Tesla T4 showed substantial acceleration for larger matrices.

For smaller matrices, GPU execution overhead reduced the performance advantage.

### 8.2 Elementwise Operations

| Operation | PyTorch CPU | PyTorch CUDA | GPU Speedup |
|---|---:|---:|---:|
| Addition | 47.2916 ms | 0.8096 ms | 58.40× |
| Subtraction | 36.5214 ms | 0.8017 ms | 45.55× |
| Multiplication | 36.0570 ms | 0.7983 ms | 45.16× |

**Benchmark methodology:** Input tensors were already allocated on the GPU before the timed operations. CUDA synchronization was used around measurements. Therefore, these CUDA timings exclude host-to-device data transfer.

### 8.3 CUDA Experiment Findings

- GPU acceleration becomes more beneficial as computational workload increases.
- Small matrix operations may not benefit from GPU execution because of launch and execution overhead.
- The Tesla T4 achieved approximately 26.24× the PyTorch CPU performance for 1024×1024 matrix multiplication in this experiment.
- Elementwise operations also showed significant GPU acceleration in the recorded measurements.
- These are PyTorch CPU-versus-CUDA results, not MiniTensor GPU results.

### 8.4 MiniTensor vs PyTorch

MiniTensor's Colab CPU results can also be viewed alongside the PyTorch results.

| Matrix Size | MiniTensor Best CPU | PyTorch CPU | PyTorch CUDA |
|---|---:|---:|---:|
| 64×64 | 0.168 ms | 0.0083 ms | 0.0315 ms |
| 128×128 | 1.140 ms | 0.0345 ms | 0.0353 ms |
| 256×256 | 5.551 ms | 0.2908 ms | 0.0687 ms |
| 512×512 | 44.856 ms | 2.4581 ms | 0.1590 ms |
| 1024×1024 | 367.324 ms | 23.8251 ms | 0.9082 ms |

This is an indicative comparison between different implementations and benchmark harnesses, not a controlled comparison of identical kernels. PyTorch uses highly optimized numerical libraries, while MiniTensor is a custom C++ implementation.

---

## 9. Testing and Validation

MiniTensor includes six independent test suites.

| Test Suite | Status |
|---|---|
| Phase 1 — Tensor Representation | Passed |
| Phase 2 — Indexing | Passed |
| Phase 3 — Mathematical Operations | Passed |
| Phase 4 — Advanced Operations | Passed |
| Phase 5 — Optimization | Passed |
| Phase 6 — Testing/Application | Passed |

**Total: 6/6 test suites passed.**

The tests were executed using CTest in both the local laptop environment and Google Colab.

An independent CMake package consumer test also successfully located and linked the installed MiniTensor library using:

```cmake
find_package(MiniTensor REQUIRED)
```

The exported target is:

```cmake
MiniTensor::minitensor
```

---

## 10. Building the Project

### Requirements

- C++ compiler with C++17 support.
- CMake.
- OpenMP support.
- GCC or Clang recommended.

### Build on Windows (MSYS2 MinGW)

Configure the Release build:

```powershell
cmake -S . -B build-release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DMINITENSOR_ENABLE_NATIVE=ON -DMINITENSOR_ENABLE_LTO=ON
```

Build:

```powershell
cmake --build build-release -j
```

Run tests:

```powershell
ctest --test-dir build-release --output-on-failure
```

Install:

```powershell
cmake --install build-release --prefix install
```

### Build on Linux

Configure:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

Build:

```bash
cmake --build build -j
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```

---

## 11. Benchmarks

The project includes separate benchmark programs for:

- Tensor construction and memory allocation.
- Indexing and reshaping.
- Matrix multiplication.
- Elementwise operations.
- Broadcasting.
- Matrix transpose.
- Reduction operations.

Run the benchmark executables after building the project:

```bash
./build-release/benchmark_matmul
./build-release/benchmark_elementwise
./build-release/benchmark_transpose
./build-release/benchmark_remaining
```

On Windows, run the corresponding `.exe` files from the build directory if required.

Benchmark results may vary depending on hardware, compiler settings, CPU allocation, and system load.

---

## 12. CMake Packaging

MiniTensor supports installation and package export through CMake.

The library target is:

```cmake
MiniTensor::minitensor
```

A separate consumer project can locate the installed package using:

```cmake
find_package(MiniTensor REQUIRED)
```

The package configuration also declares its OpenMP dependency.

---

## 13. Limitations and Future Improvements

Current limitations:

- CUDA/GPU execution is not implemented in MiniTensor.
- OpenMP is currently required by the configured build and package.
- Performance depends on compiler optimizations and hardware characteristics.
- Benchmark results may vary across execution environments.

Possible future improvements:

- Optional CUDA backend.
- GPU-accelerated matrix multiplication.
- Optional OpenMP support.
- Additional tensor manipulation operations.
- Improved parallel reductions.
- Broader compiler and platform testing.
- More extensive performance benchmarking.

---

## 14. Conclusion

MiniTensor demonstrates the implementation of a reusable multidimensional tensor library in C++.

The project covers tensor representation, memory management, indexing, mathematical operations, broadcasting, reductions, CPU optimization, testing, and CMake packaging.

Performance experiments demonstrated improvements through OpenMP parallel execution, SIMD optimization, and cache blocking. The additional PyTorch CUDA experiment provided a comparison of CPU and GPU execution, highlighting the potential advantages of GPU acceleration for larger workloads.

The project successfully completed all seven development phases and provides a foundation for further development of high-performance tensor computation in C++.

---

## 15. Project Information

**Project:** MiniTensor — C++ Mini Tensor Library  
**Challenge:** Diwali 2026 Challenge  
**Team Leader:** MANMEET SINGH MANCHANDA  
**Team Size:** 4  
**Release:** v0.1.0 — Initial Release

**GitHub Repository:**  
https://github.com/manmeetmanchanda10016/Diwali-Project

**License:** MIT License