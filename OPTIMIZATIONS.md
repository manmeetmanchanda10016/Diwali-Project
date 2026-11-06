# MiniTensor – Optimization Techniques and Implementation Details

## 1. Introduction

MiniTensor is a custom C++ tensor library designed to implement fundamental tensor operations while exploring memory management, computational efficiency and parallel execution.

This document describes the optimization techniques implemented and evaluated in the project.

The main areas of focus are:

- Memory alignment and management.
- Efficient shape and stride handling.
- SIMD vectorization.
- OpenMP parallelism.
- Cache blocking.
- Matrix multiplication.
- Transpose optimization.
- Broadcasting.
- Reduction operations.

## 2. Memory Management

### 2.1 Aligned Memory Allocation

The Storage implementation uses 64-byte aligned memory allocation.

The implementation uses:

```cpp
::operator new(bytes, std::align_val_t{ALIGNMENT});
```

and the corresponding aligned deallocation operation.

64-byte alignment provides suitably aligned storage for SIMD-oriented computation and can help the compiler generate efficient vector memory accesses.

However, actual SIMD performance also depends on compiler optimization, data layout and memory access patterns. Alignment alone does not guarantee vectorization or a performance improvement.

### 2.2 RAII and Resource Management

The Storage class manages dynamically allocated tensor memory.

The implementation supports:

- Copy construction.
- Move construction.
- Copy assignment using copy-and-swap.
- Move operations.
- Proper memory deallocation.

This ensures that dynamically allocated resources are managed safely and reduces the risk of memory leaks and double deallocation.

### 2.3 Separating Allocation and Initialization

The memory benchmark measures allocation and initialization separately.

This distinction is important because allocating memory and writing values into the allocated memory have different performance costs.

For large tensors, initialization can dominate the total construction time because every element must be written.

Separating these operations provides a clearer understanding of memory management overhead.

## 3. Shape and Stride Optimization

The Shape class stores:

- Tensor dimensions.
- Strides.
- Cached element count.

Caching the total number of elements avoids repeatedly calculating the product of dimensions.

The implementation also includes overflow checks when calculating the total element count.

Strides are used to translate multidimensional tensor indices into memory offsets.

This provides a structured mechanism for accessing tensor elements and supports efficient multidimensional indexing.

## 4. SIMD Vectorization

SIMD stands for Single Instruction, Multiple Data.

Instead of processing one element per instruction, SIMD allows supported processors to perform operations on multiple elements simultaneously.

MiniTensor includes SIMD-oriented implementations for:

- Addition.
- Subtraction.
- Multiplication.

SIMD can improve arithmetic throughput when the compiler can generate efficient vector instructions.

However, the measured results depend on memory bandwidth, compiler optimization, alignment and workload size.

The benchmark showed small and inconsistent SIMD improvements rather than a universal speedup.

For some workloads, SIMD performed slightly better than scalar execution, while for others the difference was negligible or SIMD was slightly slower.

## 5. OpenMP Parallelism

OpenMP is used to parallelize selected computational operations.

The project includes parallel variants of elementwise operations and matrix multiplication.

OpenMP distributes work across multiple threads, allowing independent computations to execute concurrently.

The matrix multiplication benchmark evaluates different thread counts:

- 1 thread.
- 2 threads.
- 4 threads.
- 8 threads.

The elementwise benchmarks also include larger thread-count experiments.

### Important Observation

The Google Colab runtime exposed only two logical CPUs.

Therefore, configurations using four or eight threads oversubscribe the available CPU allocation.

This can introduce scheduling and synchronization overhead and may reduce performance.

The benchmark results showed that OpenMP with two threads performed best for the tested 1024 × 1024 matrix multiplication workload.

The best thread count depends on the hardware and workload.

## 6. Matrix Multiplication Optimization

Matrix multiplication is one of the most computationally intensive tensor operations.

The project implements several approaches:

1. Normal sequential multiplication.
2. Sequential blocked multiplication.
3. OpenMP multiplication.
4. Hybrid multiplication.
5. Parallel blocked multiplication.

### 6.1 Sequential Implementation

The normal implementation performs the conventional matrix multiplication operation.

For matrices A and B:

\[
C_{ij}=\sum_k A_{ik}B_{kj}
\]

The output matrix C contains the dot products of the corresponding rows and columns.

### 6.2 Cache Blocking

Cache blocking divides the matrix computation into smaller tiles.

Instead of processing the entire matrix at once, the algorithm processes smaller blocks that can be reused while in cache.

The purpose is to improve data locality and reduce unnecessary memory traffic.

The project evaluates block sizes:

- 32.
- 64.
- 128.
- 256.

The optimal block size depends on matrix dimensions and hardware cache characteristics.

### 6.3 OpenMP Matrix Multiplication

OpenMP parallelizes independent portions of matrix multiplication.

This allows multiple threads to calculate different portions of the output matrix.

The performance depends on available CPU resources, thread scheduling and workload size.

### 6.4 Parallel Blocked Matrix Multiplication

The parallel-blocked implementation combines cache tiling with OpenMP parallel execution.

The implementation uses OpenMP's `collapse(2)` directive across row and column tile iterations and an inner SIMD loop.

This exposes additional independent tile work to the OpenMP runtime and creates opportunities for vectorized computation within each tile.

However, the measured Colab results did not show this variant consistently outperforming the sequential blocked or OpenMP implementations.

Further tuning should be performed on the final evaluation hardware.

### 6.5 Matrix Multiplication Benchmark Findings

For a 1024 × 1024 matrix multiplication workload, the OpenMP implementation using two threads achieved the best median execution time among the tested configurations.

| Configuration | Median execution time |
|---|---:|
| Normal sequential | 443.092 ms |
| Sequential blocked (128) | 384.882 ms |
| OpenMP (2 threads) | 371.805 ms |
| Hybrid (4 threads) | 372.913 ms |
| OpenMP (8 threads) | 383.062 ms |

The OpenMP implementation with two threads was approximately 16.1% faster than the normal sequential implementation in this benchmark run.

All 29 tested configurations passed correctness validation.

These results are specific to the Colab environment and should be re-evaluated on the final competition hardware.

## 7. Transpose Optimization

Matrix transpose changes the arrangement of matrix elements by exchanging rows and columns.

A naive transpose can lead to inefficient memory access patterns because source and destination elements are accessed in different traversal orders.

### Cache-Blocked Transpose

The project implements blocked transpose using smaller matrix tiles.

This improves locality by processing a small region of the matrix before moving to another region.

The tested block sizes are:

- 16.
- 32.
- 64.
- 128.

The benchmark showed that block size 32 performed best for the larger tested matrices in the Colab environment.

For example, the 2048 × 2048 transpose improved from approximately 66.96 ms to 30.94 ms.

This represents approximately 2.16× speedup over the normal transpose baseline.

### Transpose Benchmark Results

| Matrix size | Normal (ms) | Best block size | Blocked time (ms) | Speedup |
|---|---:|---:|---:|---:|
| 256 × 256 | 0.1964 | 16 | 0.1190 | 1.65× |
| 512 × 512 | 1.2918 | 32 | 0.8772 | 1.47× |
| 1024 × 1024 | 5.3929 | 32 | 3.7340 | 1.44× |
| 2048 × 2048 | 66.9556 | 32 | 30.9378 | 2.16× |

The results demonstrate that cache blocking can improve transpose performance, particularly for larger matrices.

## 8. Broadcasting Optimization

Broadcasting allows operations between tensors with compatible but different shapes.

The project includes general broadcasting and specialized benchmark variants.

The tested cases include:

- Same-shape tensors.
- Row broadcasting.
- Column broadcasting.
- Higher-dimensional broadcasting.

The implementation uses tensor shape and stride information to map output elements to their corresponding input elements.

Several benchmark versions were developed to investigate precomputed indexing and specialized execution paths.

The results showed improvements in some higher-dimensional cases but slowdowns in several simpler cases.

The third benchmark version used matched inputs, allocation and timing methodology to compare the library against a direct reference implementation.

Further profiling is required before claiming a general broadcasting performance improvement.

## 9. Reduction Operations

The library implements common reduction operations:

- Sum.
- Mean.
- Minimum.
- Maximum.

These operations traverse tensor elements and compute a single aggregate result.

The implementation is benchmarked on large input tensors to evaluate the cost of processing and reducing tensor data.

For a tensor containing one million elements, the recorded benchmark results were:

| Operation | Execution time (ms) |
|---|---:|
| Sum | 0.6347 |
| Mean | 0.6349 |
| Minimum | 1.2752 |
| Maximum | 1.2698 |

These measurements provide a baseline for evaluating future reduction optimizations.

## 10. Reshape Optimization

The Tensor implementation supports reshape operations.

For compatible shapes, reshaping can be performed by updating shape metadata rather than copying the underlying tensor data.

This can make reshape significantly faster than operations that modify every tensor element.

The benchmark measures two shape changes on a 262,144-element tensor.

The recorded execution time was approximately 0.0002 ms.

This demonstrates the low overhead of metadata-based reshaping in the tested configuration.

## 11. Benchmarking and Performance Evaluation

The project uses repeated measurements to evaluate performance.

The C++ matrix multiplication and memory benchmarks use:

- 3 warm-up runs.
- 10 measured runs.
- Average execution time.
- Median execution time.
- CSV result generation.

The other MiniTensor benchmark programs report their measurements through console logs. Their timing methodology should be interpreted according to each benchmark's implementation rather than assumed to be identical to the CSV-based benchmarks.

The separate PyTorch GPU benchmark uses CUDA synchronization to measure completed GPU work accurately.

Compute-only and end-to-end measurements are kept separate.

### Compute-Only Measurement

Inputs are already present on the target device, and the computation itself is measured.

### End-to-End Measurement

The measurement includes:

1. Copying input data from CPU memory to GPU memory.
2. Performing the computation on the GPU.
3. Copying the result back to CPU memory.

This distinction is important because GPU operations can have significant data-transfer overhead.

The two measurement types represent different workloads and should not be directly treated as equivalent.

## 12. Optimization Findings

The benchmark experiments produced the following observations:

| Optimization | Observation |
|---|---|
| 64-byte aligned memory | Provides aligned tensor storage |
| Cached shape metadata | Avoids repeated element-count calculations |
| SIMD | Small and workload-dependent improvements |
| OpenMP | Can improve selected larger workloads |
| Cache blocking | Improved transpose performance |
| Sequential blocked matrix multiplication | Improved selected matrix sizes |
| OpenMP matrix multiplication | Best tested result at two threads for 1024 × 1024 |
| Parallel blocked matrix multiplication | Did not consistently outperform other variants |
| Specialized broadcasting | Results vary by input shape |
| Metadata-based reshape | Very low overhead in the tested benchmark |
| Separate allocation and initialization timing | Provides clearer memory performance analysis |

Overall, the experiments demonstrate that optimization effectiveness depends on workload size, memory access patterns, hardware characteristics and parallelization overhead.

## 13. Future Optimization Opportunities

Potential improvements include:

1. Tune block sizes using the final evaluation CPU.
2. Investigate parallel scheduling overhead in matrix multiplication.
3. Profile broadcasting index calculations.
4. Improve SIMD efficiency for large elementwise workloads.
5. Evaluate OpenMP thread scaling on hardware with more physical cores.
6. Investigate memory bandwidth limitations.
7. Re-run all benchmarks using the professor's common benchmark protocol.
8. Compare optimized operations against trusted reference implementations.
9. Evaluate additional cache-aware matrix multiplication strategies.
10. Investigate the impact of compiler vectorization and hardware-specific optimization flags.

## 14. Conclusion

MiniTensor combines fundamental tensor functionality with several performance-oriented implementation techniques.

The project explores aligned memory allocation, RAII-based resource management, cached shape metadata, SIMD vectorization, OpenMP parallelism and cache blocking.

Benchmarking demonstrates that optimization effectiveness depends on workload size, hardware characteristics and implementation overhead.

The experiments show measurable improvements in selected matrix multiplication and transpose configurations, while SIMD, OpenMP and broadcasting results vary depending on the workload.

The results provide a foundation for further performance tuning and evaluation.

All reported performance measurements are preliminary Colab results. Final evaluation should use the professor's common benchmark protocol and the specified evaluation hardware.