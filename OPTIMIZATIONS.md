# MiniTensor Optimizations and Implementation Notes

## 1. Purpose

This document describes the implementation and optimization techniques present in the MiniTensor project, based on the project details and benchmark archive supplied for review. It distinguishes implementation features from measured performance outcomes; a technique being implemented does not guarantee a speedup for every workload or machine.

## 2. Memory management

### 2.1 Aligned allocation

The `Storage` implementation uses aligned allocation with a 64-byte alignment requirement:

- Allocation: `::operator new(bytes, std::align_val_t{ALIGNMENT})`
- Matching deallocation: aligned `::operator delete` using the same alignment.

This provides aligned storage suitable for SIMD-oriented data access and avoids mismatched allocation/deallocation forms.

### 2.2 Ownership and lifetime

The project includes:
- Copy and move support for `Storage`.
- Copy-and-swap assignment behavior.
- A `Tensor` that owns a `Shape` and `Storage`.
- Explicit separation of allocation, initialization and destruction in the memory benchmark.

The benchmark measures raw allocation/deallocation separately from initialization. This is important because touching each page and writing tensor values costs substantially more than obtaining or releasing an allocation from the allocator.

### 2.3 Shape metadata

`Shape` stores dimensions and strides and caches the total element count. The project also checks for element-count overflow. Cached metadata can avoid repeatedly recomputing common shape properties during operations.

## 3. Tensor API and correctness safeguards

The recorded implementation includes:
- Bounds-checked element access.
- Reshape support.
- Shape/stride metadata.
- Tensor construction from shape and storage.
- Operations for arithmetic, broadcasting, matrix multiplication, transpose and reductions.

The benchmark log shows reshape of a 262,144-element tensor completing in approximately 0.0002 ms for two shape changes. This is consistent with reshape updating metadata rather than copying all tensor elements for compatible shapes; verify the exact implementation constraints in source before making a stronger API guarantee.

Multidimensional `at({row, col})` was substantially slower than flat indexing in the supplied benchmark (5.559 ms versus 0.334 ms for 262,144 accesses). The multidimensional path performs additional index/stride and bounds work. This is an API safety/ergonomics tradeoff, not necessarily a defect.

## 4. SIMD elementwise operations

The project includes scalar and SIMD variants for elementwise addition, subtraction and multiplication. SIMD allows the compiler to process multiple values per vector instruction where supported and where loop structure, alignment and aliasing constraints permit.

Measured outcome:
- At 1M elements, addition SIMD was about 1.09× faster than scalar in the supplied run.
- Subtraction was approximately 1.01×.
- Multiplication was slightly slower (about 0.97×).
- At 4M and 16M elements, results were mostly close to parity and varied by operation.

Interpretation: SIMD is available, but the measured workload appears sensitive to memory bandwidth, compiler auto-vectorization, runtime conditions and measurement noise. Do not claim a universal SIMD speedup.

## 5. OpenMP parallelism

OpenMP variants are present for elementwise operations and matrix multiplication. The matrix benchmark tests 1, 2, 4 and 8 threads; the elementwise log tests 1, 2, 4, 8 and 16 threads.

The Colab VM exposes only two logical CPUs. Therefore:
- One and two threads are the relevant configurations for interpreting performance on this machine.
- Four, eight and sixteen threads oversubscribe the visible CPU allocation.
- More threads can increase scheduling and synchronization overhead without adding CPU resources.

In the supplied 1M-element elementwise run, 2-thread variants were modestly faster than scalar for all three operations. At 16M elements, gains remained small (roughly 1.03×–1.04× for several two-thread cases). This supports treating OpenMP as a workload- and hardware-dependent option rather than an always-on speedup.

## 6. Matrix multiplication

The benchmark compares:
- Normal sequential multiplication.
- Sequential blocked multiplication with block sizes 32, 64, 128 and 256.
- OpenMP multiplication with multiple thread counts.
- Hybrid variants.
- Parallel-blocked variants with different block sizes and thread counts.

### 6.1 Cache blocking

Blocking divides the matrix work into tiles so that sub-blocks can be reused while resident in cache. The best block size depends on cache hierarchy, matrix size, compiler and CPU.

In this Colab run:
- 512 × 512: Hybrid with 2 threads measured 44.745 ms versus 48.556 ms for normal sequential (about 1.09×).
- 1024 × 1024: OpenMP with 2 threads measured 371.805 ms versus 443.092 ms for normal sequential (about 1.19×).
- ParallelBlocked was slower than the best sequential/OpenMP/Hybrid variants in the tested configurations.

### 6.2 Parallel tile scheduling

The parallel-blocked implementation was updated to use `collapse(2)` across row/column tiles with an inner `omp simd` loop, according to the project development notes. This exposes more independent tile work to OpenMP and vectorizes the innermost work. However, the current benchmark results do not show this variant winning on the 2-logical-CPU Colab VM. Further tuning should be evaluated on dedicated hardware and the official test sizes.

## 7. Transpose blocking

Transpose accesses source and destination matrices with different memory traversal directions. A naive transpose can cause poor cache locality. Tiled transpose processes a small submatrix at a time to improve locality.

The supplied benchmark reports:
- 256 × 256: block 16, 0.1190 ms versus normal 0.1964 ms (1.65×).
- 512 × 512: block 32, 0.8772 ms versus normal 1.2918 ms (1.47×).
- 1024 × 1024: block 32, 3.7340 ms versus normal 5.3929 ms (1.44×).
- 2048 × 2048: block 32, 30.9378 ms versus normal 66.9556 ms (2.16×).

Block 32 was the best tested choice for the three largest sizes in this run. This is an empirical result for the current Colab CPU and should be retested on the final evaluation hardware.

## 8. Broadcasting

The project includes general broadcasting and optimized/specialized experiments. Broadcasting relies on shape/stride compatibility and mapping output indices to input indices.

The supplied V1 and V2 logs show inconsistent performance: higher-dimensional cases improve by about 2×, while some same-shape and simple row/column cases show large slowdowns for precomputed/optimized variants. V3 uses the same inputs, allocation and timing methodology for library and direct-reference comparisons; its results are mostly near parity with mixed wins/losses.

Recommended documentation stance:
- Describe broadcasting support and the specialized experiments.
- Do not claim a general broadcasting speedup based on V1/V2.
- Keep V3 as the more controlled comparison and profile index mapping, allocation, and loop overhead before further optimization.

## 9. Reductions and utility operations

The project benchmarks fill, sum, mean, min and max. For 1M elements, the supplied log reports:
- Fill: 0.4426 ms.
- Sum: 0.6347 ms.
- Mean: 0.6349 ms.
- Min: 1.2752 ms.
- Max: 1.2698 ms.

These are recorded results, not universal performance guarantees.

## 10. Separate PyTorch CUDA experiment

The project archive also contains a PyTorch CPU-versus-Tesla-T4 benchmark. This is a separate experiment and must not be described as MiniTensor GPU support.

The final controlled run fixes PyTorch CPU threads at 2, uses float64, three warmups, ten measurements, CUDA synchronization and correctness checks. It measures:
- Device-resident compute-only operations.
- End-to-end operations that copy inputs from CPU to GPU and output back to CPU.

The results show large compute-only CUDA gains for large elementwise workloads, but end-to-end CUDA is slower than CPU for elementwise cases because transfer overhead dominates. CUDA is faster end-to-end for the 1024 × 1024 matrix multiplication case in this run.

## 11. Recommended next optimization work

1. Re-run using the professor's common protocol and identical test data.
2. Re-run on a stable machine with a known CPU model and physical/logical core count.
3. Profile before changing algorithms; separate allocation, initialization, computation and synchronization costs.
4. Tune matrix tile sizes on the target CPU rather than assuming one block size is universally optimal.
5. Reassess whether parallel-blocked matmul is worthwhile for the target machine; current Colab results do not show a win.
6. Investigate broadcasting V1/V2 timing methodology and compare variants under identical allocation and timing boundaries.
7. Keep 1-thread and 2-thread baselines for the current Colab VM; test higher thread counts only on hardware that actually exposes those cores.
8. Preserve correctness tests for every optimized path and compare outputs against a trusted reference.

## 12. Claims to avoid

- Do not claim that SIMD always improves performance.
- Do not claim that more OpenMP threads always improve performance.
- Do not claim the parallel-blocked matrix multiplication is the fastest implementation; it was not in the supplied run.
- Do not claim general broadcasting acceleration from V1/V2.
- Do not claim MiniTensor supports CUDA based on the PyTorch GPU benchmark.
- Do not present Colab measurements as official competition results until the common protocol is applied.
