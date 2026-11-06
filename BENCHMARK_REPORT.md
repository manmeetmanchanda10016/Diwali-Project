# MiniTensor Benchmark Report

## 1. Executive summary

This report records the current MiniTensor CPU benchmark run and a separate PyTorch CPU-versus-CUDA experiment on a Google Colab runtime. The MiniTensor implementation passed the recorded correctness checks in the supplied benchmark logs, and the project test suite was previously reported as passing all six CTest suites.

The measurements show:
- Matrix multiplication performance varies by size and implementation. In the recorded MiniTensor run, the best observed median among the tested configurations was 44.745 ms for a 512 × 512 matrix using the Hybrid variant with 2 threads, and 371.805 ms for a 1024 × 1024 matrix using OpenMP with 2 threads.
- The transpose benchmark's 2048 × 2048 blocked-32 variant measured 30.938 ms versus 66.956 ms for the normal transpose baseline (about 2.16× faster).
- Elementwise SIMD and OpenMP gains were modest and inconsistent on the 2-logical-CPU Colab VM. Configurations using 4, 8, or 16 threads should not be interpreted as scaling results on this machine.
- In the controlled PyTorch experiment, CUDA compute-only execution was much faster for large operations, but end-to-end measurements that include host-to-device input copies and device-to-host output copy were slower than CPU for elementwise operations. For 1024 × 1024 matrix multiplication, CUDA was faster both compute-only and end-to-end in this run.

These are development measurements from a shared, virtualized Colab environment, not final official competition results. The professor's common benchmark protocol, when issued, must take precedence.

## 2. Test environment

### MiniTensor / C++ benchmark environment

| Item | Recorded value |
|---|---|
| Operating system | Ubuntu 24.04.4 LTS |
| Kernel/runtime | Linux x86_64, Google Colab virtual machine |
| CPU | Intel Xeon CPU @ 2.00 GHz |
| Visible logical CPUs | 2 |
| Topology reported by `lscpu` | 1 socket, 1 core, 2 threads/core |
| Compiler | GCC 13.3.0 |
| CMake | 3.31.10 |
| C++ standard | C++17 (from the project build configuration previously verified) |
| Release flags | `-O3 -DNDEBUG` |
| OpenMP | Found and enabled; GCC libgomp, OpenMP 4.5 |
| GPU for separate PyTorch experiment | NVIDIA Tesla T4 |
| PyTorch | 2.11.0+cu130 |
| CUDA runtime reported by PyTorch | 13.0 |

The Colab CPU is virtualized and shared. The two visible logical CPUs are the fair reference for CPU thread-count conclusions. The benchmark includes 4/8-thread configurations for exploration, but these oversubscribe the visible CPU allocation and must not be presented as evidence of 4/8-core scaling.

### PyTorch benchmark configuration

- Data type: `torch.float64`, selected to match MiniTensor's `double` storage.
- CPU threads: fixed to 2 in the final controlled run.
- Warm-up iterations: 3.
- Measured iterations: 10.
- Reported statistics: arithmetic average and median.
- CUDA timing: explicit `torch.cuda.synchronize()` around measured work.
- Correctness: CPU and CUDA outputs compared with `torch.allclose(rtol=1e-8, atol=1e-8)`.
- Workloads: matrix multiplication at 64, 128, 256, 512 and 1024; elementwise add/subtract/multiply at 1,024, 100,000, 1,000,000, 4,000,000 and 16,000,000 elements.
- End-to-end CUDA timing includes copying both input arrays from CPU to GPU, performing the operation, and copying the output back to CPU.

## 3. Methodology and interpretation

The MiniTensor CSV benchmarks record 3 warmups and 10 measurements for matrix multiplication and memory operations. The console logs for elementwise, transpose, broadcasting and remaining operations provide the reported timings and checksums/correctness indicators.

The PyTorch CUDA benchmark reports two different measurements:
1. **Compute-only:** inputs are already resident on the target device; the operation is timed, with CUDA synchronization.
2. **End-to-end:** inputs begin on the CPU, both inputs are copied to CUDA, the operation is performed, and the result is copied back to CPU.

These metrics answer different questions and must not be mixed. Compute-only timings describe device-resident workloads; end-to-end timings describe a one-operation workflow where data starts and ends on the host.

## 4. MiniTensor results

### 4.1 Matrix multiplication

Selected median results from `matmul_benchmark.csv`:

| Matrix size | Configuration | Median (ms) | GFLOPS |
|---:|---|---:|---:|
| 256 × 256 | SequentialBlocked, block 256, 1 thread | 5.688 | 5.899 |
| 512 × 512 | Hybrid, 2 threads | 44.745 | 5.999 |
| 1024 × 1024 | OpenMP, 2 threads | 371.805 | 5.776 |

Baseline comparisons from the same run:

| Matrix size | Normal sequential median (ms) | Best listed median (ms) | Approx. speedup |
|---:|---:|---:|---:|
| 256 × 256 | 5.832 | 5.688 | 1.03× |
| 512 × 512 | 48.556 | 44.745 | 1.09× |
| 1024 × 1024 | 443.092 | 371.805 | 1.19× |

The best configuration depends on matrix size. The parallel-blocked variants were generally slower than the sequential or OpenMP/Hybrid alternatives in this Colab run. This suggests that parallel scheduling and tiling overhead outweighed their benefits for these tested conditions. It does not prove that the same ordering will hold on dedicated hardware.

### 4.2 Transpose

Selected timings from `transpose_results.txt`:

| Matrix size | Normal (ms) | Best blocked variant | Best blocked time (ms) | Speedup vs normal |
|---:|---:|---|---:|---:|
| 256 × 256 | 0.1964 | Block 16 | 0.1190 | 1.65× |
| 512 × 512 | 1.2918 | Block 32 | 0.8772 | 1.47× |
| 1024 × 1024 | 5.3929 | Block 32 | 3.7340 | 1.44× |
| 2048 × 2048 | 66.9556 | Block 32 | 30.9378 | 2.16× |

Blocking improved the measured transpose time for all four sizes in this run. Block size 32 was best for the three largest matrices; block 16 was best at 256 × 256.

### 4.3 Elementwise operations

The elementwise log reports scalar, SIMD, and OpenMP results at 1M, 4M and 16M elements. Examples:

| Elements | Operation | Scalar (ms) | SIMD (ms) | SIMD speedup | Best tested parallel result |
|---:|---|---:|---:|---:|---:|
| 1,000,000 | Addition | 1.465 | 1.347 | 1.09× | 2 threads: 1.287 ms |
| 1,000,000 | Subtraction | 1.418 | 1.401 | 1.01× | 2 threads: 1.291 ms |
| 1,000,000 | Multiplication | 1.348 | 1.387 | 0.97× | 2 threads: 1.296 ms |
| 4,000,000 | Addition | 8.459 | 8.832 | 0.96× | 4 threads: 7.870 ms |
| 16,000,000 | Addition | 89.104 | 87.681 | 1.02× | 2 threads: 85.451 ms |
| 16,000,000 | Multiplication | 86.841 | 87.248 | 1.00× | 4 threads: 82.348 ms |

The gains are small and vary by operation and size. SIMD does not consistently beat the scalar baseline in these measurements. OpenMP with 2 threads often gives a modest improvement, while larger thread counts frequently regress due to the two-logical-CPU environment and runtime overhead.

### 4.4 Broadcasting

The supplied broadcasting benchmark logs are not uniformly comparable:
- V1 shows a substantial improvement for the higher-dimensional case (roughly 1.91×–1.97×), but the reported “optimized” implementation is much slower in several simple/same-shape cases.
- V2 shows a roughly 2.04× improvement for its higher-dimensional specialized/precomputed cases, but large slowdowns for the “precomputed” variant in simple cases.
- V3 explicitly states that inputs, allocation and timing methodology are matched. Its library-vs-direct-reference results are close, with wins and losses generally around parity rather than a consistent improvement.

Do not claim a general broadcasting speedup from these logs. The V1/V2 timing differences suggest that the compared paths may include different work or overhead; treat them as diagnostic experiments, not headline performance claims. V3 is the more controlled comparison among the supplied broadcasting logs.

### 4.5 Memory, construction and other operations

Selected memory medians from `memory_benchmark.csv`:

| Elements | Data size | Operation | Median |
|---:|---:|---|---:|
| 262,144 | 2 MiB | Raw allocation | 0.083 µs |
| 262,144 | 2 MiB | Raw allocation + initialization | 87.293 µs |
| 262,144 | 2 MiB | Storage construction | 87.249 µs |
| 262,144 | 2 MiB | Tensor construction | 87.441 µs |
| 1,048,576 | 8 MiB | Raw allocation | 0.084 µs |
| 1,048,576 | 8 MiB | Raw allocation + initialization | 378.721 µs |
| 1,048,576 | 8 MiB | Storage construction | 469.078 µs |
| 1,048,576 | 8 MiB | Tensor construction | 435.976 µs |

Very small raw allocation/deallocation values are sensitive to allocator behavior and timer resolution. They should not be interpreted as the total cost of initializing or using tensor memory. The benchmark separates allocation, initialization and destruction to make this distinction visible.

Other reported medians:
- Tensor construction: 1M elements, 0.3918 ms.
- Flat indexing: 262,144 accesses, 0.3340 ms.
- Multidimensional `at({row, col})`: 262,144 accesses, 5.5587 ms.
- Two reshape operations on 262,144 elements: 0.0002 ms.
- Fill: 1M elements, 0.4426 ms.
- Sum/mean over 1M elements: about 0.635 ms each.
- Min/max over 1M elements: about 1.27 ms each.

## 5. PyTorch CPU versus CUDA results

The following table uses the **final controlled run** (`final_controlled_gpu_benchmark.csv`) and median timings. Speedup is CPU compute-only median divided by CUDA compute-only median.

### 5.1 Matrix multiplication

| Size | CPU compute (ms) | CUDA compute (ms) | CUDA end-to-end (ms) | Compute-only speedup |
|---:|---:|---:|---:|---:|
| 64 × 64 | 0.0152 | 0.2687 | 0.3705 | 0.06× |
| 128 × 128 | 0.0769 | 0.2740 | 0.4755 | 0.28× |
| 256 × 256 | 0.5302 | 0.5207 | 1.0654 | 1.02× |
| 512 × 512 | 3.9840 | 3.7812 | 5.5136 | 1.05× |
| 1024 × 1024 | 63.0601 | 13.9156 | 15.9228 | 4.53× |

For small matrices, CUDA launch and transfer overhead dominate. At 1024 × 1024, CUDA was about 4.53× faster for compute-only and about 3.96× faster end-to-end than the CPU compute-only time (63.06 / 15.92).

### 5.2 Elementwise operations

| Operation | Elements | CPU compute (ms) | CUDA compute (ms) | CUDA end-to-end (ms) | Compute-only speedup |
|---|---:|---:|---:|---:|---:|
| Addition | 1,000,000 | 1.0861 | 0.1173 | 5.1228 | 9.26× |
| Subtraction | 1,000,000 | 1.1592 | 0.1260 | 6.0450 | 9.20× |
| Multiplication | 1,000,000 | 4.0036 | 0.1238 | 5.1269 | 32.35× |
| Addition | 16,000,000 | 56.2036 | 1.5918 | 142.6512 | 35.31× |
| Subtraction | 16,000,000 | 57.7352 | 1.5938 | 139.5210 | 36.23× |
| Multiplication | 16,000,000 | 57.2397 | 1.5892 | 165.8819 | 36.02× |

These compute-only CUDA speedups apply when inputs already reside on the GPU. In the measured host-to-device-to-host workflow, the end-to-end GPU time is higher than CPU compute time for every elementwise case in this table. The benchmark is therefore evidence for GPU acceleration of device-resident workloads, not a claim that moving data to the GPU for every individual elementwise operation is faster.

## 6. Correctness and validation

The supplied logs record PASS for all listed MiniTensor matrix-multiplication benchmark configurations, elementwise benchmark paths, and transpose completion. The user also reported that all six CTest suites passed in the Colab Release build.

The final controlled PyTorch benchmark printed PASS for:
- 5 matrix multiplication sizes.
- 5 elementwise sizes × 3 operations.
- CPU-versus-CUDA output comparisons using `torch.allclose`.

These are separate validation layers: PyTorch CPU/CUDA equivalence does not by itself validate MiniTensor against PyTorch. The project’s own CTest and MiniTensor benchmark correctness checks remain the relevant MiniTensor validation.

## 7. Limitations and reproducibility notes

1. **Shared virtual machine:** Colab is virtualized and not a dedicated, fixed CPU benchmark host. Results can vary with host load and runtime allocation.
2. **Only two visible logical CPUs:** Results for 4, 8 and 16 OpenMP threads are oversubscribed experiments, not fair multi-core scaling measurements.
3. **Separate GPU experiment:** MiniTensor is benchmarked as a CPU library. PyTorch CUDA timings are a separate reference experiment and do not imply MiniTensor CUDA support.
4. **Different benchmark harnesses:** MiniTensor and PyTorch use different code paths and timing harnesses. Avoid direct claims that one library is faster than the other unless the professor's common harness is used.
5. **Broadcasting experiments:** V1 and V2 show large unexplained differences between variants; use V3 as the more controlled direct-reference comparison and avoid broad claims.
6. **Protocol pending:** Replace or supplement these measurements when the professor publishes the common benchmark test cases and measurement protocol.
7. **Repeatability:** Re-run on the same dedicated hardware, compiler, flags, thread count, input dimensions, and protocol for official comparison.

## 8. Files included in the benchmark archive

- `matmul_benchmark.csv`
- `memory_benchmark.csv`
- `elementwise_results.txt`
- `transpose_results.txt`
- `broadcast_results.txt`
- `broadcast_v2_results.txt`
- `broadcast_v3_results.txt`
- `remaining_results.txt`
- `pytorch_gpu_benchmark.csv`
- `pytorch_gpu_benchmark_v2.csv`
- `cpu_gpu_comparison.csv`
- `gpu_end_to_end.csv`
- `final_controlled_gpu_benchmark.csv`
- `system_info.txt`

## 9. Final conclusion

The current MiniTensor implementation demonstrates correct behavior across the recorded tests and has measurable benefits from transpose blocking and selected matrix-multiplication parallel configurations. SIMD/OpenMP elementwise improvements are modest on the available Colab CPU, and the broadcasting experiments do not yet support a general optimization claim.

The separate PyTorch experiment demonstrates that the Tesla T4 can substantially accelerate sufficiently large, device-resident operations, while data transfer can erase the benefit for one-off elementwise work. These results should be described as preliminary Colab measurements until repeated under the professor's official common protocol.
