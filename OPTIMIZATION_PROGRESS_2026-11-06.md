# MiniTensor — pending CPU optimization work completed locally

Date: 6 November 2026 (India time).

## State and preservation

Original GitHub/main commit: `7f0e891050128fd31aed4f5afda9769af054ff13`.
Working branch: `continue/matmul-validation`. All changes and commits are local.
Nothing has been pushed, merged, or modified on GitHub; local main remains at
the original commit.

The first continuation commit, `2be3eae`, improved parallel matrix multiplication
and added independent matrix tests. This document records the subsequent pass
through the pending optimization ledger. Existing README, BENCHMARK_REPORT,
OPTIMIZATIONS, original benchmark sources, and historical results are preserved.

The original Windows stash is still an external recovery item. GitHub clones
do not contain it. No stash was applied, removed or overwritten in this session.
Inspect it in the original Windows checkout before reconciling README/report
changes; a clean clone cannot establish whether those changes were recovered.

## Code changes retained

1. **Matrix multiplication:** retain the four-row reuse kernel from `2be3eae`.
   Rows remain independent across workers. Its one-row fallback preserves
   useful parallelism when a matrix has few rows.
2. **Output memory traffic:** private tagged Storage/Tensor constructors and an
   Ops.cpp-only access helper avoid initializing buffers that a kernel completely
   overwrites. Elementwise operations, broadcasting and transpose use this path.
   Public constructors still initialize all elements. Matrix multiplication
   still initializes its accumulation buffer to zero. Allocation alignment,
   overflow checks, ownership and copy/move behavior remain intact.
3. **General broadcasting:** normalize missing and singleton axes to zero
   strides once, coalesce compatible contiguous suffixes, and advance outer
   coordinates with an odometer. Inner runs avoid division/modulo per element.
   Add scalar and equal-element-count paths, including leading singleton axes.
   Existing 2D row/column paths are retained. Subtraction operand order is kept.
4. **Short parallel elementwise calls:** use the sequential SIMD implementation
   below 65,536 elements, or when only one worker is requested. This avoids
   OpenMP team overhead for small arrays.
5. **Transpose:** keep the existing tile size 16, but write contiguous output
   elements inside each tile. Test 8/16/32/64 alternatives rather than assuming
   the historical Xeon winner is best on this machine.
6. **Reductions:** cache element counts. Sum/mean use parallel SIMD for at least
   262,144 elements when more than one worker is requested. Min/max scan ordered
   contiguous chunks and combine in chunk order, retaining strict-comparison
   NaN and signed-zero behavior. No fast-math flag was added. Floating-point sum
   order can differ; tolerance-based correctness applies, not bitwise identity.
7. **Indexing allocations:** initializer-list indexing passes the existing
   coordinates directly to a shared pointer/count bounds-checking routine,
   removing the temporary vector created by each `at({row,column})` call.
   Vector indexing, const access, rank checks and bounds checks are preserved.

These are implementations of the pending CPU work, not new GPU, view/slicing,
in-place, or new-element-type APIs.

## Measurements and their limits

Host: Apple M4 MacBook Air, 16 GB RAM, 10 CPU cores (4 performance, 6 efficiency),
macOS 26.5.2 arm64. Compiler: GCC 16.1.0. OpenMP: 5.2. CMake: 4.4.4.
Release: `-std=c++17 -O3 -DNDEBUG -flto=auto -fopenmp`, native tuning off.
No CPU affinity pinning or hardware counters were available in this session.

The additional operations are compared with the preserved library at `2be3eae`.
That baseline already contains the accepted matrix optimization; the other
operation implementations are unchanged from the original `7f0e891`.
Both libraries run the same focused driver, compiler and optimization flags.
Three paired rounds run serially: baseline/candidate, candidate/baseline,
baseline/candidate. These are development measurements on an interactive laptop.

Each case has 3 warmup batches and 10 measured batches. For at most 4,096
elements, a batch contains 256 calls, because individual calls can fall below
the clock's resolution. Other cases use one call. CSVs explicitly record
`BatchCalls`; times and samples are normalized to milliseconds per call.
The tensor output buffer escapes through a compiler barrier to prevent unused
stores from being optimized away. Input setup and full correctness scans are
outside timing. Allocation, initialization when needed, computation and output
destruction are included. Indexing cases time the full 512² traversal.

Tables show the **median of three run medians**, not confidence intervals or
pooled sample medians. Raw per-run averages, medians, all ten samples, run ranges,
and metadata are retained.

| Operation / workload | Threads | Baseline ms | Optimized ms | Less runtime |
|---|---:|---:|---:|---:|
| Add, 4,000,000 elements | 1 | 1.5995 | 1.0620 | 33.60% |
| Add broadcast, full 1024² | 1 | 0.3535 | 0.2260 | 36.07% |
| Add broadcast, 4096×257 plus row | 1 | 0.2910 | 0.1680 | 42.27% |
| Add broadcast, 4096×257 plus column | 1 | 0.2885 | 0.1620 | 43.85% |
| Add broadcast, 1M plus scalar | 1 | 1.1135 | 0.1390 | 87.52% |
| Add broadcast, 16×257×129 plus last-axis vector | 1 | 1.2135 | 0.0875 | 92.79% |
| Add broadcast, (8,1,257,1) plus (1,16,1,33) | 1 | 3.9845 | 0.1810 | 95.46% |
| Add broadcast, leading singleton dimensions | 1 | 2.9515 | 0.2285 | 92.26% |
| Transpose, 512² | 1 | 0.1170 | 0.0870 | 25.64% |
| Transpose, 2048² | 1 | 11.6235 | 4.6415 | 60.07% |
| Transpose, 513×517 | 1 | 0.1310 | 0.0830 | 36.64% |
| Initializer-list indexing, full 512² traversal | 1 | 5.6165 | 1.0400 | 81.48% |
| Sum, 4,000,000 elements | 2 | 1.1745 | 0.6155 | 47.59% |
| Min, 4,000,000 elements | 2 | 3.9320 | 0.9930 | 74.75% |
| Max, 4,000,000 elements | 2 | 3.8855 | 0.9965 | 74.35% |

The earlier matrix pass, measured separately against `7f0e891`, reduced 1024²
two-thread median-of-run-medians from 77.3415 to 57.3880 ms (25.80% less time).
Its raw evidence remains in the sibling `paired/` directory and its full report
is MATMUL_PROGRESS_2026-11-06.md. Percentages across stages must not be added.

### Regressions and unchanged areas

There is no claim that every operation improved. In the final paired runs,
one-thread 4M sum was about 2.4% slower; the short 31-element add case about 10%
slower in absolute times below 0.0002 ms per call. Copy at 1M was about 14% slower
(0.1320 vs 0.1505 ms), and at 4M about 8.4% slower. Copy's implementation is
unchanged, run ranges overlap, and an earlier complete paired pass showed a
different direction at 4M. These observations are retained as variability and
potential follow-up concerns, not silently removed or recast as gains.

No allocator or copy speedup is claimed. The memory optimization is fewer output
writes and fewer indexing allocations. The selected improvements are clear for
the listed larger workloads; they need repetition on the evaluation machine.
These Mac figures cannot be compared directly with the old Xeon/Colab figures,
and no single whole-library percentage or competition ranking is established.

## Remaining matrix experiments: measured, rejected

`benchmark_matmul_experiments` checked 84 configurations: sizes 257, 512, 1024;
tiles 32, 64, 96, 128, 192, 256; sequential/parallel blocked variants; one and two
threads; runtime static/dynamic/guided schedules; and B transposition followed
by SIMD dot products. All correctness checks passed; each case used 3+10 calls
and reports mean, median, GFLOPS and raw samples.

At 1024²/two threads:

| Experiment | Median ms |
|---|---:|
| Current four-row parallel kernel | 56.7460 |
| B-transposed dot products, including packing and temporary lifetime | 193.4525 |
| Runtime static, chunk 1 | 71.8025 |
| Runtime dynamic, chunk 1 | 71.9005 |
| Runtime guided, chunk 1 | 58.1490 |
| Best parallel-blocked tile in this sweep: 96 | 99.1735 |

These are exploratory single-run comparisons, not three-pair results. They do
not justify adopting packing, changing production scheduling, or changing a
universal blocked default. The production compile-time static schedule and
blocked APIs/defaults are retained. Packing costs and additional temporary
memory are included, rather than hiding them in a compute-only number.

The SIMD-only matrix candidate had previously shown mixed outcomes and was
superseded by the four-row approach. Compiler vectorization diagnostics were
also collected for the preserved baseline and current code using the actual
GCC/Release/LTO flags. They report 16-byte vectorized loops in elementwise,
broadcast, matrix and sum kernels, plus missed loops. This is compiler evidence,
not an isolated SIMD speedup claim; scalar min/max comparisons remain deliberate.

## Correctness and checks completed

- All **8 CTest suites** pass in Release.
- All 8 pass with AddressSanitizer and UndefinedBehaviorSanitizer, `-O1 -g`, LTO
  off. Leak detection was disabled; no leak-detector or race-detector claim.
- Independent matrix oracle: 133 shapes and **4,788** variant/thread/tile checks,
  plus shared inputs, unchanged inputs, invalid ranks/shapes and zero tiles.
- New operation reference suite: 225 fixed broadcast shape pairs, 200 randomized
  pairs, and 5 additional shape/threshold pairs. It checks full outputs with
  independently formed coordinates, not only checksums.
- Same-shape arithmetic, shared inputs, alignment, empty/scalar tensors, bounds
  checking, transpose rectangles/fringes, ordinary initialization (including
  empty-brace initial values), small/large reduction thresholds, NaNs, infinities,
  cancellation tolerance, signed zeros, and threads 1/2/4/8 are covered.
- Final original matrix benchmark: **145/145** configurations pass.
- Final paired operation runs: **852/852** configurations pass (142 × 2 × 3).
- Matrix experiments: **84/84** pass.
- Existing memory benchmark: 45 rows, with separate allocation/deallocation,
  initialization, construction and destruction categories, using 3+10. Its CSV
  does not contain a correctness-status field; construction/copy/fill behavior
  is covered by tests instead.

No benchmark overlapped sanitizer execution or compilation during the final
paired comparisons. Final paired data and library/binary SHA-256 fingerprints
are recorded under `results/macos_2026-11-06/remaining-ops/`.

## Protocol audit

The old matrix and memory harnesses already use 3 warmups/10 measurements and
mean/median. Several historical exploration programs do not: elementwise uses
2+5, transpose/broadcast V1/V2 and remaining operations use 2+7 by default,
broadcast V3 uses 2+9. Several report only a median. Their old results remain
historical; they must not be relabeled as final 3+10 results.

The new focused operation harness supplies consistent statistics and raw samples
for arithmetic, broadcasting, transpose, reductions, indexing, construction and
copying, with the batching disclosure above. It does not prove compliance with
an unavailable professor-prescribed common workload matrix.

`scripts/verify_ubuntu.sh` prepares a fresh Ubuntu 22.04+ Release build, records
compiler/flags/CPU affinity/quota and oversubscribed thread counts, checks all
tests, runs matrix/operation/memory/experiment harnesses, and runs instrumented
tests separately. It optionally records aggregate perf counters when available.
It performs no installation, upload, Git commit, or push.

The script and embedded Python were syntax checked. **It was not executed on
Ubuntu in this Mac session.** Neither Linux perf nor a Linux runtime is available
locally. Counter collection and the final Ubuntu/professor run remain environment
dependent; no hardware-counter measurements or official compliance are invented.

## Status against the supplied pending ledger

| Item | Status |
|---|---|
| P01 inspect source | Done against real source |
| P02 loop order | Existing i-k-j confirmed; no redundant reorder |
| P03 tile sweep | Extended and measured; keep existing defaults |
| P04 B packing/transposition | Measured end to end; rejected locally |
| P05 vectorization diagnostics | Collected; no general SIMD percentage claimed |
| P06 OpenMP tuning | Thread/scaling results recorded; alternative schedules tested |
| P07 broadcasting | Implemented cached strides, coalescing and scalar paths |
| P08 reductions | Implemented and checked, including special values |
| P09 allocations/temporaries | Removed output zeroing and per-access index vectors where safe |
| P10 official audit | Historical mismatches identified; consistent harness/Ubuntu runner ready; official workload/environment verification still pending |
| P11 sanitizers/edges | Done locally: 8 suites and independent reference checks |
| P12 counters | Unavailable locally; collection hook prepared for Ubuntu |

## Reproduce without touching main or GitHub

Build/test the checked-out optimization branch:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build-release -j2
ctest --test-dir build-release --output-on-failure
```

For paired operation comparisons, snapshot **all** baseline headers and source,
not only Ops.cpp. Use a fresh sibling directory:

```bash
mkdir -p ../ops-evaluation/baseline
git archive 2be3eae include src | tar -x -C ../ops-evaluation/baseline
g++ -std=c++17 -O3 -DNDEBUG -flto=auto -fopenmp \
    -I../ops-evaluation/baseline/include \
    ../ops-evaluation/baseline/src/Shape.cpp \
    ../ops-evaluation/baseline/src/Storage.cpp \
    ../ops-evaluation/baseline/src/Tensor.cpp \
    ../ops-evaluation/baseline/src/Ops.cpp \
    benchmarks/benchmark_ops_focus.cpp -o ../ops-evaluation/baseline-ops
python3 scripts/compare_matmul_runs.py \
    ../ops-evaluation/baseline-ops build-release/benchmark_ops_focus \
    ../ops-evaluation/paired-results --ops
```

Use `7f0e891` instead of `2be3eae` and the matrix focused driver to compare the
complete matrix change against the original. A source ZIP has no Git objects:
use an existing clone or the supplied Git bundle for `git archive` commands.
On this Mac, the compiler was `/opt/homebrew/bin/g++-16` instead of `g++`.

On Ubuntu with GCC, CMake, Python and OpenMP installed:

```bash
bash scripts/verify_ubuntu.sh ../minitensor-ubuntu-results
```

Result directories must be unused. No source files or previous results are
overwritten. Check the professor's exact operation/input matrix against the
recorded cases before labeling any run official.

## Still requiring the original environment

1. Safely inspect/reconcile the Windows README/report stash.
2. Run the prepared verification on Ubuntu/evaluation hardware and obtain the
   professor's exact workload matrix. Repeat questionable copy/tiny cases there.
3. Collect hardware counters if that host permits it.
4. Review these local changes before granting any permission to push to GitHub.
