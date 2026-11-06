# MiniTensor local continuation — 6 November 2026

This records the first, matrix-only stage at `2be3eae`. Later whole-library
changes and current reproduction instructions are in
`OPTIMIZATION_PROGRESS_2026-11-06.md`. For a complete original baseline after
those later changes, snapshot all source and headers; the single-Ops.cpp
baseline command below describes the earlier matrix-only stage.

## Preservation and scope

This work starts from `7f0e891050128fd31aed4f5afda9769af054ff13` in
`manmeetmanchanda10016/Diwali-Project`. It is isolated on the local branch
`continue/matmul-validation`. Nothing has been pushed or merged to GitHub.
The local `main` branch remains at the starting commit.

The original README, benchmark report, optimization report, benchmark programs,
and historical result files are preserved. New results have their own directory.
The handoff's Windows stash cannot be recovered from a GitHub clone: it still
needs inspection in the original Windows working copy. No stash was dropped,
applied or overwritten here. Before changing that working copy, inspect:

```powershell
git status
git stash list
# Inspect the matching entry; substitute its actual index.
git stash show -u --stat 'stash@{0}'
```

Do not assume that the tracked README/report contains every stashed revision.
Compare the matching stash with those files before applying it; retain the stash
until recovery is verified. This local continuation does not resolve that item.

## Implementation

`matmul_parallel()` now updates four output rows together, reusing each value
from B across four products. Each OpenMP task owns complete output rows. The
accumulation order along the inner dimension remains ascending, and the public
API and double-precision storage remain unchanged.

The inner column loop uses `omp simd`. One-row tasks are retained when four-row
groups would provide fewer tasks than `omp_get_max_threads()`. Fringe rows use
the one-row kernel. Zero-dimension products return the initialized result before
pointer arithmetic or an OpenMP parallel region.

The automatic `matmul()` dispatcher already calls this function for larger
products, so it benefits without changing its existing dispatch threshold.
Blocked implementations are unchanged.

## Correctness

- All six original CTest suites passed before modification.
- All seven suites passed after modification, including the new
  `MatmulReferenceTests`.
- The reference suite tests 133 shapes and 4,788 combinations of implementation,
  thread count and block size. It uses independently ordered i-j-k dot products
  accumulated in `long double`, signed random inputs, shape checks, explicit
  finite-value checks and absolute/relative tolerances.
- Coverage includes empty dimensions, singleton dimensions, odd rectangles,
  tile fringes, dispatch-boundary sizes, block size 1, maximum-size block values,
  unchanged inputs, shared inputs, invalid ranks/shapes and zero block sizes.
- All seven suites also passed with AddressSanitizer and UndefinedBehaviorSanitizer.
  This is an instrumented Debug build (`-O1 -g`), with LTO off. Leak checking was
  disabled (`ASAN_OPTIONS=detect_leaks=0`); this is not a leak-detector or race-detector result.
- Test logs are in `results/macos_2026-11-06/`.

## Measured results

These are new **Mac development measurements**, separate from the historical
Ubuntu/Colab results. They do not establish performance on the professor's
Ubuntu evaluation environment.

Environment: Apple M4 MacBook Air, 10 CPU cores (4 performance, 6 efficiency),
16 GB RAM, macOS 26.5.2 arm64; Homebrew GCC 16.1.0; OpenMP 5.2; CMake 4.4.4.
Release flags: `-O3 -DNDEBUG -std=c++17 -flto=auto -fopenmp`; native tuning off.
Threads were explicitly set to 1, 2, 4 and 8, with OpenMP dynamic teams disabled.
CPU affinity was not fixed. This is a shared interactive laptop, not a dedicated
benchmark host; its heterogeneous cores and scheduling can affect results.

The focused harness was compiled against the original and final implementations.
Three paired rounds ran serially in this order: baseline/candidate,
candidate/baseline, baseline/candidate. Every workload in every run used 3 warmups
and 10 measured iterations. Timing includes output allocation, initialization,
computation and destruction, while inputs and correctness checks are outside
the timer. All 120 focused configuration runs passed their correctness checks.

The table reports the **median of three run medians**, not a pooled median or a
statistical confidence interval. Each raw CSV also contains the average, median,
GFLOPS based on that run's median, and all ten samples.

| Square matrix | Threads | Original (ms) | Four-row (ms) | Runtime reduction |
|---:|---:|---:|---:|---:|
| 512 | 1 | 19.3645 | 12.9045 | 33.36% |
| 512 | 2 | 10.4070 | 6.9170 | 33.54% |
| 512 | 4 | 5.8345 | 3.7295 | 36.08% |
| 512 | 8 | 4.3740 | 2.9200 | 33.24% |
| 1024 | 1 | 143.1300 | 106.4795 | 25.61% |
| 1024 | 2 | 77.3415 | 57.3880 | 25.80% |
| 1024 | 4 | 46.0975 | 33.5335 | 27.26% |
| 1024 | 8 | 35.0800 | 24.9390 | 28.91% |

For 1024/two threads, the three baseline run medians ranged from 76.9310 to
83.5235 ms; candidate medians ranged from 56.0470 to 61.3950 ms. They did not
overlap in this experiment. This supports retaining the candidate locally, but
does not guarantee the same gain on other hardware.

The full summary also includes `(257,259) × (259,261)` and
`(4,1024) × (1024,512)`. The latter's two-thread result was about 0.68% slower
(0.2215 vs 0.2230 ms), within overlapping run ranges. Small-workload results
are noisy; there is no claim that every workload improves.

### Experiments retained as evidence

- `baseline-full/`: unchanged original implementation; all 145 configurations passed.
- `simd-only-full/`: adding only `omp simd` produced mixed timing changes and was
  superseded; all 145 configurations passed.
- `four-row-initial-full/`: four-row kernel before adding the few-row fallback;
  all 145 configurations passed. This is not the final-source benchmark.
- `paired/`: final source with the fallback, three paired focused rounds,
  20 configurations per executable per round, raw samples and aggregate summary.
- `final-full/`: original full benchmark rerun against the final source; all 145 configurations passed.

All of these live under `results/macos_2026-11-06/`. The earlier Ubuntu figures
(443.09247 ms normal and 371.8054 ms two-thread OpenMP at 1024) remain historical;
comparing them directly with these Mac timings would mix hardware and compiler
changes with the implementation change. No whole-library speedup is claimed.

## Reproduce on Ubuntu using GCC

Run from this branch's project directory. Use an unused output directory: the
comparison script deliberately refuses to overwrite existing measurements.

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build-release -j2
ctest --test-dir build-release --output-on-failure

mkdir -p ../minitensor-evaluation/baseline-source
git show 7f0e891:src/Ops.cpp > ../minitensor-evaluation/baseline-source/Ops.cpp
g++ -std=c++17 -O3 -DNDEBUG -flto=auto -fopenmp -Iinclude \
    src/Shape.cpp src/Storage.cpp src/Tensor.cpp \
    ../minitensor-evaluation/baseline-source/Ops.cpp \
    benchmarks/benchmark_matmul_focus.cpp \
    -o ../minitensor-evaluation/baseline
python3 scripts/compare_matmul_runs.py \
    ../minitensor-evaluation/baseline \
    build-release/benchmark_matmul_focus \
    ../minitensor-evaluation/paired-results

# The original full benchmark writes its CSV in the current directory.
mkdir -p ../minitensor-evaluation/full-results
(cd ../minitensor-evaluation/full-results && ../../MiniTensor/build-release/benchmark_matmul)
```

Adjust the last executable path to the actual checkout folder name (the command
above assumes `MiniTensor`). The paired baseline command is valid for this
change because Ops.cpp is the only changed library source and the headers are
unchanged from `7f0e891`. A source-only ZIP has no Git object database: run the
`git show` command in a clone that contains the original commit. The Mac run used
`/opt/homebrew/bin/g++-16` instead of `g++` and the locally installed CMake.

For instrumented checks (keep timings separate):

```bash
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=g++ -DMINITENSOR_ENABLE_LTO=OFF \
    '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer' \
    '-DCMAKE_CXX_FLAGS_DEBUG=-O1 -g'
cmake --build build-sanitize -j2
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
    ctest --test-dir build-sanitize --output-on-failure
```

## Remaining work

1. Inspect/reconcile the original Windows stash without dropping it.
2. Repeat this comparison on Ubuntu/Colab with captured compiler, flags and CPU
   metadata. Keep oversubscribed thread configurations explicitly labeled.
3. Audit every operation against the actual professor-prescribed workload matrix;
   the handoff does not supply a complete authoritative matrix.
4. Review the local patch and measurements before authorizing any GitHub push.
