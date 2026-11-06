#!/usr/bin/env bash
# Fresh Linux verification. No uploads, commits, installs or source edits.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compiler="${MINITENSOR_CXX:-g++}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
output_dir="${1:-${repo_dir}/../minitensor-ubuntu-$(date -u +%Y%m%dT%H%M%SZ)}"

python3 - <<'PY'
import platform
if platform.system() != 'Linux':
    raise SystemExit('Run this script on Ubuntu 22.04 or newer; Mac results are separate.')
from pathlib import Path
fields = dict(line.split('=',1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
version = fields.get('VERSION_ID','0').strip('"')
if fields.get('ID','').strip('"') != 'ubuntu' or tuple(map(int,version.split('.')[:2])) < (22,4):
    raise SystemExit('Ubuntu 22.04 or newer is required for this verification.')
PY
command -v "$compiler" >/dev/null
command -v cmake >/dev/null
mkdir "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"

python3 - "$output_dir/environment.json" "$compiler" "$repo_dir" <<'PY'
import datetime,json,os,platform,subprocess,sys
from pathlib import Path
def read_command(args):
    result=subprocess.run(args,capture_output=True,text=True)
    return {'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
affinity=len(os.sched_getaffinity(0))
quota=None
try:
    limit,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    if limit!='max': quota=int(limit)/int(period)
except (OSError,ValueError): pass
available=min(affinity,quota) if quota else affinity
metadata={
    'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'platform':platform.platform(),'logical_cpus':os.cpu_count(),
    'affinity_cpus':affinity,'cpu_quota':quota,'effective_cpu_limit':available,
    'oversubscribed_requested_threads':[n for n in (1,2,4,8) if n>available],
    'compiler':read_command([sys.argv[2],'--version']),
    'cmake':read_command(['cmake','--version']),
    'cpu':read_command(['lscpu']),
    'source_commit':read_command(['git','-C',sys.argv[3],'rev-parse','HEAD']),
    'source_status':read_command(['git','-C',sys.argv[3],'status','--short']),
    'OpenMP_environment':{k:v for k,v in os.environ.items() if k.startswith('OMP_')},
    'protocol':'3 warmups, 10 samples; averages and medians; correctness outside timing',
    'limitation':'The actual professor workload matrix must still be compared with these development workloads.'
}
Path(sys.argv[1]).write_text(json.dumps(metadata,indent=2)+'\n')
PY

cmake -S "$repo_dir" -B "$output_dir/build-release" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$compiler" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 2>&1 | tee "$output_dir/configure.log"
cmake --build "$output_dir/build-release" -j2 2>&1 | tee "$output_dir/build.log"
ctest --test-dir "$output_dir/build-release" --output-on-failure 2>&1 | tee "$output_dir/tests.log"
python3 - "$output_dir/build-release/compile_commands.json" <<'PY'
import json,sys
for entry in json.load(open(sys.argv[1])):
    command=entry.get('command',' '.join(entry.get('arguments',[])))
    if '/src/' in entry['file'] and ('-O3' not in command or '-DNDEBUG' not in command):
        raise SystemExit('Missing required Release flags for '+entry['file'])
PY

mkdir "$output_dir/matmul" "$output_dir/memory"
(cd "$output_dir/matmul" && "$output_dir/build-release/benchmark_matmul" > benchmark.log 2>&1)
(cd "$output_dir/memory" && "$output_dir/build-release/benchmark_memory" > benchmark.log 2>&1)
"$output_dir/build-release/benchmark_ops_focus" > "$output_dir/ops.csv" 2> "$output_dir/ops.log"
"$output_dir/build-release/benchmark_matmul_experiments" > "$output_dir/matmul-experiments.csv" 2> "$output_dir/matmul-experiments.log"

python3 - "$output_dir" <<'PY'
import csv,math,sys
from pathlib import Path
root=Path(sys.argv[1])
for name in ('matmul/matmul_benchmark.csv','ops.csv','matmul-experiments.csv'):
    rows=list(csv.DictReader((root/name).open()))
    if not rows or any(r['Correctness']!='PASS' or r['Warmups']!='3' or r['Measurements']!='10'
                       or not math.isfinite(float(r['Median_ms'])) for r in rows):
        raise SystemExit('Invalid benchmark data: '+name)
    print(name+': '+str(len(rows))+' configurations passed')
memory=list(csv.DictReader((root/'memory/memory_benchmark.csv').open()))
if not memory or not {'RawAllocation','RawDeallocation'}.issubset({r['Operation'] for r in memory}) \
        or any(r['Warmups']!='3' or r['Measurements']!='10' for r in memory):
    raise SystemExit('Missing separate allocation/deallocation data')
print('Memory benchmark saved; allocation/deallocation categories are separate.')
PY

cmake -S "$repo_dir" -B "$output_dir/build-sanitize" \
    -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER="$compiler" \
    -DMINITENSOR_ENABLE_LTO=OFF \
    '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer' \
    '-DCMAKE_CXX_FLAGS_DEBUG=-O1 -g' 2>&1 | tee "$output_dir/sanitizer-configure.log"
cmake --build "$output_dir/build-sanitize" -j2 2>&1 | tee "$output_dir/sanitizer-build.log"
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
    ctest --test-dir "$output_dir/build-sanitize" --output-on-failure 2>&1 | tee "$output_dir/sanitizer-tests.log"

# Counter access often requires host permissions; record failure without
# substituting wall-clock timings for unavailable hardware counters.
if command -v perf >/dev/null; then
    if perf stat -e cycles,instructions,cache-misses \
        "$output_dir/build-release/benchmark_ops_focus" \
        > "$output_dir/perf-run.csv" 2> "$output_dir/perf-stat.log"; then
        printf '%s\n' 'Hardware counters recorded.'
    else
        printf '%s\n' 'Hardware counters unavailable; see perf-stat.log.'
    fi
fi
printf 'Saved verification results: %s\n' "$output_dir"
