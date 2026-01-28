#!/usr/bin/env python3

# BSD 3-Clause License
#
# Copyright (c) 2026, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Run fltflt arithmetic benchmarks and summarize results.
Shows performance relative to single-precision (1x baseline).
"""

import subprocess
import re
import sys
from pathlib import Path
from collections import defaultdict

def find_benchmark_executable(build_dir):
    """Find the matx_bench executable."""
    benchmark_path = build_dir / "bench" / "matx_bench"

    if benchmark_path.exists():
        return benchmark_path

    print(f"Error: Could not find matx_bench at {benchmark_path}")
    return None

def run_benchmark(executable_path, benchmark_name):
    """Run a specific benchmark and capture output."""
    print(f"Running benchmark: {benchmark_name}")

    try:
        result = subprocess.run(
            [str(executable_path), "--benchmark", benchmark_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"  Warning: Benchmark failed with return code {result.returncode}")
            print(f"  stderr: {result.stderr}")
            return None

        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"  Benchmark timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"  Error running benchmark: {e}")
        return None

def parse_time_value(time_str):
    """Parse time string like '668.707 us' or '6.785 ms' and convert to milliseconds."""
    time_str = time_str.strip()

    # Match number and unit
    match = re.match(r'([\d.]+)\s*(us|ms|ns|s)', time_str)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)

    # Convert to milliseconds
    if unit == 'us':
        return value / 1000.0
    elif unit == 'ms':
        return value
    elif unit == 'ns':
        return value / 1_000_000.0
    elif unit == 's':
        return value * 1000.0
    else:
        return value

def parse_benchmark_output(output):
    """
    Parse the table format output from nvbench.

    Expected format:
    |      T       |   Array Size    | ... |  GPU Time  | ...
    |--------------|-----------------|-----|------------|-----
    |          F32 | ...                   | 668.707 us | ...
    |          F64 | ...                   |  47.650 ms | ...
    | matx::fltflt | ...                   |   6.785 ms | ...
    """
    results = {}
    lines = output.strip().split('\n')

    # Find the header line to locate GPU Time column
    gpu_time_col_idx = None
    for i, line in enumerate(lines):
        if '|' in line and 'GPU Time' in line:
            # Split by | and find GPU Time column index
            cols = [col.strip() for col in line.split('|')]
            for j, col in enumerate(cols):
                if 'GPU Time' in col:
                    gpu_time_col_idx = j
                    break
            break

    if gpu_time_col_idx is None:
        print("  Warning: Could not find GPU Time column in output")
        return results

    # Parse data rows
    for line in lines:
        if '|' not in line:
            continue

        # Skip header and separator lines
        if 'GPU Time' in line or '---' in line or 'T' in line.split('|')[1]:
            continue

        cols = [col.strip() for col in line.split('|')]

        if len(cols) <= gpu_time_col_idx:
            continue

        # Get type column (usually first or second)
        type_col = cols[1] if len(cols) > 1 else None

        if not type_col:
            continue

        # Map type names
        if 'F32' in type_col:
            precision = 'float'
        elif 'F64' in type_col:
            precision = 'double'
        elif 'fltflt' in type_col:
            precision = 'fltflt'
        else:
            continue

        # Extract GPU time
        gpu_time_str = cols[gpu_time_col_idx]
        gpu_time_ms = parse_time_value(gpu_time_str)

        if gpu_time_ms is not None:
            results[precision] = gpu_time_ms

    return results

def calculate_relative_performance(results):
    """
    Calculate performance relative to float (single-precision).
    float = 1.0x (baseline)
    Higher values mean slower (took more time relative to float)
    """
    relative = {}

    for bench_name, timings in results.items():
        if 'float' not in timings:
            print(f"Warning: No float baseline for {bench_name}, skipping")
            continue

        float_time = timings['float']
        relative[bench_name] = {}

        for precision, time_value in timings.items():
            # Relative slowdown: how many times slower than float
            relative[bench_name][precision] = time_value / float_time

    return relative

def print_summary(results, relative):
    """Print a formatted summary table."""
    print("\n")
    print("=" * 80)
    print("FLTFLT BENCHMARK SUMMARY")
    print("=" * 80)
    print()
    print("Performance relative to single-precision (float = 1.0x baseline)")
    print("Higher values indicate slower performance")
    print()

    # Print header
    print(f"{'Benchmark':<15} {'float':<12} {'double':<12} {'fltflt':<12} {'fltflt vs dbl':<15}")
    print("-" * 66)

    # Order benchmarks
    bench_order = ['add', 'sub', 'mul', 'div', 'sqrt', 'abs', 'fma', 'madd']

    for bench in bench_order:
        if bench not in relative:
            continue

        rel = relative[bench]
        timings = results[bench]

        # Get values with defaults
        float_rel = rel.get('float', 1.0)
        double_rel = rel.get('double', None)
        fltflt_rel = rel.get('fltflt', None)

        # Calculate fltflt speedup vs double (double_time / fltflt_time)
        fltflt_vs_double = None
        if 'double' in timings and 'fltflt' in timings:
            fltflt_vs_double = timings['double'] / timings['fltflt']

        # Format output
        float_str = f"{float_rel:.2f}x"
        double_str = f"{double_rel:.2f}x" if double_rel is not None else "N/A"
        fltflt_str = f"{fltflt_rel:.2f}x" if fltflt_rel is not None else "N/A"
        speedup_str = f"{fltflt_vs_double:.2f}x" if fltflt_vs_double is not None else "N/A"

        print(f"{bench:<15} {float_str:<12} {double_str:<12} {fltflt_str:<12} {speedup_str:<15}")

    print()
    print("-" * 80)
    print("Raw timings (milliseconds):")
    print()
    print(f"{'Benchmark':<15} {'float':<12} {'double':<12} {'fltflt':<12} {'fltflt vs dbl':<15}")
    print("-" * 66)

    for bench in bench_order:
        if bench not in results:
            continue

        timings = results[bench]

        float_time = timings.get('float', None)
        double_time = timings.get('double', None)
        fltflt_time = timings.get('fltflt', None)

        # Calculate fltflt speedup vs double
        fltflt_vs_double = None
        if double_time is not None and fltflt_time is not None:
            fltflt_vs_double = double_time / fltflt_time

        float_str = f"{float_time:.3f}" if float_time is not None else "N/A"
        double_str = f"{double_time:.3f}" if double_time is not None else "N/A"
        fltflt_str = f"{fltflt_time:.3f}" if fltflt_time is not None else "N/A"
        speedup_str = f"{fltflt_vs_double:.2f}x" if fltflt_vs_double is not None else "N/A"

        print(f"{bench:<15} {float_str:<12} {double_str:<12} {fltflt_str:<12} {speedup_str:<15}")

    print("=" * 80)

def main():
    # Find MatX build directory
    script_dir = Path(__file__).parent

    # Try common build directory locations
    possible_build_dirs = [
        script_dir / "build",
        script_dir / "repos" / "MatX" / "build",
        script_dir / "../build",
        script_dir / "../../build",
    ]

    build_dir = None
    for bd in possible_build_dirs:
        if bd.exists():
            build_dir = bd
            break

    if build_dir is None:
        print("Error: Could not find MatX build directory")
        print("Please run this script from the MatX source directory or specify build path")
        sys.exit(1)

    print(f"Using build directory: {build_dir}")

    # Find benchmark executable
    benchmark_exe = find_benchmark_executable(build_dir)

    if benchmark_exe is None:
        sys.exit(1)

    print(f"Found benchmark: {benchmark_exe}")
    print()

    # List of benchmarks to run
    benchmarks = ['add', 'sub', 'mul', 'div', 'sqrt', 'abs', 'fma', 'madd']

    all_results = {}

    # Run each benchmark
    for bench in benchmarks:
        bench_name = f"fltflt_bench_{bench}"
        print(f"\n{'=' * 80}")
        output = run_benchmark(benchmark_exe, bench_name)

        if output is None:
            print(f"  Skipping {bench} due to error")
            continue

        # Parse results
        results = parse_benchmark_output(output)

        if not results:
            print(f"  Warning: Could not parse results for {bench}")
            print("  Raw output:")
            print(output)
            continue

        all_results[bench] = results
        print(f"  Parsed: {', '.join([f'{k}={v:.3f}ms' for k, v in results.items()])}")

    print(f"\n{'=' * 80}")

    if not all_results:
        print("\nError: No benchmark results collected")
        sys.exit(1)

    print(f"\nSuccessfully collected results for {len(all_results)} benchmarks")

    # Calculate relative performance
    relative = calculate_relative_performance(all_results)

    # Print summary
    print_summary(all_results, relative)

if __name__ == "__main__":
    main()
