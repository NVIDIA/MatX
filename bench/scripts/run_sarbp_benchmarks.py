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
Run SAR backprojection benchmarks and summarize results.
Computes gigabackprojections per second (Gproj/s) for each benchmark variant.
"""

import subprocess
import re
import sys
import argparse
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
            timeout=600  # 10 minute timeout for larger benchmarks
        )

        if result.returncode != 0:
            print(f"  Warning: Benchmark failed with return code {result.returncode}")
            print(f"  stderr: {result.stderr}")
            return None

        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"  Benchmark timed out after 10 minutes")
        return None
    except Exception as e:
        print(f"  Error running benchmark: {e}")
        return None

def parse_time_value(time_str):
    """Parse time string like '668.707 us' or '6.785 ms' and convert to seconds."""
    time_str = time_str.strip()

    # Match number and unit
    match = re.match(r'([\d.]+)\s*(us|ms|ns|s)', time_str)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)

    # Convert to seconds
    if unit == 'us':
        return value / 1_000_000.0
    elif unit == 'ms':
        return value / 1_000.0
    elif unit == 'ns':
        return value / 1_000_000_000.0
    elif unit == 's':
        return value
    else:
        return value

def parse_benchmark_output(output):
    """
    Parse the table format output from nvbench for sarbp benchmarks.

    Expected format:
    | Problem Size | ... |  GPU Time  | ...
    |--------------|-----|------------|-----
    |         1000 | ... | 123.456 ms | ...
    |         2000 | ... | 987.654 ms | ...
    """
    results = {}
    lines = output.strip().split('\n')

    # Find the header line to locate GPU Time and Problem Size columns
    gpu_time_col_idx = None
    problem_size_col_idx = None
    for i, line in enumerate(lines):
        if '|' in line and 'GPU Time' in line:
            # Split by | and find column indices
            cols = [col.strip() for col in line.split('|')]
            for j, col in enumerate(cols):
                if 'GPU Time' in col:
                    gpu_time_col_idx = j
                elif 'Problem Size' in col:
                    problem_size_col_idx = j
            break

    if gpu_time_col_idx is None:
        print("  Warning: Could not find GPU Time column in output")
        return results

    if problem_size_col_idx is None:
        print("  Warning: Could not find Problem Size column in output")
        return results

    # Parse data rows
    for line in lines:
        if '|' not in line:
            continue

        # Skip header and separator lines
        if 'GPU Time' in line or '---' in line or 'Problem Size' in line:
            continue

        cols = [col.strip() for col in line.split('|')]

        if len(cols) <= max(gpu_time_col_idx, problem_size_col_idx):
            continue

        # Get problem size
        problem_size_str = cols[problem_size_col_idx]
        try:
            problem_size = int(problem_size_str)
        except ValueError:
            continue

        # Extract GPU time
        gpu_time_str = cols[gpu_time_col_idx]
        gpu_time_s = parse_time_value(gpu_time_str)

        if gpu_time_s is not None:
            results[problem_size] = gpu_time_s

    return results

def calculate_gproj_per_sec(problem_size, time_seconds):
    """
    Calculate gigabackprojections per second.

    Each sarbp execution computes:
    num_pulses * image_width * image_height backprojection operations

    For our benchmarks: all dimensions = problem_size
    So: operations = problem_size^3
    """
    operations = problem_size ** 3
    giga_operations = operations / 1e9
    gproj_per_sec = giga_operations / time_seconds
    return gproj_per_sec

def print_summary(all_results):
    """Print a formatted summary table."""
    print("\n")
    print("=" * 100)
    print("SAR BACKPROJECTION BENCHMARK SUMMARY")
    print("=" * 100)
    print()
    print("Performance in Gigabackprojections per second (Gproj/s)")
    print("Higher values indicate better performance")
    print()
    print(f"Note: Operations = num_pulses × image_width × image_height = problem_size³")
    print()

    # Print detailed results for each variant
    variants = ['float', 'double', 'mixed', 'fltflt']

    for variant in variants:
        if variant not in all_results:
            continue

        print(f"\n{variant.upper()} Precision:")
        print("-" * 80)
        print(f"{'Problem Size':<15} {'Operations':<18} {'Time (ms)':<15} {'Gproj/s':<15}")
        print("-" * 80)

        for problem_size in sorted(all_results[variant].keys()):
            time_s = all_results[variant][problem_size]
            time_ms = time_s * 1000.0
            operations = problem_size ** 3
            gproj_s = calculate_gproj_per_sec(problem_size, time_s)

            print(f"{problem_size:<15} {operations:<18,} {time_ms:<15.3f} {gproj_s:<15.3f}")

    # Print comparative summary
    print("\n")
    print("=" * 100)
    print("COMPARATIVE SUMMARY (Gproj/s)")
    print("=" * 100)
    print()

    # Get all problem sizes
    all_problem_sizes = set()
    for variant_results in all_results.values():
        all_problem_sizes.update(variant_results.keys())
    all_problem_sizes = sorted(all_problem_sizes)

    # Print header
    header = f"{'Problem Size':<15}"
    for variant in variants:
        if variant in all_results:
            header += f" {variant:<15}"
    print(header)
    print("-" * 100)

    # Print data rows
    for problem_size in all_problem_sizes:
        row = f"{problem_size:<15}"
        for variant in variants:
            if variant in all_results and problem_size in all_results[variant]:
                time_s = all_results[variant][problem_size]
                gproj_s = calculate_gproj_per_sec(problem_size, time_s)
                row += f" {gproj_s:<15.3f}"
            else:
                row += f" {'N/A':<15}"
        print(row)

    # Print relative performance (relative to float)
    print("\n")
    print("=" * 100)
    print("RELATIVE PERFORMANCE (float = 1.0x baseline)")
    print("=" * 100)
    print()

    # Print header
    header = f"{'Problem Size':<15}"
    for variant in variants:
        if variant in all_results:
            header += f" {variant:<15}"
    print(header)
    print("-" * 100)

    # Print data rows
    for problem_size in all_problem_sizes:
        row = f"{problem_size:<15}"

        # Get float baseline
        float_gproj_s = None
        if 'float' in all_results and problem_size in all_results['float']:
            time_s = all_results['float'][problem_size]
            float_gproj_s = calculate_gproj_per_sec(problem_size, time_s)

        for variant in variants:
            if variant in all_results and problem_size in all_results[variant]:
                time_s = all_results[variant][problem_size]
                gproj_s = calculate_gproj_per_sec(problem_size, time_s)

                if float_gproj_s is not None and float_gproj_s > 0:
                    relative = gproj_s / float_gproj_s
                    row += f" {relative:<15.3f}"
                else:
                    row += f" {1.0:<15.3f}" if variant == 'float' else f" {'N/A':<15}"
            else:
                row += f" {'N/A':<15}"
        print(row)

    print("=" * 100)

def main():
    parser = argparse.ArgumentParser(
        description="Run SAR backprojection benchmarks and summarize results."
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="Path to the MatX build directory containing bench/matx_bench. "
             "If not specified, the current working directory is checked first, "
             "then common locations relative to the script are searched.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        metavar="VARIANT",
        help="Run only specific benchmark variants (e.g. float double). "
             "Defaults to all variants: float double mixed fltflt.",
    )
    args = parser.parse_args()

    # Find MatX build directory
    if args.build_dir is not None:
        build_dir = args.build_dir
        if not build_dir.exists():
            print(f"Error: Specified build directory does not exist: {build_dir}")
            sys.exit(1)
    else:
        # Check if the current working directory looks like a valid build directory
        # (i.e. it already contains bench/matx_bench). This lets users run the script
        # from any build directory without needing --build-dir.
        cwd = Path.cwd()
        if (cwd / "bench" / "matx_bench").exists():
            build_dir = cwd
        else:
            # Fall back to searching common locations relative to the script
            script_dir = Path(__file__).parent
            possible_build_dirs = [
                script_dir / "build",
                script_dir / "repos" / "MatX" / "build",
                script_dir / "../build",
                script_dir / "../../build",
            ]

            build_dir = None
            for bd in possible_build_dirs:
                bd_resolved = bd.resolve()
                if bd_resolved.exists() and (bd_resolved / "bench" / "matx_bench").exists():
                    build_dir = bd_resolved
                    break

            if build_dir is None:
                print("Error: Could not find MatX build directory")
                print("Try running from a build directory, or use --build-dir to specify one")
                sys.exit(1)

    print(f"Using build directory: {build_dir}")

    # Find benchmark executable
    benchmark_exe = find_benchmark_executable(build_dir)

    if benchmark_exe is None:
        sys.exit(1)

    print(f"Found benchmark: {benchmark_exe}")
    print()

    # List of SAR BP benchmark variants
    all_variants = ['float', 'double', 'mixed', 'fltflt']
    variants = args.variants if args.variants is not None else all_variants

    all_results = {}

    # Run each benchmark variant
    for variant in variants:
        bench_name = f"sarbp_{variant}"
        print(f"\n{'=' * 100}")
        output = run_benchmark(benchmark_exe, bench_name)

        if output is None:
            print(f"  Skipping {variant} due to error")
            continue

        # Parse results
        results = parse_benchmark_output(output)

        if not results:
            print(f"  Warning: Could not parse results for {variant}")
            print("  Raw output:")
            print(output)
            continue

        all_results[variant] = results

        # Print parsed results with Gproj/s
        parsed_str = ', '.join([
            f'size={size}: {time_s*1000:.3f}ms ({calculate_gproj_per_sec(size, time_s):.3f} Gproj/s)'
            for size, time_s in sorted(results.items())
        ])
        print(f"  Parsed: {parsed_str}")

    print(f"\n{'=' * 100}")

    if not all_results:
        print("\nError: No benchmark results collected")
        sys.exit(1)

    print(f"\nSuccessfully collected results for {len(all_results)} benchmark variants")

    # Print summary
    print_summary(all_results)

if __name__ == "__main__":
    main()
