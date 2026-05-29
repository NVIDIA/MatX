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
Run a profile of MatX benchmarks via nvbench and emit per-profile summaries.

Each profile defines:
  - the per-source CMake bench executable to invoke,
  - the list of nvbench benchmark names to run,
  - an optional summary handler that walks the nvbench JSON output and
    prints a domain-specific table (e.g. fp32/fp64/fltflt slowdown ratios
    for fltflt, Gproj/s for sarbp).

Output:
  - bench_results/<profile>.json  (raw nvbench JSON, the source of truth)
  - bench_results/<profile>.md    (rendered nvbench markdown table)
  - bench_results/<profile>.csv   (raw nvbench CSV)
  - stdout                        (the profile's domain-specific summary)

Examples:
  python run_benchmarks.py                       # run every profile
  python run_benchmarks.py --profile fltflt      # run a single profile
  python run_benchmarks.py --profile sarbp -- --profile  # forward --profile to nvbench

The script does not parse markdown -- it reads nvbench's JSON output, which
is part of nvbench's stable contract.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# nvbench JSON helpers -- schema: benchmarks[].states[].summaries[].data[].value
# ---------------------------------------------------------------------------

def summary_float(state, tag, default=None):
    """Pull a float-valued summary tag out of one nvbench state."""
    for s in state.get("summaries", []):
        if s.get("tag") != tag:
            continue
        for d in s.get("data", []):
            if d.get("name") == "value":
                try:
                    return float(d.get("value"))
                except (TypeError, ValueError):
                    return default
    return default


def axis_value(state, name):
    """Pull a per-state axis value (int or string)."""
    for a in state.get("axis_values", []):
        if a.get("name") == name:
            v = a.get("value")
            try:
                return int(v)
            except (TypeError, ValueError):
                return v
    return None


def states_for_benchmark(data, bench_name):
    """Yield all states for a given benchmark name from a parsed JSON file."""
    for b in data.get("benchmarks", []):
        if b.get("name") == bench_name:
            yield from b.get("states", [])


GPU_TIME_TAG = "nv/cold/time/gpu/mean"  # seconds


def fmt_time(seconds):
    """Format seconds in auto-scaled units."""
    if seconds is None:
        return "N/A"
    if seconds < 1e-6:
        return f"{seconds * 1e9:.3f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.3f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds:.3f} s"


# ---------------------------------------------------------------------------
# fltflt summary: per-op fp32 / fp64 / fltflt slowdown ratios.
# ---------------------------------------------------------------------------

# Map from nvbench type-axis "input_string" to a friendly precision label.
_PRECISION_FROM_AXIS = {
    "F32": "float",
    "F64": "double",
    "matx::fltflt": "fltflt",
}


def _fltflt_pick_time(states, *, prefer=None):
    """For a list of states (one per type), return {precision: gpu_time_seconds}.

    `prefer`, if given, is a dict of axis_name -> value that selects the row
    used when the bench has additional axes (e.g. {Blocks: 1, Chain Length:
    16384} for the latency bench). Without `prefer`, the first matching row
    is taken (typical for benches with only the type axis).
    """
    out = {}
    for st in states:
        type_str = axis_value(st, "T")
        prec = _PRECISION_FROM_AXIS.get(type_str)
        if prec is None or prec in out:
            continue
        if prefer is not None:
            ok = True
            for k, v in prefer.items():
                if axis_value(st, k) != v:
                    ok = False
                    break
            if not ok:
                continue
        t = summary_float(st, GPU_TIME_TAG)
        if t is not None:
            out[prec] = t
    return out


def summarize_fltflt(json_path):
    """Walk nvbench JSON for the fltflt profile and emit a slowdown table."""
    with open(json_path) as f:
        data = json.load(f)

    # (display name, nvbench bench name, optional axis filter for picking a row).
    rows = [
        ("add_throughput",  "fltflt_bench_add_throughput", None),
        ("add_latency",     "fltflt_bench_add_latency",
            {"Blocks": 1, "Chain Length": 16384}),
        ("sub",             "fltflt_bench_sub",            None),
        ("mul",             "fltflt_bench_mul",            None),
        ("div",             "fltflt_bench_div",            None),
        ("sqrt",            "fltflt_bench_sqrt",           None),
        ("sqrt_fast",       "fltflt_bench_sqrt_fast",      None),
        ("norm3d",          "fltflt_bench_norm3d",         None),
        ("abs",             "fltflt_bench_abs",            None),
        ("fma",             "fltflt_bench_fma",            None),
        ("fma_approx",      "fltflt_bench_fma_approx",     None),
        ("madd",            "fltflt_bench_madd",           None),
        ("round",           "fltflt_bench_round",          None),
        ("trunc",           "fltflt_bench_trunc",          None),
        ("floor",           "fltflt_bench_floor",          None),
        ("fmod",            "fltflt_bench_fmod",           None),
        ("cast2dbl",        "fltflt_bench_cast2dbl",       None),
        ("cast2fltflt",     "fltflt_bench_cast2fltflt",    None),
    ]

    print()
    print("=" * 86)
    print("FLTFLT BENCHMARK SUMMARY")
    print("=" * 86)
    print("float / double / fltflt: slowdown vs float (lower is better, float = 1.0x).")
    print("dbl/fltflt:              speedup of fltflt over double (higher is better).")
    print()
    print(f"{'Benchmark':<18}{'float':>12}{'double':>12}{'fltflt':>12}{'dbl/fltflt':>14}")
    print("-" * 86)

    for label, bench, prefer in rows:
        states = list(states_for_benchmark(data, bench))
        if not states:
            continue
        t = _fltflt_pick_time(states, prefer=prefer)
        f = t.get("float")
        d = t.get("double")
        ff = t.get("fltflt")
        if f is None:
            continue

        def slow(x):
            return f"{x / f:.2f}x" if x is not None else "N/A"

        # Speedup framing: bigger means fltflt wins more vs double.
        speedup_vs_dbl = (
            f"{d / ff:.2f}x" if (d is not None and ff is not None and ff > 0) else "N/A"
        )
        print(f"{label:<18}{slow(f):>12}{slow(d):>12}{slow(ff):>12}{speedup_vs_dbl:>14}")

    print()
    print("Raw GPU times (cold mean):")
    print(f"{'Benchmark':<18}{'float':>14}{'double':>14}{'fltflt':>14}")
    print("-" * 86)
    for label, bench, prefer in rows:
        states = list(states_for_benchmark(data, bench))
        if not states:
            continue
        t = _fltflt_pick_time(states, prefer=prefer)
        print(f"{label:<18}{fmt_time(t.get('float')):>14}"
              f"{fmt_time(t.get('double')):>14}{fmt_time(t.get('fltflt')):>14}")
    print("=" * 86)


# ---------------------------------------------------------------------------
# sarbp summary: Gproj/s = problem_size**3 / time, plus relative table.
# ---------------------------------------------------------------------------

def _sarbp_results(data):
    """Return {variant: {problem_size: gpu_time_seconds}}."""
    out = {}
    for variant in ("float", "double", "mixed", "fltflt"):
        bench_name = f"sarbp_{variant}"
        per_size = {}
        for st in states_for_benchmark(data, bench_name):
            ps = axis_value(st, "Problem Size")
            t = summary_float(st, GPU_TIME_TAG)
            if ps is not None and t is not None:
                per_size[ps] = t
        if per_size:
            out[variant] = per_size
    return out


def _gproj_per_sec(problem_size, time_s):
    return (problem_size ** 3) / 1e9 / time_s


def summarize_sarbp(json_path):
    with open(json_path) as f:
        data = json.load(f)

    results = _sarbp_results(data)
    if not results:
        print("(sarbp: no results)")
        return

    variants = [v for v in ("float", "double", "mixed", "fltflt") if v in results]
    all_sizes = sorted({s for r in results.values() for s in r})

    print()
    print("=" * 90)
    print("SAR BACKPROJECTION BENCHMARK SUMMARY")
    print("=" * 90)
    print("Gigabackprojections per second (Gproj/s) -- operations = problem_size^3.")
    print()

    print(f"{'Problem Size':<14}" + "".join(f"{v:>14}" for v in variants))
    print("-" * 90)
    for ps in all_sizes:
        row = f"{ps:<14}"
        for v in variants:
            t = results[v].get(ps)
            row += f"{_gproj_per_sec(ps, t):>14.3f}" if t else f"{'N/A':>14}"
        print(row)

    if "float" in results:
        print()
        print("Relative throughput (float = 1.0x):")
        print(f"{'Problem Size':<14}" + "".join(f"{v:>14}" for v in variants))
        print("-" * 90)
        for ps in all_sizes:
            row = f"{ps:<14}"
            f_t = results["float"].get(ps)
            f_g = _gproj_per_sec(ps, f_t) if f_t else None
            for v in variants:
                t = results[v].get(ps)
                if t is None or f_g is None:
                    row += f"{'N/A':>14}"
                else:
                    row += f"{_gproj_per_sec(ps, t) / f_g:>13.3f}x"
            print(row)
    print("=" * 90)


# ---------------------------------------------------------------------------
# Profile registry -- one entry per logical bench family.
# ---------------------------------------------------------------------------

PROFILES = {
    "fltflt": {
        "exe_stems": ["bench_00_misc_fltflt_arithmetic", "matx_bench"],
        "benchmarks": [
            "fltflt_bench_add_throughput",
            "fltflt_bench_add_latency",
            "fltflt_bench_sub",
            "fltflt_bench_mul",
            "fltflt_bench_div",
            "fltflt_bench_sqrt",
            "fltflt_bench_sqrt_fast",
            "fltflt_bench_norm3d",
            "fltflt_bench_abs",
            "fltflt_bench_fma",
            "fltflt_bench_fma_approx",
            "fltflt_bench_madd",
            "fltflt_bench_round",
            "fltflt_bench_trunc",
            "fltflt_bench_floor",
            "fltflt_bench_fmod",
            "fltflt_bench_cast2dbl",
            "fltflt_bench_cast2fltflt",
        ],
        "summary": summarize_fltflt,
        # Inherited from the deleted run_fltflt_benchmarks.py.
        "timeout_seconds": 300,
    },
    "sarbp": {
        "exe_stems": ["bench_00_transform_sarbp", "matx_bench"],
        "benchmarks": ["sarbp_float", "sarbp_double", "sarbp_mixed", "sarbp_fltflt"],
        "summary": summarize_sarbp,
        # Inherited from the deleted run_sarbp_benchmarks.py; sarbp at the
        # default Problem Size runs longer than the fltflt sweep.
        "timeout_seconds": 600,
    },
}

# Default timeout for any future profile that doesn't set its own.
DEFAULT_TIMEOUT_SECONDS = 600


# ---------------------------------------------------------------------------
# Build-dir / executable resolution.
# ---------------------------------------------------------------------------

def _resolve_exe(build_dir, stems):
    bench_dir = build_dir / "bench"
    for stem in stems:
        for cand in (bench_dir / stem, bench_dir / f"{stem}.exe"):
            if cand.is_file():
                return cand
    return None


def find_default_build_dir():
    cwd = Path.cwd()
    if (cwd / "bench").is_dir() and any(
        p.name.startswith("bench_") for p in (cwd / "bench").iterdir() if p.is_file()
    ):
        return cwd
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir / "../../build", script_dir / "../../../build"):
        c = candidate.resolve()
        if c.exists() and (c / "bench").is_dir():
            return c
    return cwd  # let _resolve_exe produce a clear error


# ---------------------------------------------------------------------------
# Per-profile orchestration.
# ---------------------------------------------------------------------------

def run_profile(name, profile, build_dir, out_dir, extra_args):
    exe = _resolve_exe(build_dir, profile["exe_stems"])
    if exe is None:
        print(f"[{name}] could not find any of {profile['exe_stems']} under {build_dir}/bench/",
              file=sys.stderr)
        return False

    json_path = out_dir / f"{name}.json"
    md_path   = out_dir / f"{name}.md"
    csv_path  = out_dir / f"{name}.csv"

    cmd = [str(exe)]
    for b in profile["benchmarks"]:
        cmd += ["--benchmark", b]
    cmd += [
        "--json", str(json_path),
        "--md",   str(md_path),
        "--csv",  str(csv_path),
    ]
    cmd += extra_args
    timeout = profile.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    print(f"[{name}] {' '.join(cmd)}  (timeout {timeout}s)")
    try:
        res = subprocess.run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[{name}] nvbench exceeded timeout of {timeout}s", file=sys.stderr)
        return False
    if res.returncode != 0:
        print(f"[{name}] nvbench exited with status {res.returncode}", file=sys.stderr)
        return False

    if profile.get("summary"):
        try:
            profile["summary"](json_path)
        except Exception as e:
            print(f"[{name}] summary failed: {e}", file=sys.stderr)
            return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--build-dir", type=Path, default=None,
        help="Build directory containing bench/<exe>. "
             "Defaults to the current directory if it has bench_*; otherwise <repo>/build.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES) + ["all"],
        default="all",
        help="Which profile to run (default: all).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("bench_results"),
        help="Directory for nvbench JSON/MD/CSV output (default: ./bench_results).",
    )
    parser.add_argument(
        "nvbench_args", nargs=argparse.REMAINDER,
        help="Extra args forwarded verbatim to the nvbench executable. "
             "Use `--` to separate them from this script's flags.",
    )
    args = parser.parse_args()
    extra = [a for a in args.nvbench_args if a != "--"]

    build_dir = args.build_dir if args.build_dir else find_default_build_dir()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    profiles = sorted(PROFILES) if args.profile == "all" else [args.profile]
    failures = 0
    for name in profiles:
        ok = run_profile(name, PROFILES[name], build_dir, args.out_dir, extra)
        if not ok:
            failures += 1

    print(f"\nDone. {len(profiles) - failures}/{len(profiles)} profile(s) succeeded; "
          f"output under {args.out_dir}/")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
