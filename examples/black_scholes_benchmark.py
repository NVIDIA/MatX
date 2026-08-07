#!/usr/bin/env python3
"""Benchmark eager NumPy and CuPy Black-Scholes array expressions.

This companion to black_scholes.cu evaluates the same FP32 equation and array
size as the MatX apply() case. Input generation and host/device transfers are
outside the timed region; the eager array expression and its intermediates are
inside it.
"""

import argparse
import gc
import statistics
import time


def black_scholes(xp, ndtr, strike, stock, volatility, rate, maturity):
    """Return the Black-Scholes call price using eager array operations."""
    volatility_sqrt_time = volatility * xp.sqrt(maturity)
    d1 = (
        xp.log(stock / strike)
        + (rate + xp.float32(0.5) * volatility * volatility) * maturity
    ) / volatility_sqrt_time
    d2 = d1 - volatility_sqrt_time
    return (
        stock * ndtr(d1)
        - strike * xp.exp(-rate * maturity) * ndtr(d2)
    )


def make_inputs(xp, size, seed):
    rng = xp.random.default_rng(seed)
    # The MatX example uses uniform FP32 inputs. Avoid the rare exact zero so
    # strike division and sqrt(T) remain finite; setup is outside timed regions.
    epsilon = xp.float32(1.0e-6)
    return tuple(
        xp.maximum(rng.random(size, dtype=xp.float32), epsilon)
        for _ in range(5)
    )


def report(name, samples_ms, size):
    median_ms = statistics.median(samples_ms)
    throughput = size / (median_ms * 1.0e3)
    print(f"{name}: {median_ms:.2f} ms ({throughput:.1f} M options/s)")
    print("  trials: " + ", ".join(f"{sample:.2f}" for sample in samples_ms))


def benchmark_numpy(size, warmups, trials):
    import numpy as np
    from scipy.special import ndtr

    inputs = make_inputs(np, size, seed=1234)
    for _ in range(warmups):
        output = black_scholes(np, ndtr, *inputs)
        del output

    samples_ms = []
    for _ in range(trials):
        start = time.perf_counter()
        output = black_scholes(np, ndtr, *inputs)
        samples_ms.append((time.perf_counter() - start) * 1.0e3)
        del output
    report(f"NumPy {np.__version__} / SciPy", samples_ms, size)

    del inputs
    gc.collect()


def benchmark_cupy(size, warmups, trials):
    import cupy as cp
    from cupyx.scipy.special import ndtr

    inputs = make_inputs(cp, size, seed=1234)
    for _ in range(warmups):
        output = black_scholes(cp, ndtr, *inputs)
        cp.cuda.get_current_stream().synchronize()
        del output

    samples_ms = []
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    for _ in range(trials):
        start.record()
        output = black_scholes(cp, ndtr, *inputs)
        stop.record()
        stop.synchronize()
        samples_ms.append(cp.cuda.get_elapsed_time(start, stop))
        del output
    report(f"CuPy {cp.__version__}", samples_ms, size)

    del inputs
    cp.get_default_memory_pool().free_all_blocks()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("numpy", "cupy", "all"),
                        default="all")
    parser.add_argument("--size", type=int, default=100_000_000)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    if args.backend in ("numpy", "all"):
        benchmark_numpy(args.size, args.warmups, args.trials)
    if args.backend in ("cupy", "all"):
        benchmark_cupy(args.size, args.warmups, args.trials)


if __name__ == "__main__":
    main()
