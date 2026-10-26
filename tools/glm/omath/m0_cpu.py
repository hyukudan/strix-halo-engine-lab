"""O-math M0, CPU side (design notes §4): OpenBLAS fp64 dgemm rate + run-to-run bitwise identity, per thread count.
usage: OPENBLAS_NUM_THREADS=T m0_cpu.py M,N,K ..."""
import os, sys, time
import numpy as np

thr = os.environ.get('OPENBLAS_NUM_THREADS', '?')
for i, spec in enumerate(sys.argv[1:]):
    m, n, k = map(int, spec.split(','))
    rng = np.random.default_rng(i + 1)
    a, b = rng.standard_normal((m, k)), rng.standard_normal((k, n))
    a @ b  # warm-up
    t0 = time.perf_counter(); c1 = a @ b; s = time.perf_counter() - t0
    c2 = a @ b
    print(f'm0 cpu dgemm threads {thr} M {m} N {n} K {k}: {s:.3f} s {2 * m * n * k / s / 1e12:.3f} TFLOP/s '
          f'det {"yes" if np.array_equal(c1.view(np.uint64), c2.view(np.uint64)) else "NO"}', flush=True)
