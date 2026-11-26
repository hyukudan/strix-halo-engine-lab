# Synthetic size and reconstruction error for Strix Pack

Two seeded 192×256 weight matrices used either standard Gaussian values or a Student t distribution with
3 degrees of freedom scaled to unit variance. The weight RNG was MT19937 with seeds 2693 and 2719.
Two independent 512×256 activation sets used seeds 8273 (fit) and 8931 (evaluation), column correlation
`z[j] = 0.85*z[j-1] + sqrt(1-0.85^2)*Normal(0,1)`, and column standard deviations from 0.5 to 2.375
repeated every 16 columns. RTN and calibration shared the default table for each width. Calibration
used diagonal importance plus full GPTQ-style feedback with damping 0.1. The study uses the production
coder, with experimental group geometry for 64 and 128. The distributions use the C++ standard library's
normal sampler, so their raw draws are reproducible with the same standard library.

Weight error is the relative Frobenius norm `||Wq-W||/||W||`. Output error is the relative RMS error
`sqrt(sum ||(Wq-W)x||² / sum ||Wx||²)` on the independent evaluation set. Payload bytes per weight
are `bits/8 + 1/group`. Each tensor also has a 16-byte logical header (32 hex bytes on disk), a
container entry and alignment gaps. Both encoder techniques have the same payload size.

| Weight distribution | Format | Group | Payload B/weight | RTN weight error | Calibrated weight error | RTN output error | Calibrated output error |
|---|---|---:|---:|---:|---:|---:|---:|
| Gaussian | SP4 | 32 | 0.531250 | 8.24% | 10.54% | 8.27% | 5.02% |
| Gaussian | SP4 | 64 | 0.515625 | 8.82% | 11.12% | 8.76% | 5.24% |
| Gaussian | SP4 | 128 | 0.507812 | 9.32% | 11.75% | 9.26% | 5.52% |
| Gaussian | SP3 | 32 | 0.406250 | 16.12% | 20.61% | 15.94% | 9.74% |
| Gaussian | SP3 | 64 | 0.390625 | 17.21% | 21.69% | 17.28% | 10.27% |
| Gaussian | SP3 | 128 | 0.382812 | 17.93% | 22.53% | 17.78% | 10.71% |
| Gaussian | SP2 | 32 | 0.281250 | 32.77% | 41.01% | 32.63% | 20.23% |
| Gaussian | SP2 | 64 | 0.265625 | 33.73% | 41.88% | 33.74% | 20.73% |
| Gaussian | SP2 | 128 | 0.257812 | 34.26% | 42.49% | 34.14% | 21.12% |
| Student t(3) | SP4 | 32 | 0.531250 | 10.10% | 13.15% | 9.97% | 5.94% |
| Student t(3) | SP4 | 64 | 0.515625 | 12.23% | 15.83% | 11.99% | 7.16% |
| Student t(3) | SP4 | 128 | 0.507812 | 14.64% | 19.23% | 14.53% | 8.45% |
| Student t(3) | SP3 | 32 | 0.406250 | 22.63% | 29.25% | 22.72% | 12.93% |
| Student t(3) | SP3 | 64 | 0.390625 | 26.98% | 34.49% | 27.75% | 15.23% |
| Student t(3) | SP3 | 128 | 0.382812 | 30.89% | 38.90% | 32.08% | 17.62% |
| Student t(3) | SP2 | 32 | 0.281250 | 52.65% | 64.30% | 54.57% | 29.42% |
| Student t(3) | SP2 | 64 | 0.265625 | 60.97% | 73.78% | 64.11% | 33.20% |
| Student t(3) | SP2 | 128 | 0.257812 | 72.91% | 87.35% | 75.37% | 38.11% |

At group 32, SP3 saves 0.125 B/weight relative to SP4, and SP2 saves a further 0.125 B/weight. Each
step increases the calibrated output error, from 5.02% to 9.74% to 20.23% on Gaussian weights
and from 5.94% to 12.93% to 29.42% on Student t(3) weights. Increasing the group from 32 to 128
saves 0.0234375 B/weight at any bit width. The calibrated SP2 output penalty for that change is
0.88 percentage points on Gaussian weights and 8.69 points on Student t(3) weights. Calibration
reduced held-out output error in every cell, with a higher weight reconstruction error in each.
This experiment measures a specified correlated-input distribution. Model-tier admission uses
the model's own quality gate.

Build `strix-weight-host-analysis` in a Release host build and run it with `--study` to print the
table as TSV. The [Strix Pack specification](strix-pack-v1.md) defines the registered group-32
wire format.
