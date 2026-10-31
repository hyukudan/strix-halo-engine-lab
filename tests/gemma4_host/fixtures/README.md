# Host fixtures

These are small synthetic fixtures for the public mathematical and storage contracts;
they contain no checkpoint slices, prompts or external inspection receipts.

`make_fixtures.py` in the parent directory regenerates them. It needs numpy and gmpy2;
the C++ tests do not. `rsqrt.bin` contains little-endian `(input_bits,output_bits)` pairs
checked with 128-bit MPFR and a binary32 rounding proof. `rope.bin` contains interleaved
cos/sin for positions 0, 511, 512 and 65535, sliding then global per position, generated
by the checked offline table generator. `quant.bin` stores two blocks per format in
Q8_0/Q4_K/Q6_K order, each followed by its decoded FP32 oracle values. Q6 scales include
-128, -127, -65, -1, 0, 1, 64 and 127. `nonlinear.bin` stores input/exp/tanh/GELU records
from independent explicitly rounded numpy arithmetic. `arrays.npz` tests compressed
fixture parsing; `config.json` is the synthetic supported text configuration.
