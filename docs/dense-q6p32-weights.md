# Dense Q6P32 weights (`--dense-q6p-pack`)

Status: **draft**. The kernels are bit-exact against their Q8_0 counterparts; the quality of a
6-bit dense checkpoint and its speed are measured separately and are not qualified by this note.

## Format

Q6P32 stores blocks of 32 weights `w = d * q` with `q` in [-32, 31] and `d` an f16: 6.5 bits per
weight against 8.5 for Q8_0. A tensor of R rows (outputs) and K columns (inputs, K % 32 == 0) is
row-major, `K/32 * 26` bytes per row. Inside a row, chunk `c` covers weights `256c..` (n =
min(256, K - 256c) of them) and starts at byte `208c`; with `c_e = q_e + 32`:

| part | bytes | block b of the chunk | content |
|---|---|---|---|
| qs | n/2 | at 16b | byte 4k+i = lo4(c[8k+i]) \| lo4(c[8k+4+i]) << 4 |
| qh | n/4 | at 8b | byte 4j+i = sum_f hi2(c[16j+4f+i]) << 2f |
| d | n/16 | at 2b | f16 scale |

A wave that reads 256 weights of a row touches one contiguous 208-byte range. One qs dword and
half a qh dword give the codes of 8 consecutive weights; `((x | 0x80808080) - 0x20202020) ^
0x80808080` turns the four 6-bit codes of a dword into signed bytes without borrows
(`src/hip_q6p_weights.hpp`). The decoded bytes are, in order, the code bytes of the equivalent
Q8_0 block.

A per-tensor planar layout (all qs, then all qh, then all d) was measured first: it read 23 %
fewer bytes yet ran slower than Q8_0 on the 6144 -> 2560 output projections (three far-apart
streams per row). The chunked rows fixed it.

## Exactness by construction

Every Q6P32 block is an exact Q8_0 block (same `d`, codes `q`). A checkpoint whose dense Q8_0
tensors hold Q6P32 values (the *emulated* checkpoint) therefore runs unchanged on the Q8_0
kernels, and the Q6P32 kernels compute the same bits from the 6.5-bit copies:

- the emulated GGUF carries the numerics (quality is measured on it with the ordinary engine);
- the pack carries the bytes the window kernels read.

`tools/q6p_quantize.py SRC.gguf EMULATED.gguf [--imatrix F]` picks `d` and `q` per block by
importance-weighted error (column importance from an importance-matrix GGUF when given), from
the source Q8_0 values. Routers (F32) are rounded to BF16 values. `tools/q6p_pack.py EMULATED.gguf
PACK.gguf` writes `<name>.q6p` (GGUF I8, Q6P32 rows) and `<name>.bf16` tensors and pins the
emulated file by SHA-256 and content identity.

## Engine

`--dense-q6p-pack PATH` maps and registers the pack, decodes every copy on the host and checks
it against the target tensor bytes, then records `original address -> copy` in `strix::q6p`
(`include/strix/q6p_registry.hpp`). Launchers that have a Q6P32 form look their weights up and
fall back to the Q8_0 kernel when there is no copy or the shape differs. `STRIX_Q6P=0` keeps
every launcher on its original kernel, for A/B on one file.

| path | Q6P32 form |
|---|---|
| q8_rows windows (GDN qkv+z / out, QSA q+k+v / out, shared gate+up / down) | `project_rows_kernel<..., Q6=true>`; wider than 4 direct rows take one output per wave |
| R1 GDN / QSA / shared prefetch8 projections | one-row q8_rows sites (bitwise the R1 kernels); GDN islands keep their recorded kernel |
| LM head rows and R1 (also the MTP head) | `prefetch8_wave_q8_lm_head_rows_kernel<..., Q6=true>` (R1 = one row) |
| HC split-K down (rows, R1) and lane-chain up (rows, R1, register form) | `Q6` template variants |
| router logits (rows, R1) | BF16 copy of the BF16-exact F32 weights |

A pack built from the admitted checkpoint also admits its emulated derivative wherever the
admitted content identity is required (numerical profile, A8 down).

## Tests

- `strix-hip-q8-rows-projection-tests`: six-bit Q8_0 weights through their Q6P32 copy equal the
  Q8_0 rows kernel bitwise for every site, T = 1..8 (3 for GDN), real and tail shapes; the Q8_0
  tensor is poisoned first. `STRIX_Q8_ROWS_BENCH=1` adds cold/warm timings.
- `strix-hip-known-window-tests`: LM head rows and R1, HC rows (split-K and lane chain) and router
  rows with registered copies equal the Q8_0/F32 bytes and read the copies (`q6p::hits()`).
