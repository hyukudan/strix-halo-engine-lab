# Hand-off transport kit (docs/xfer.md)

No host names here: endpoints and files are arguments.

- `strix-xfer-probe` (CMake target, host only):
  - `listen PORT`: the receiver.
  - `send HOST:PORT --sizes 64K,1M,16M --streams 1,2,4 --codecs identity,bp-v1 --data random|bf16|fp32|file:PATH --width W`:
    piece-size sweep, one JSON line per session with MB/s, % of line rate, and CPU s/GB on both ends.
  - `send ... --schedule FILE [--data4 SRC]`: releases each piece at its producer time and prints the tail after the
    last ready piece.
  - `plan --schedule FILE`: the shared cost model for every link class.
  - `codec --data ...`: bp-v1 ratio and MB/s.
- `schedule.py qwen|ds41|uniform ...`: writes piece schedules ("READY_MS BYTES WIDTH" lines) from a family's state
  layout and a measured prefill time.
- `state_ratio.py STATE.rtxs [--out DIR]`: zstd-1 plain vs byte-plane ratio per region of a real Qwen state (Python
  >= 3.14). `--out` writes the regions for `strix-xfer-probe codec --data file:...`.
- `copy_probe.cpp`: H2D/D2H pageable vs pinned, alone and beside a busy kernel. One source for CUDA (nvcc) and HIP
  (hipcc).

Order for a two-host run: start the sender first (it retries the connection for up to 15 min), then the listener
inside the GPU turn of the node. Launching a turn over ssh can block until it ends, so launch it in the background.
