# prefill-trace: window-loop census from a kernel trace

`window_census.py` reads one rocprofv3 kernel trace (`rocprofv3 --kernel-trace --output-format csv`) of a prefill
process and reports, for one run of it (default: the last, i.e. the warm run of a `--repeat 2` process):

- the per-window launch loop: number of clusters, launches, span, GPU busy-union and idle inside it;
- inclusive time, count and CTAs per kernel kind, and the launches per hardware queue;
- how many compute units the CU-bound kernel (`--fill`, default the shared flash core `flash_rows`) leaves free:
  a sweep of the CTAs in flight capped at `--cus`, split into the loop edges and its interior, then again with
  every kernel whose LDS cannot share a CU with a fill CTA (`--cu-lds`) counted as occupying one.

Both CU figures are lower bounds of free CU time (running kernels' CTAs are counted as resident). Use it before any
scheduling change to a window loop (grouped launches, more lanes, overlap of the loop edges): a lever that only
removes launches cannot gain more than the loop's idle, and one that only repacks CTAs cannot gain more than the
free CU time.

```
python3 window_census.py r0_kernel_trace.csv [--run N] [--split embedding_kernel] [--kinds a,b,...]
                         [--fill flash_rows] [--cus 40] [--cu-lds 65536] [--edge-us 300]
python3 -m unittest test_window_census.py
```

`--kinds` defaults to the DS41 window set (index, select, flash or exact attention, combine, empty top-k memset); GLM
and MiniMax window loops pass their own kernel names. `--cus` and `--cu-lds` come from the device declaration
(gfx1151: 40 CUs, 64 KiB LDS per CU). No host names or paths: the trace path is the only input.
