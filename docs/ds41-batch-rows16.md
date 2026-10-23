# Sixteen-row BatchForward steps

`STRIX_DS41_BATCH_ROWS16=1` enables up to sixteen real rows in one
`BatchForward::step()`. The default remains eight. Set the flag before creating
the forward and use the same value on both ranks.

`BatchForward::row_limit()` reports the limit captured by a forward.
`BatchForward::configured_row_limit()` reports the environment setting for
planning before construction. `BatchForward::max_rows` remains eight: it is the
maximum consecutive segment of one request, including a speculative window.
For example, sixteen requests can each contribute one row, or two requests can
each contribute eight rows. Longer segments are rejected before state changes.

The batch runner accepts `--n 9,10,11,12,13,14,15,16` with the flag. Planners
must pass the forward's runtime limit as the third argument of `plan_step()`;
the default argument retains the eight-row schedule. Each prompt segment is
still bounded to eight rows.

The wider step allocates sixteen-row activations, attention scratch, routing
storage, logits and exchange buffers. Dense Q8/packed and F16/F8 products share
weight loads across the wider rows. Routing uses a separate 128-entry builder
above 64 routes. Routed products and trellis adapters retain their existing
internal row tiles and iterate across the full union. Per-request attention
rings, compression carries, cache reservations and state-file layouts retain
their existing semantics.

For serving, the initial control block remains 512 bytes. Only a step above
eight rows exchanges a further 64 bytes containing rows 8 through 15. The hello
checks that both ranks agree on the row-limit flag. `--slots` can be up to
sixteen; `--prompt-rows` remains bounded to eight.

Qualification is experimental. Native product tests and focused CPU tests pass.
Four-layer two-rank emulation passes at nine through sixteen requests: each
request's tokens, rank-local logits digests and final state digest match its
independent greedy run. Full-model two-node exactness and the short wider-row
measurement remain pending in the combined integration gate. The flag remains
off by default.
