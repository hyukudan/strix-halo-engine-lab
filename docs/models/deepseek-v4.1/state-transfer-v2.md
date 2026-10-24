# Experimental state transfer v2

`tools/ds41_state_transfer_v2.py` is a disk-backed local receiver stub. It is started explicitly and does not
import device state or change the existing v1 import route. The prefill service has an experimental, default-off
sender; its device streaming and distributed commit qualification remain pending.

Start it with `--root DIR --expected-identity ID`; optional arguments select host, port, and bearer key. `ID` must
identify the effective weights, overlays, and numerical configuration. The caller is responsible for computing
that identity. The default listener is loopback port 8097. Use a dedicated root for its staging and committed files.

The proposed HTTP contract uses `POST /admin/import-state/v2/<id>/<operation>`, where `id` is 32 lowercase hex
digits. A configured key is sent as `Authorization: Bearer KEY`. Operations are:

- `begin`: JSON `{position, identity, sections:[{name, bytes}]}`. Section sizes are capacity bounds; total capacity
  must fit the receiver budget. The stub allows one active transaction, 1.5 GiB total, and at most 128 receipts.
- `chunk`: up to 4 MiB of bytes with `X-State-Section`, `X-State-Offset`, and `X-State-Sha256`. Identical retries
  succeed. Conflicting retries, overlaps, excess bytes, and incorrect checksums fail before writing the chunk.
- `finalize`: JSON `{sha256, segments:[{section, offset, bytes}]}` describing the exact v1 file in serialization
  order. Each used section must be covered once from offset zero, without gaps or unreferenced received bytes.
- `validate`: empty body or `{}`. Assemble a temporary v1 file, check its complete SHA and header/position, and flush
  it to disk. The production receiver must additionally run the complete native v1 semantic reader before import.
- `commit`: empty body or `{}`. Publish the validated file with an atomic rename. Retries return the same receipt.
  The stub returns `imported:false`; file publication is its entire commit operation.
- `abort`: empty body or `{}`. Remove an uncommitted transaction's staging files.

KV and index-key sections can arrive as immutable entries become available. `final` carries the remaining v1
bytes: headers, rings, carries, selections, tokens, digest, and checksum. `split_v1` separates an existing v1 artifact
without changing any bytes and provides the ordered assembly manifest. The CUDA observer keeps two pinned 4 MiB
buffers and waits for acknowledgement before reusing either buffer. The service enables it with `--pass-rows P
--handoff-v2 --state-identity ID`; the request's `handoff` is the base `/admin/import-state` URL. KV and index keys
stream during prefill; the remaining bytes and complete v1 SHA are finalized afterward. A failed handoff remains
available through `/v1/handoff`, which performs a complete upload under a new transaction id. Individual HTTP
operations retry the same transaction and bytes up to three times on connection failure.

For CPU-only transport checks, use `strix-cuda-ds41-prefill-service STATE_FILE --handoff-only URL --handoff-v2
--state-identity ID`. This explicit utility mode loads the native v1 file and transfers it without constructing the
CUDA engine or querying a device. Optional `--handoff-api-key` and `--v2-timeout` apply (30 seconds per operation by
default, up to three attempts; abort uses one attempt of at most five seconds). The receipt includes
separate read, serialization, SHA and transfer times. This mode measures neither prefill nor decode.

The Windows v2 sender uses the system SHA-256 provider and verifies empty-string and `abc` known answers before
starting. Other platforms use the existing project SHA implementation. Neither path changes serialized bytes.
The final v1 seal and receiver validation still scan the complete state; the final latency target is unqualified.
Transport backpressure can delay reaching the next cancellation boundary.

The stub serializes operations within one process. A new begin expires uncommitted staging after 15 minutes
without a successful operation, so an orphan cannot permanently occupy admission. It does not resume transactions
after restart, import to a
device, replicate across ranks, provide a distributed atomic commit, or qualify handoff latency. Committed files
remain in `DIR/committed` for inspection. These limits must be resolved by the production integration.

Run CPU checks with `python tests/ds41_state_transfer_v2_tests.py`. Optional repeated `--state FILE` arguments
check complete byte identity after reconstruction of existing native state artifacts. Tests cover authenticated
HTTP, out-of-order chunks, retries, loss, corruption, overlaps, conflicts, capacity limits, publication failures,
and rejection of commit before validation.
