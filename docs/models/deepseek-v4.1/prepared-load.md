# Prepared rank-local expert loading (v1)

Opt-in storage adapter for resident EP or TP routed experts in `DeviceForward` and
`BatchForward`. Unset `STRIX_DS41_PREPARED_LOAD` retains the original loader. The
adapter changes storage and upload scheduling; it does not change tensor arithmetic.

Prepared chunk verification uses OpenSSL SHA-256 automatically when its development
package is available at configure time; otherwise it uses the portable SHA-256 backend.
CMake and the runtime load report identify the selected backend. Both produce exactly
the same v1 checksums. `STRIX_USE_OPENSSL_SHA256=ON` additionally requires OpenSSL for
full-file content hashing. Run `strix-ds41-prepare-load` once on each node, with the
same dense environment used at runtime (including the trellis mapping, sites and headout setting). If the app supplies
these through options, set the preparation environment to those effective values:

```sh
strix-ds41-prepare-load MODEL.gguf NEW_DIRECTORY RANK tp OVERLAY.gguf 40 packed
# mode: ep|tp; overlay: file or -; final argument: packed|dup
export STRIX_DS41_PREPARED_LOAD=NEW_DIRECTORY
```

The destination must not exist. Prepare a new directory for a new version; a failed
preparation removes its own `.incomplete` directory. An interrupted process can leave
that directory for operator cleanup. Runtime never creates or repairs artifacts.

The manifest binds the actual model SHA-256 and structural layout identity, overlay
SHA-256, trellis source SHA-256s and environment mapping (including mix-B), world size,
rank, EP/TP choice and down-row storage. The TP identity is
`ds41-expert-tp2-i1152-v1`, intermediate width 1152. Source files are also bound to
canonical paths, device/inode, size and nanosecond mtime/ctime. Full-file digests use
the existing stat-validated identity cache; initial hashing is an offline cost. Files
or artifacts copied to another node should be prepared there, rather than reusing
another node's source stamps. Sources must remain stable during preparation.

Each tensor holds the final rank-local bytes, padded to 4096 bytes. The manifest has
its own SHA-256 and a SHA-256 for every unpadded 16 MiB payload chunk. Files are synced
before the complete directory is published by rename. Version, namespace, inventory,
length and full chunk checks are mandatory. A namespace rejection prints the first
differing field with expected and artifact values (model/source stamp, layout, overlay,
rank/world, EP/TP identity, down layout or dense configuration). In particular, an EP
server needs its rank-local EP pack; a TP pack cannot serve that identity. A server with
expert TP and device prefill uses the `dup` down layout and needs a TP `dup` pack.
Any rejection logs its reason and uses the original loader. If a later chunk fails after earlier uploads, the entire destination
tensor is overwritten through the original path; the remaining cache is disabled for
that forward construction. Artifacts are local derived storage, not authenticated
model-distribution packages.

Offline preparation reads/converts the next expert while hashing/writing the previous
one. Its scratch storage is bounded by a small number of chunks and experts; it never
materializes a layer or model. Runtime allocates sixteen pinned, page-aligned 16 MiB slots
once (256 MiB plus 4096 bytes for alignment). Up to fifteen bounded read/verify jobs
overlap the consumed upload, also avoiding a three-core limit on the portable backend.
Reads are serialized; checksums run outside the I/O lock, allowing parallel SHA verification. Verification precedes every upload, and a slot is not reused until its
upload completes. Linux uses O_DIRECT; a failed direct open uses buffered reads with
consumed pages advised away. The original loaders' existing staging buffers and mapped
source/overlay storage are additional to these bounds. Dense and trellis weights retain
their existing load paths; this artifact contains routed experts only.

`prepared rank ...` reports hit count, payload bytes, pipeline wall time, worker read
and verification time, initial pin allocation time, H2D time, staging size and whether
reads bypassed the page cache. Constructor load time includes pin allocation and all pipeline stages. Stage times
must not be added to that total; read and verification worker times also overlap.
`STRIX_DS41_WEIGHT_HASHES=1` downloads every tensor in the forward's ordinary device
weight map and prints a bounded-memory digest: SHA256 of the concatenated ASCII chunk
SHA256s, in byte order. This diagnostic is excluded from load timing. Trellis library
allocations are outside that ordinary weight map and keep their source-content checks.
The diagnostic uses sixteen ordinary host slots (256 MiB) in addition to any load-time
pinned staging. Downloads run serially on the caller thread while chunk hashes run in
parallel; their digests are collected in byte order before a slot is reused. The
reported per-tensor digest retains the original formula.

Qualification requires device-weight digests for both ranks separately, EP and TP,
identical greedy tokens/logit digests, and only then per-node load microbenchmarks.
Report page-cache policy explicitly; O_DIRECT bypasses the OS cache but does not flush
storage-controller caches. Record process-to-ready and constructor load time separately.
Repeated, controlled final measurements are required before claiming the restart target.
