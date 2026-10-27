# Service health

Qwen HTTP serving publishes `GET /health` and `GET /v1/health` without authentication.
The default response uses `format: "strix-service-health"`, revision 1. The
[schema](service-health.schema.json) defines the public fields. The legacy `status`
and configured public `model` remain present. Treat the configured model ID as public;
do not place credentials or private paths in it.

`STRIX_SERVE_HEALTH_RECORD=0` restores the exact legacy minimal JSON body and HTTP
status behavior. A service without a health record also retains that body.
The health record describes resolved settings frozen at startup.
When `--http-capabilities on` (the default), `capabilities_id` matches the existing
[capabilities document](../engine/serving/serving.md#chat-requests) and
`capabilities_url` is `/v1/capabilities`. With `--http-capabilities off`, both
fields are null and the capability route remains unavailable. Features are described
only by `GET /v1/capabilities`; health carries no route or feature inventory.
The legacy body retains its `capabilities_id` only when capabilities are on.
Health never arms performance observation, submits generation work, reads a cache
inventory, takes runner locks, or calls a device API.

## Readiness and backend liveness

`ready` requires service state ready and fresh device completion evidence for every
required rank. `backend.state` is one of `starting`, `responsive`, `probe_pending`,
`unresponsive`, `failed`, `draining`, or `unchecked`. Proof is null when unknown,
`control-only` for an owner acknowledgement, `completion-boundary` for completed device
work, or `idle-scratch-event` for an isolated completed probe. Control acknowledgements
cannot establish readiness. Ages are nonnegative seconds; device proof must be at
most one second old. Missing or stale peer evidence prevents readiness.

Qwen enables backend probing by default. Concurrent health readers share one pending
generation in fixed atomic storage. Generations never wrap; exhaustion permanently
latches backend failure before the reserved zero value can be published. A reader waits
up to 100 ms for its acknowledgement, then returns available evidence. Fresh cached evidence is reused for one second.
The actual backend owner acknowledges at completed work boundaries: after the batched
runner returns from successful decode work, or after the single-request bridge has a
completed target prediction or verified cycle. No prompt, token, generation ticket,
sampling operation or device synchronization is submitted for a health request.
The unrequested owner hook performs one relaxed atomic load; clock reads and evidence
publication occur only when a request is pending, outside batch runner timing scopes.

This implementation uses the idle fallback: a health control request wakes the owner's
existing queue wait, and the fully idle owner acknowledges `control-only`. It does not
allocate or submit a scratch event. Idle control proof keeps `backend.state` at
`probe_pending` and `ready: false`. A prompt-only batch step leaves the probe
pending; device readiness conservatively waits for successful decode completion.
An earlier device proof can establish readiness while it remains fresh, even after
its request finishes. A busy owner without a completed boundary leaves the request
pending; this can occur during a healthy long prefill and is not a GPU fault.

An otherwise healthy Qwen service retains HTTP 200 while proof is pending, with
`ready: false`. Use `ready` and the proof fields to assess backend progress; HTTP
success alone establishes reachability. `STRIX_SERVE_BACKEND_PROBE=0` explicitly
skips probing and returns `unchecked`, `ready: false` and HTTP 503 with the health
record enabled. Failed or draining services always return HTTP 503. Pending evidence
alone is not a restart trigger. Idle scratch-event integration and native device
qualification remain outstanding; CPU tests do not establish inference exactness.

## Fields and evidence

- `release` contains the configured build version, full source SHA, digest of the
  embedded build JSON bytes, and its parsed allowlisted build information. Configured
  compiler evidence is not a complete compile-command receipt. Missing identity is null.
- `identity.model_key` and `content_identity` remain null until independently bound
  loader evidence is available. Raw manifest names are never copied into public identity.
  `assets` is unbound without a manifest. `--release-manifest assets/assets.json`
  performs full bundle verification once before model/device startup, with members
  rooted beside the manifest. It requires `--http-serve` and rejects cohort requests,
  sample repeats, file-backed serving/prefix followups and offline compile provenance
  during option validation, before model/device work. Single and batched HTTP are supported.
  Verification failures stop startup with bounded public messages, never filesystem paths.
  Explicit target, draft, calibration, vocabulary and profile-file inputs must belong
  to that verified
  package with their corresponding role. Offline tuning caches use `profile`. The
  HTTP API-key file is an operator secret, never part of a release package: it is not
  bound, hashed or reported. Runtime prefix-cache entries and HTTP state imports are mutable service
  state, outside the static release inventory. `verification: "full"` and `coverage: "complete"`
  describe the declared manifest closure, not publisher authenticity or model quality.
  Role digests summarize sorted key/byte/hash records; no member paths are exported.
  Inspection with `--print-effective-config` does not hash assets or start the service.
- `capacity` gives context tokens and actual runner slots.
- `execution` identifies the resolved decode/prefill arithmetic profiles, prefill tile
  rows and recognized draft profile. An absent draft profile is null. Zero tile rows
  means no tiled prefill configured.
- `storage.prefix` reports enabled state and host/disk budget bytes, including known
  configured zero. Directories are private. Session storage and resident pool support
  remain false because these handlers are not installed on this branch.
- `load` gives a current queue snapshot: queued and running request counts. Its age is
  null because the queue does not publish an independent sample timestamp.
- `backend` contains evidence states, proof and ages described above, plus rank counts.
  Unknown counters are null, never inferred as zero from an HTTP acknowledgement.
- `preflight` contains cached startup warning/advisory codes and the age in seconds
  since the original steady-clock sample completed. State is `ok`, `advisory`, `warning`
  or `unavailable`. `warnings` and `advisories` keep the startup record's split
  (advisories are normal facts, never alert triggers); consult
  [startup preflight](startup-preflight.md) for their meanings. Disabling preflight or
  startup events leaves this evidence unavailable. No health request repeats collection.

Unknown values use JSON null; known disabled capacities use zero or false. Durations
and ages are seconds, capacities are tokens/rows, storage limits are bytes, and queue
and rank values are counts. Revision 1 can gain optional fields; incompatible semantic
changes require a new revision.

Serialization admits controlled profile identifiers, warning codes, numeric
values and digests. It excludes arbitrary options, engine descriptions, raw errors,
paths, hosts, process identifiers, command lines, prompts and session identifiers.
CPU contract tests cover synthetic proof transitions; they do not qualify a device
adapter or establish inference exactness. No throughput gain is claimed.
