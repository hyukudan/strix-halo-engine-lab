# Decode defaults (D-055)

The final measurement campaign compared the decode switches on the same binary and node: dense A8 profile, MTP
windows 2/4/8, 256 tokens, eight workloads (code, English and Spanish prose and chat, French prose, a document
summary, an agent prompt with tool calls). These defaults follow from it. Every one of them only changes which
drafts are proposed and which verify window runs; the target verifies every token, so the output is the output
of the same command with the switches given explicitly (the gates below check it).

## Draft profiles (`include/strix/mtp_draft_profiles.hpp`)

The engine takes every asset explicitly: the MTP sidecar is a local file with SHA-256 and provenance pins. A
*draft profile* names a qualified sidecar and its best decode setting:

| profile | sidecar | min-p default | campaign, decode vs the base draft at 0.85 (geometric mean, 8 workloads) |
|---|---|---:|---:|
| `base` | the published Q8_0 draft | 0.85 (D-022) | — |
| `distill-d` | distilled draft, round 1 (`mtp/distill`) | 0.7 | +8.2 % |
| `distill-g` | distilled draft, round 2 | 0.7 | **+10.0 %** (English/Spanish prose +14-15 %) |

- `--mtp-draft-profile NAME` fills the sidecar's logical name, size, SHA-256, source repository/revision/path and
  converter revision/SHA-256s, so a command needs only `--mtp-sidecar PATH` (the file still has to match every
  pin: the loader checks them). An explicit pin that disagrees with the profile is an error.
- Without `--mtp-spec-min-p`, a run with a row set (`--mtp-target-window-rows-set`) takes the min-p of the
  draft identified by `--mtp-sidecar-sha256` (or by the profile); an unprofiled sidecar keeps 0.5 and a single
  target window still does not stop the chain early. An explicit `--mtp-spec-min-p` always wins.
- **Recommended:** `--mtp-draft-profile distill-g` (`mtp_recommended_draft_profile`), for the command line and
  for `--http-serve` (same flags).

Measured per arm (tok/s, cluster node 2; acceptance rises with the distilled drafts, the lower min-p proposes
more drafts per cycle):

| arm | code | ES | EN | EN chat | ES prose | FR prose | ES summary | agent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base draft, 0.85 | 53.7 | 39.7 | 39.7 | 46.1 | 42.0 | 45.2 | 55.2 | 53.4 |
| distillD, 0.7 | 56.4 | 45.2 | 44.7 | 49.7 | 47.0 | 47.3 | 59.3 | 54.7 |
| distillG, 0.85 | 54.8 | 40.0 | 44.8 | 49.5 | 47.0 | 45.1 | 56.9 | 54.9 |
| **distillG, 0.7** | **57.3** | **45.6** | **45.1** | **50.6** | 47.5 | 47.5 | **63.0** | **55.3** |

## Context n-gram extension: `auto` by default

`--mtp-ngram-draft` (D-047) now has three modes, default `auto`:

- `auto`: on wherever it is admissible (draft chain, prefix windows, greedy, one request, the D-022 controller),
  including every request of `--http-serve`; off under sampling, cohort requests, golden dumps and the survival
  controller. Its own per-request gate is the detection of editing requests: the would-be extension is scored
  against the emitted tokens every cycle (host only), and rows are proposed only while their learned acceptance
  is at least 0.5, i.e. while the output is copying the context.
- `on` (or the bare flag): required; the command is rejected where it cannot run.
- `off`: disabled.

A server request can set `"ngram_draft": true` (on where admissible) or `false`; the response's `strix` object
reports `ngram_rows` and `ngram_accepted`.

Campaign (tok/s, decode): document correction English 81.1 → 85.8 (+5.8 %), Spanish 80.3 → 86.5 (+7.7 %);
English chat 46.1 → 46.2 and Spanish summary 55.2 → 55.0 (the gate did not open: within noise).

## Gates

- Options contract (`strix-hip-mtp-decode-options-tests`): min-p by draft identity, profile pins, disagreeing and
  unknown profiles, the three n-gram modes and where `auto` stays off.
- Engine: the default command gives the tokens of the command with the explicit switches (min-p and
  `--mtp-ngram-draft`), replay error 0; the server's tokens equal the command line's with the same mode.
