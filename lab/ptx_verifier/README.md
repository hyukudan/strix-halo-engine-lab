# Experimental PTX verification pilot

**Status: synthetic integration passed; verification of complete engine modules
is blocked. Native, model and bitwise qualification remain pending.**

This lab gives selected CUDA kernels a fixed, reviewable verification contract,
captures complete translation units with a local compiler and invokes an
explicitly supplied local verifier. Its purpose is to collect additional evidence
before native correctness and performance evaluation. It is independent of engine
startup, serving, production defaults and the normal build. Its scripts do not
download dependencies, ask a model to generate code or launch a CUDA kernel.

The initial scope is one CTA, fixed dimensions, disjoint allocations and scalar
parameters fixed by the selected case. The entry point, argument order, array
layout and output specification are part of the contract. Results apply only to
the submitted PTX and case.

## Recorded pilot status

The following results were recorded on 2026-10-27 with the pinned tool revision.

| Evidence | Result | Scope |
| --- | --- | --- |
| CPU unit tests | 47 passed | Adapter, capture, integration and compatibility admission behavior, using controlled fixtures. |
| Real verifier on synthetic PTX | 7/7 checks passed | Three positive cases, three numerical mutations and one missing-output mutation. |
| Module-compatibility probes | Baseline accepted; two limitations reproduced | Independent minimal modules reproduce the two complete-module blockers. |
| Complete engine translation units | 2 `CAPTURED` | Unchanged sources compiled to complete PTX modules with recorded provenance. |
| Engine cases using those modules | 3 `ERROR`; none accepted | Both carry cases and HC expansion were blocked before a final verifier verdict. |

The synthetic checks establish that the actual pinned executable can accept the
small fixtures and detect their intended defects. They do not establish that
compiler-emitted engine PTX is supported or correct. `CAPTURED` records successful
compilation and source checks; it is not a verification verdict. No native,
model-level, bitwise-correctness or performance claim follows from these results.

## Optional verifier and provenance

The protocol is pinned to
[`francois141/ptx_verifier@0c7e372f27e5599ee3ad28215f487115b1883585`](https://github.com/francois141/ptx_verifier/tree/0c7e372f27e5599ee3ad28215f487115b1883585).
This optional development tool is MIT licensed. No upstream implementation is
vendored here; preserve its copyright and license with the separately built tool.
The [source ledger](../../docs/reference/sources-ledger.md#optional-development-tools)
records its provenance.

With the upstream Rust and native build prerequisites installed, build that
revision in a separate checkout and record the resulting executable hash:

```sh
git clone https://github.com/francois141/ptx_verifier.git ../ptx-verifier
git -C ../ptx-verifier checkout --detach 0c7e372f27e5599ee3ad28215f487115b1883585
cargo build --manifest-path ../ptx-verifier/Cargo.toml \
  --locked --release -p volta_cli --bin volta
sha256sum ../ptx-verifier/target/release/volta
```

The verifier performs symbolic analysis on the CPU. Its source pin identifies the
intended protocol, while its executable hash identifies the supplied artifact.
Neither `--locked` nor a binary hash attests that a binary was reproducibly built
from unchanged source. Retain compiler, flags and build provenance separately.

## Inspect and run synthetic integration

Run commands from the repository root. Python 3.9 or newer and the standard
library are sufficient for inspection and the CPU unit tests:

```sh
python3 lab/ptx_verifier/runner.py list
python3 -m unittest discover -s lab/ptx_verifier/tests -v
```

Supply the reviewed executable hash to run all seven checks against the real
local verifier. The output directory must be new and its parent must exist outside
the checkout:

```sh
mkdir -p ../private-ptx-runs
PTX_VERIFIER_SHA256='replace-with-the-reviewed-64-hex-digit-hash'
python3 lab/ptx_verifier/integration.py \
  --verifier ../ptx-verifier/target/release/volta \
  --verifier-sha256 "$PTX_VERIFIER_SHA256" \
  --output-dir ../private-ptx-runs/synthetic-01 \
  --timeout-seconds 60
```

The project-owned [fixtures](fixtures/) are handwritten synthetic PTX. The three
positive checks require aggregate `ACCEPT` with complete coverage. Wrong valid
flag, stash-row offset and HC coefficient mutations require numerical refutations
with aggregate `REJECT`. The missing-store mutation requires a `REJECT` explicitly
identifying the unwritten `valid[0]`. An unrelated error, timeout or incomplete
acceptance cannot satisfy a negative check. `summary.json` separates fixture
results from every engine qualification flag.

The separate module-compatibility diagnostic uses the same executable and hash:

```sh
python3 lab/ptx_verifier/compatibility.py \
  --verifier ../ptx-verifier/target/release/volta \
  --verifier-sha256 "$PTX_VERIFIER_SHA256" \
  --output-dir ../private-ptx-runs/compatibility-01
```

It requires acceptance of a minimal baseline before testing an unselected entry
containing `match.any.sync.b32` and an unreferenced external shared-memory
declaration. Its diagnostic result is `LIMITATIONS_REPRODUCED`, not a qualification
or a successful engine verification.

## Capture complete engine PTX

[capture.py](capture.py) supplies a Linux/GCC standalone compilation profile for
`carry_restore_kernel` and `hc_expand_kernel`. Use a complete local CUDA toolkit
with nvcc, CUDA runtime headers and CCCL/CUB. The recorded captures used CUDA
12.8.1's nvcc 12.8.93 and GCC 13.3.0, producing PTX 8.7 for `sm_120`.

The source root must be a Git checkout containing the catalog's immutable source
revision. The translation unit, CMake file and every compiler-reported project
header must match that revision byte for byte. Obtain and review the nvcc hash
before supplying it:

```sh
NVCC_SHA256='replace-with-the-reviewed-64-hex-digit-hash'
python3 lab/ptx_verifier/capture.py \
  --source-root . \
  --kernel carry_restore_kernel \
  --nvcc ../cuda-toolkit/bin/nvcc \
  --nvcc-sha256 "$NVCC_SHA256" \
  --host-cxx /usr/bin/g++ \
  --output-dir ../private-ptx-runs/carry-capture-01 \
  --timeout-seconds 120
```

Use `--kernel hc_expand_kernel` and a different new output directory for HC.
Capture directories must be outside Git worktrees, with an existing parent.
The script compiles the complete `.cu` and preserves the full emitted module; it
does not extract, rewrite or remove kernel bodies or module declarations.

The fixed profile uses C++20, `compute_120`, 64-bit addressing, `-O3 -DNDEBUG`,
`--fmad=false -prec-div=true -prec-sqrt=true -ftz=false` and
`-Xcompiler=-ffp-contract=off,-Wall,-Wextra`, with the project `include` directory.
The numeric flags follow the existing CUDA target declarations in
[CMakeLists.txt](../../CMakeLists.txt). The optimization profile is explicit and
standalone; it does not claim to reproduce a particular production build.

`capture.json` records the command, controlled environment, compiler versions,
source revision, input and toolchain hashes, full PTX hash and unique entry symbol.
The pre-compilation dependency set must match the dependency set reported during
compilation, and input hashes must remain unchanged. Injected compiler flags and
include-path environment variables are rejected. A successful capture sets
`source_files_match_declared_revision=true`, while retaining
`production_build_equivalence="not_attested"` and qualification flags `false`.
Toolchain hashes cover recorded files; they are not a complete build-environment
attestation. Failures after capture starts retain their logs and error receipt.

## Check a captured module

Copy the exact `ptx.symbol` from the capture receipt. C++ entry names are mangled;
the catalog's descriptive kernel name alone is insufficient for these modules.
`plan` reads the supplied file and prints the invocation and input identities
without starting the verifier:

```sh
PTX_ENTRY='replace-with-the-exact-symbol-from-capture.json'
python3 lab/ptx_verifier/runner.py plan \
  --case carry_even_row0 \
  --ptx ../private-ptx-runs/carry-capture-01/module.ptx \
  --kernel "$PTX_ENTRY"
python3 lab/ptx_verifier/runner.py run \
  --case carry_even_row0 \
  --ptx ../private-ptx-runs/carry-capture-01/module.ptx \
  --kernel "$PTX_ENTRY" \
  --verifier ../ptx-verifier/target/release/volta \
  --verifier-sha256 "$PTX_VERIFIER_SHA256" \
  --output ../private-ptx-runs/carry-even-row0.json \
  --timeout-seconds 60
```

The receipt directory must already exist and the output filename must be new.
Input-validation errors before execution emit JSON on stdout and exit 3 without
a receipt. Completed receipts include stdout and stderr, bounded to 1 MiB each;
a timeout or truncated log cannot produce acceptance. Exit codes are 0 for
`ACCEPT`, 1 for `REJECT`, 2 for `UNKNOWN`, and 3 for an adapter error or timeout.
Retain the capture and verifier receipts together, matching their PTX hashes.

### Observed complete-module blockers

Both carry cases encountered a parser error on `match.any.sync.b32` in another
entry of the attention module. The selected carry entry was not the source of
that instruction. HC encountered the dynamic-shared-memory configuration guard
because its complete module declares external shared memory, although the
selected HC entry does not reference it. The case requests zero dynamic shared
memory. These failures preceded a final verdict and output-coverage record; all
three runner results correctly remained `ERROR` and unaccepted.

The attention module also declares external shared memory. Resolving its parser
error might therefore expose the same configuration issue; this is a follow-up
inference, not an additional observed result. Keep the captured modules intact
while investigating compatibility. Removing unrelated entries or declarations
would change the input whose provenance was recorded.

## Initial project-owned contracts

The catalog is [cases.json](cases.json), with output equations in [specs](specs/).
The two candidate kernels are [carry restore](../../src/cuda/ds41_attention.cu)
and [HC expansion](../../src/cuda/ds41_core_norm.cu).

| Case ID | Fixed scope | Expected output elements |
| --- | --- | ---: |
| `carry_even_row0` | Even position zero, stash row zero, one CTA of 256 threads | 1,025: 1,024 state values and one valid flag |
| `carry_even_row1` | Even position zero, stash row one, one CTA of 256 threads | 1,025: same outputs, with a nonzero stash offset |
| `hc_expand_rows1_h32` | One row, hidden width and stride 32, one CTA, no optional carry output | 128: four streams of 32 values |

Odd-position carry is excluded: proving preservation of unwritten state needs
an unchanged-memory contract beyond the current written-output footprint.
The HC case exercises indexing and the declared four-stream expression. Each
additional shape, optional output and launch configuration needs its own case.
HC also requires native checks of operation order and compiler contraction policy
against the engine's numeric identity.

## Interpretation and architecture limits

An intermediate equivalence message is not final acceptance. The runner forces
`--sample 0` and requires final `ACCEPT`, exit zero and complete nonzero output
coverage matching the case. Partial coverage, unknown results, unsupported
instructions, missing outputs and contradictory process status remain unaccepted.
Receipts always retain `native_qualified=false`, `model_qualified=false` and
`bitwise_verified=false`. `--verify-numeric` evaluates symbolic expressions using
the tool's numerical oracle; it does not execute the CUDA kernel.

Acceptance means that the optional experimental tool accepted that case under
its model; it provides no general soundness guarantee. Its real-valued treatment
of floating-point expressions does not establish IEEE rounding, fused-operation
behavior, subnormal handling or bitwise identity on a GPU. The existing
[CUDA core tests](../../tests/cuda/ds41_core_tests.cu), state/model gates and
complete-workload measurements remain independent requirements.

The input is NVIDIA PTX. This lab does not analyze HIP, AMDGPU ISA, wave32/LDS or
gfx1151 WMMA; a CUDA translation would not validate an AMD binary. `sm_120` also
does not establish support for the datacenter `tcgen05`/TMEM path. Consult the
[CUDA architecture table](https://developer.nvidia.com/cuda/gpus) and
[NVIDIA instruction support guidance](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py).

Data-dependent addresses and branches, cross-CTA communication and host ordering
are outside the initial cases. The full Q8/MoE paths need their own instruction
and access inventory. Integer MMA support alone does not establish support for
programmatic launch dependencies. Single-kernel analysis also does not cover
stream/event lifetimes, DMA completion, inter-device state transfer, USB4 or
session continuation.

## Qualification work to follow

Resolve the complete-module compatibility blockers while preserving source and
capture provenance, then rerun the real engine cases and meaningful negative
controls. Any candidate for adoption still needs native operator tests and the
engine's state/model gates. Measure performance on the target device with the
same model, profile and resources only after correctness requirements pass.

Keep generated PTX, build artifacts, raw output and local receipts outside Git.
Public changes contain project-owned contracts, scripts, fixtures and tests;
private run and external inspection material stays outside the checkout.
