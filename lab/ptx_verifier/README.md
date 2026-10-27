# Experimental PTX verification pilot

**Status: development experiment; native execution and model qualification pending.**

This lab gives selected CUDA kernels a fixed, reviewable verification contract and
invokes an explicitly supplied local verifier. Its purpose is to collect additional
evidence about a candidate before native correctness and performance evaluation.
The lab is independent of engine startup, serving, production defaults and the
normal build. It does not download dependencies or ask a model to generate code.

The initial scope is one CTA, fixed dimensions, disjoint allocations and scalar
parameters fixed by the selected case. The kernel entry point, argument order,
array layout and output specification are part of the case. A result applies only
to that PTX and that contract.

## Optional verifier and provenance

The protocol is pinned to
[`francois141/ptx_verifier@0c7e372f27e5599ee3ad28215f487115b1883585`](https://github.com/francois141/ptx_verifier/tree/0c7e372f27e5599ee3ad28215f487115b1883585).
The upstream project is MIT licensed; its copyright and license remain with its
separately built executable. No upstream implementation is vendored here. The
[source ledger](../../docs/reference/sources-ledger.md#optional-development-tools)
records this optional development dependency.

Build the chosen revision separately using its own build instructions and supply
its local executable explicitly. The verifier performs symbolic analysis on the
CPU. Compiling production CUDA to PTX and checking its numerical behavior on the
target device are separate operations.

A source revision identifies the intended protocol. Record the actual executable
hash as well: a source revision alone does not identify a compiler, build flags,
local modifications or the resulting binary.

## Use

Run commands from the repository root. Python 3.9 or newer and the standard library are
sufficient for inspection and adapter tests:

```sh
python3 lab/ptx_verifier/runner.py list
python3 -m unittest discover -s lab/ptx_verifier/tests -v
```

`plan` reads a caller-supplied PTX file and prints the intended invocation and
input identities without starting the verifier:

```sh
python3 lab/ptx_verifier/runner.py plan \
  --case carry_even_row0 \
  --ptx build/carry.ptx
```

Use `--kernel ACTUAL_PTX_ENTRY_SYMBOL` when the compiled entry has a different
name, such as a mangled CUDA symbol. The supplied module must retain the selected
kernel's actual body and interface. This pilot does not extract or generate PTX.
Source, compiler, flags and target provenance remain separate evidence until a
production build capture is completed.

To run a check, first obtain and review the SHA-256 of the separately built
verifier executable. Supply that exact hash:

```sh
mkdir -p ../private-ptx-runs
PTX_VERIFIER_SHA256='replace-with-the-reviewed-64-hex-digit-hash'
python3 lab/ptx_verifier/runner.py run \
  --case carry_even_row0 \
  --ptx build/carry.ptx \
  --verifier ../verifier-build/target/release/volta \
  --verifier-sha256 "$PTX_VERIFIER_SHA256" \
  --output ../private-ptx-runs/carry-even-row0.json \
  --timeout-seconds 60
```

The supplied hash binds the run to an executable artifact; it is not a
reproducible-build attestation that the binary came from the pinned source. Keep
that build provenance separately. The runner does not install the optional tool
or launch a CUDA kernel.

The receipt directory must already exist and the output filename must be new;
existing files are never overwritten. Input-validation failures before execution
emit an error record on stdout and exit with code 3, without writing a receipt.
Completed run receipts include stdout and stderr, bounded to 1 MiB each; a timeout
or truncated log cannot produce acceptance. Run exit codes are 0 for `ACCEPT`, 1
for `REJECT`, 2 for `UNKNOWN` and 3 for an adapter error or timeout.

## Initial project-owned contracts

The case catalog is [cases.json](cases.json); its output equations are in
[specs](specs/). These contracts describe the engine's own computations.

| Candidate | Engine source | Initial scope | What this exercise can establish |
| --- | --- | --- | --- |
| Carry restore | [`carry_restore_kernel`](../../src/cuda/ds41_attention.cu) | Fixed even position and stash row zero or one; one CTA | The selected state copy and valid flag, including the nonzero stash offset. |
| HC expansion | [`hc_expand_kernel`](../../src/cuda/ds41_core_norm.cu) | One small row, fixed hidden width and stride, no optional carry output; one CTA | Output indexing and the declared four-stream algebraic expression. |

| Case ID | Fixed shape | Expected output elements |
| --- | --- | ---: |
| `carry_even_row0` | Position zero, stash row zero, 256 threads | 1,025: 1,024 state values and one valid flag |
| `carry_even_row1` | Position zero, stash row one, 256 threads | 1,025: 1,024 state values and one valid flag |
| `hc_expand_rows1_h32` | One row, hidden width and stride 32 | 128: four output streams of 32 values |

The odd-position carry branch is excluded from this catalog: proving preservation
of unwritten state needs an unchanged-memory contract beyond the current output
footprint check.

The small HC shape exercises the contract; it does not qualify the production
shape or a multi-CTA launch. Each additional shape, optional output and launch
configuration requires its own case and native checks.

The carry contract is useful for bringing up the adapter without floating-point
arithmetic. The HC contract introduces symbolic floating-point data and several
outputs per thread. HC acceptance still requires the native operation order and
compiler contraction policy to be checked against the engine's numeric identity.

## Interpretation of a result

Keep process completion, expression equivalence, coverage and graph checks
separate. An intermediate equivalence message is not a final acceptance result.
A partial comparison, unknown result, unsupported instruction, timeout, missing
output or contradictory process status must remain unaccepted.

The runner forces `--sample 0`. Acceptance requires the final `ACCEPT` verdict,
exit code zero and nonzero complete output coverage matching the selected case.
Receipts always retain `native_qualified=false`, `model_qualified=false` and
`bitwise_verified=false`; a symbolic acceptance cannot promote those fields.
The requested `--verify-numeric` check evaluates the symbolic expressions using
the tool's numerical oracle; it is not an execution of the CUDA kernel.

An accepted result means only that the selected tool accepted the submitted case
under its model. The optional tool is experimental and does not provide a general
soundness guarantee. It represents floating-point arithmetic over reals, which
does not establish IEEE rounding, fused-operation behavior, subnormal handling or
bitwise identity on the GPU.

The following evidence remains independent:

| Evidence | Required interpretation |
| --- | --- |
| Adapter tests | The runner interprets representative outcomes conservatively and constructs the intended invocation. |
| Symbolic result | The specified case was accepted, rejected or left unresolved under the recorded tool model. |
| Native operator tests | The compiled kernel satisfies its own reference and declared numerical policy on the target device. |
| State and model gates | The change preserves the required state, logits and token behavior for the selected engine identity. |
| Complete-workload measurement | The accepted change improves the relevant workload with the same model, profile and resource conditions. |

The existing [CUDA core tests](../../tests/cuda/ds41_core_tests.cu) and engine
qualification gates continue to apply. Host-only adapter tests must never be
reported as native verification or a model-level pass.

## Architecture and analysis limits

The input is NVIDIA PTX. This lab does not analyze HIP, AMDGPU ISA, wave32/LDS
behavior or gfx1151 WMMA instructions. An AMD backend would need its own execution
and arithmetic model; a CUDA translation would not validate the AMD binary.

An architecture label in a PTX module also does not establish hardware support.
In particular, the RTX PRO 6000 Blackwell target is `sm_120`; the datacenter
`tcgen05`/TMEM path is a different target capability. Consult the
[official CUDA architecture table](https://developer.nvidia.com/cuda/gpus) and
[NVIDIA's instruction support guidance](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py)
when selecting instructions.

Keep data-dependent addresses and branches, cross-CTA communication and host-side
ordering outside the initial cases. The full CUDA Q8/MoE paths require a separate
instruction and memory-access inventory. For example, support for an integer MMA
does not establish support for programmatic launch dependencies in the same
kernel. Removing synchronization to make a different PTX module analyzable does
not validate the original module.

Single-kernel analysis does not cover stream/event lifetimes, DMA completion,
inter-device state transfer, USB4 transport or session continuation.

## Qualification work to follow

The checked-in adapter tests use controlled process results. The optional
verifier and production kernels have not been executed as part of this pilot's
initial implementation.

1. Run each initial contract against the pinned local verifier, retaining the
   executable, PTX, spec and case hashes with its complete outcome.
2. Add deliberate index/output mutations of a local PTX candidate and establish
   that the actual verifier rejects them under the selected contract.
3. Capture PTX from the unchanged engine source with recorded compiler and target
   flags, and bind each analyzed module to that source/build identity.
4. Run the existing native operator and engine-specific state/model gates for any
   candidate intended for adoption.
5. Measure accepted candidates on the target device with an interleaved reference
   and candidate comparison. Select further kernels from measured workload cost.

Write run artifacts and raw verifier output outside the checkout. Public changes
contain the contracts, implementation and test code; local receipts and external
inspection material remain private. No speed or native-correctness claim follows
from adding this lab.
