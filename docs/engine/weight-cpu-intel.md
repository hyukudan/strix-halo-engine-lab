# CPU and Intel weight sources

The weight registry keeps two existing Q8_0 contracts distinct:

| Backend | Canonical arithmetic | Operand and reduction |
|---|---|---|
| CPU | `cpu-q8a8-tree32@1` | Signed Q8 codes, integer dot32, existing scale multiplication and tree32 |
| Intel | `intel-q8a8-f16-ascending@1` | F16-rounded weight and activation operands, ascending FP32 sum |

Intel retains the public policy alias `dense-q8a8-f16-v1`. Its canonical host
reference identity remains separate from the HIP policy bearing that spelling.
Neither registration claims CPU/SYCL cross-device equality or qualifies XMX,
AVX-512, AMX or NEON arithmetic. Unimplemented architecture registrations return
unavailable; IQ, trellis and NVFP4 cannot silently use these Q8 routes.

`Q8Storage` owns raw block geometry. `Q8RawSource<CpuTree32>` and
`Q8F16Source<IntelAscending>` supply compile-time operand access. Host adapters
validate logical `[N,K]`, preserve row prefixes/padding through an owning source,
and explicitly prepare a contiguous host binding when requested. They retain
the original storage identity. Preparation never occurs in a numerical loop.

CPU direct and generic enqueue use one registration and typed argument schema.
Intel's explicitly named `reference_execution_context()` is a host oracle.
Its native context always invokes the existing Intel backend enqueue boundary;
a build without SYCL returns unavailable and never evaluates the host oracle as
an offload fallback. Native source installation retains the vendor runtime's
allocation, event and copy callbacks. Queue and USM lifetimes remain caller-owned.

The Intel cell and SYCL kernel bodies remain unchanged. Mechanical substitution
of their raw addressing is isolated pending authorization under the no-kernel-edit
scope. Host operand traits and wrappers can be qualified without claiming that
this deferred kernel migration has occurred.

Configure `tests/wfmt` with Release and `STRIX_WFMT_CAPTURE=OFF` for production
output receipts. `STRIX_WFMT_CAPTURE=ON` adds a diagnostic copy of the existing
CPU lane accumulators immediately before tree32. Freeze both old executables
before changing operand access, then compare new OFF against old OFF and new ON
against old ON using the same compiler, flags and device. The macro is off in
normal builds. OFF receipts do not invent partials. Intel host receipts capture
source bytes, activation codes/scales and outputs; no register or native-device
capture is claimed for its unchanged cell.

`strix-wfmt-host-tests --wfmt-backend-capture DIR` writes both backend record
sets. The dedicated `strix-wfmt-cpu-tests` and `strix-wfmt-intel-tests` accept the
same adapter command for their own records. Pass `--adapter backend` to the
capture kit. All binaries, payloads and receipts remain outside Git.
