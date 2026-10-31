# Model families

The engine core knows no model. It provides the GGUF reader, residency, the quantised-format library, MoE and
verify windows, golden replay, the prefix cache, the two-rank communicator and the server. Every model family is an
optional module that registers itself with the core and owns its contract, execution plan and kernels. Any subset of
families must build and pass its own tests. Today `deepseek41` is a module (CMake option `STRIX_FAMILY_DEEPSEEK41`,
default ON). The Qwen path will move behind the same registry in a dedicated step with its own gates; until then it
is built unconditionally and does not depend on any family module. The `gemma4` family (Nemotron 3.5
Bolt E4B, CMake option `STRIX_FAMILY_GEMMA4`, default OFF) builds its own targets under `src/gemma4/` and does not
register with the core yet; see [nemotron-3.5-bolt-e4b](nemotron-3.5-bolt-e4b/README.md).

## The registry (core)

`include/strix/model_family_registry.hpp`:

- `ModelFamily{name, architectures, validate, describe, reader_limits}`:
  - `architectures`: the `general.architecture` values the family admits.
  - `validate`: structural admission of a parsed file. It throws `std::invalid_argument` naming the first violation.
  - `describe`: a one-line summary for logs and receipts.
  - `reader_limits`: GGUF reader limits the family needs, for example numeric metadata arrays it keeps through
    `Limits::max_retained_array_elements`.
- `register_model_family(family)`: re-registering an identical family is allowed. Clashing names or architectures throw.
- `find_model_family(architecture)`, `registered_model_families()`.
- `admit_model_file(path)` reads the architecture key with default limits. It then re-reads the file with the
  family's limits and validates it. It returns `{family, info}` and throws when no registered family admits the file.

Registration is an explicit call, for example `strix::deepseek41::register_family()`. Static libraries therefore
keep their registration without linker tricks: a program registers exactly the families it links.

## Adding a family `foo`

1. **Contract.** Add `include/strix/foo_model.hpp` and `src/foo_model.cpp`. Read the GGUF metadata into the family's
   geometry and build the expected tensor manifest: names, types, shapes, absolute offsets and byte sizes.
   `validate_target(info)` throws on the first mismatch. Admission by SHA-256 of pinned files is optional; see
   `deepseek41_model.hpp`.
2. **Registration.** Add `src/foo_family.cpp` with `register_family()` filling a `ModelFamily` from the contract.
3. **Execution plan.** Add `include/strix/foo_plan.hpp`. It describes one decode step:
   - the ordered operations;
   - the weights each one reads (checked against the manifest and the storage type expected per operation);
   - the activation buffers they pass and the persistent state they touch;
   - the owner and the numerical identity of each operation.
   See `deepseek41_plan.hpp`. Runners interpret the plan, so host-only, mixed and device-only runs of the same plan
   compare operation by operation.
4. **Pieces.** For each piece, write a host reference that declares its arithmetic order, plus device kernels that
   reproduce it bit for bit, in new files (`foo_*.hpp`, `src/foo_*.cpp`, `src/hip_foo_*.hip`).
   - Reuse core libraries through their interfaces: format decoders and products, MoE window routes, A8, the
     communicator.
   - Do not template or edit another family's hot kernels. If a shape differs, write a new kernel. Changing a shared
     core kernel requires an identical per-function ISA diff for every registered family, or that family's full gates.
5. **Runners and the gate.** Add a host forward over the plan and a device forward that hashes every operation's
   outputs the same way. A lockstep app runs both in one process and fails on the first differing operation (see
   `strix-deepseek41-lockstep`). Compare the host forward against the published reference implementation on the
   same layers; for deepseek41 this is the off-repo `run_ref.py` harness.
6. **CMake.** Add `option(STRIX_FAMILY_FOO "Build the foo model family module" ON)` and an
   `if(STRIX_FAMILY_FOO) ... endif()` block:
   - the libraries `strix_family_foo` (contract, plan, registration; links `strix_core`), the piece libraries and
     runtimes;
   - HIP targets under `STRIX_ENABLE_HIP_RUNTIME` with the strict arithmetic flags (`-ffp-contract=off
     -fno-fast-math -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee`, wave32);
   - its tests.

   Nothing outside the block may reference the family's targets.
7. **Gates before merging to main.**
   - A core-only build (every family OFF) builds and passes ctest.
   - Each family alone builds and passes its tests.
   - Every existing family's gates still hold. For Qwen these are: the full ctest, the goldens and MTP replay 0 for
     both profiles, 32-token identity at 1K/4K/16K, fast-profile identity, an unchanged ISA of its kernels versus
     main, and a decode/prefill smoke within noise.

## deepseek41 as the worked example

| piece | files | library |
|---|---|---|
| contract, plan, registration | `deepseek41_model`, `deepseek41_plan`, `deepseek41_family` | `strix_family_deepseek41` |
| formats, A8, MoE windows | `ds41_formats`, `ds41_moe`, `hip_ds41_formats` | `strix_ds41_formats`, `strix_ds41_hip` |
| layer sites, Engram | `ds41_layer`, `ds41_engram`, `hip_ds41_layer` | `strix_ds41_layer`, `strix_ds41_layer_hip` |
| attention | `ds41_attention`, `hip_ds41_attention` | `strix_ds41_attention`, `strix_ds41_attention_hip` |
| prefill experts | `ds41_prefill`, `hip_ds41_prefill` | `strix_ds41_prefill`, `strix_ds41_prefill_hip` |
| trimmed runners | `deepseek41_host_forward`, `deepseek41_attention_host`, `hip_deepseek41_device_forward` | `strix_ds41_runtime`, `strix_ds41_device_forward` |

The only core change deepseek41 needed is in the GGUF reader: `Limits::max_retained_array_elements`, default 0,
keeps top-level numeric arrays such as Engram primes and multipliers or per-layer ratios. It is opt-in and leaves
the default behaviour unchanged.
