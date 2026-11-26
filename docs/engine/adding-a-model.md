# Adding a model

Spanish version: [adding-a-model.es.md](adding-a-model.es.md).

Registering descriptive metadata for an already supported compiled contract requires one description file and one
registry row. These files do not add inference support for a new size or checkpoint contract. A changed contract still
requires family adapter/engine work and qualification, and a new architecture also needs its own family module; the
common parts it plugs into are listed at the end.

## Supported contract: two files

### 1. The description, `models/<name>.toml`

Start from the file of the same family (`models/qwen3.8-flash-next.toml`, `models/deepseek-v4.1-flash.toml`,
`models/glm-5.3-flash.toml`, `models/minimax-m3.1-flash.toml`, `models/nemotron-3.5-bolt-e4b.toml`). Tools:

- Qwen: `strix-model-describe MODEL.toml [FILE.gguf]` loads a description and, with a GGUF, checks the two against each
  other.
- DeepSeek: `strix-deepseek41-model-describe --config-json CONFIG.json|--gguf MODEL.gguf --output MODEL.toml` writes a
  description from a config or a GGUF.
- GLM: `src/glm/load/glm_model_description.hpp` (`validate_model_directory`); MiniMax:
  `include/strix/minimax/model_description.hpp`; Nemotron: `src/gemma4/model/description.hpp`.

Fields common to all files: `schema = "strix-model/v1"` and a unique `name`. The remaining tables (`[source]`, `[dims]`,
attention, MoE, formats) are read by the family adapter, which is the only code that interprets them. The file is
descriptive: it selects no kernel, schedule or switch and prints no identity. What is checked, and against what, differs
per family; see the scope list below.

Optional `[multi_device]` (read by the shared profile resolver, `include/strix/profile/multi_device.hpp`): the
placement forms the model admits on several devices (`replica`, `layers`, `tensor`, `speculation`, with their cut,
rows, wire and drafter data) and ordered `regimes` that pick a form by phase, prompt rows, ready requests and tier.
A model without it resolves `single` on any number of devices. The wire `reference` keeps every exact join; a
narrower wire never resolves under EXACT. `strix run|serve --device-count N --placement FORM --print-effective-config`
prints the decision and who executes it (`executed_by`). Inspection and execution are distinct: a `run` / `serve`
launch executes one engine, so it is admitted only when the decision is `single`; `replica` is executed only by
`strix-router`, which holds the endpoints (start each worker with `--device-count 1`, then the router with the
worker addresses); any other form is refused with its reason until an engine consumes it. A regime may name only
declared forms, and its `wire` only forms that carry one (`layers`, `tensor`, `speculation`). Engine-local switches (`STRIX_*_PF_PIPE_CUT`, ...)
still act inside the engine; the effective config lists the ones that are set under `multi_device.legacy_env`. That
inventory is the model's own data: a model declares the switches its engines read under `[multi_device.legacy_env]`
(`STRIX_NAME = "forms.layers.cut"`, the key planned to replace it), and only the selected model's entries are listed. A
section with only `legacy_env` declares no placement policy. A regime `wire` that differs from the form's `wire` replaces it
in the report (`wire`, `settings.wire`); the form's `declared_wire` is shown beside it, and a bit-changing wire is
report-only (`wire_status`) until the shared resolver resolves a qualified lever for it (`lever` is parsed, not resolved).

The reader (`src/model_description.cpp`) accepts a TOML subset: bare and dotted keys, tables, inline tables, arrays,
single-line strings, decimal integers and floats, booleans. Anything else is an error with line and column.

### 2. The registry row, `models/registry.toml`

Add one table `[models.<id>]` (`schema = "strix-registry/v1"` is set once at the top of the file):

| key | content |
|---|---|
| `name` | equals `name` in the description |
| `description` | repo-relative path of the description |
| `family` | engine family (`qwen`, `deepseek41`, `glm5`, `minimax`, `gemma4`) |
| `cmake` | CMake option that builds the family (empty: always built) |
| `profile_tag`, `allow_pending_env` | profile table tag and its allow-pending variable (empty when the engine has none) |
| `defaults` | source files holding the engine's switch and defaults tables (array, not empty) |
| `apps` | entry points (array, not empty) |
| `adapter` | header of the family adapter validating the description (array) |

The registry is data. No engine reads it, and it adds no inference support. `src/model_family_registry.cpp` is a different,
run-time registry (the GGUF architectures a family admits); `models/registry.toml` only lists what exists.

### Scope of the description check per family

The description is checked against what each engine actually reads; nothing here is generic.

- **Qwen:** decode admission compares the GGUF metadata that is present with the compiled contract
  (`STRIX_QWEN_MODEL_DESCRIPTION_CHECK=0` switches it off). A description file is loaded only when
  `STRIX_QWEN_MODEL_DESCRIPTION` names it.
- **DeepSeek:** inspection and export only (`strix-deepseek41-model-describe`); no inference hook.
- **GLM:** model load validates a deployed `glm-5.3-flash.toml` and, when `config.json` exists, the keys it carries;
  a directory without `config.json` still loads. `STRIX_GLM_MODEL_DESCRIPTION_CHECK=0` skips the check.
- **MiniMax:** loading validates a deployed `minimax-m3.1-flash.toml` and the role and dimension assertions on the
  safetensors headers (structural checks always stay). `STRIX_MM_MODEL_DESCRIPTION_CHECK=0` skips the sidecar and the
  assertions.
- **Nemotron:** inspection and export only; no inference hook.

The adapters compare against compiled names, dimensions and templates. Changing the name or a dimension in a TOML and
adding a registry row therefore does not make a different size run.

### Commands

From the repository root, host only (no GPU, no model files):

```sh
cmake -S . -B build
cmake --build build --target strix-model-registry-tests strix-model-description-tests
ctest --test-dir build -R "registry|description"
```

`strix-model-registry-tests` fails when a row points at a missing file, when its `name` differs from the
description's, when two rows share a name, or when a `models/*.toml` has no row. It checks names, file existence and
coverage; it does not prove support and does not run the family adapters on a new row. The family tests are
`strix-deepseek41-model-description-tests`, `strix-minimax-model-description-tests` (`-DSTRIX_FAMILY_MINIMAX=ON`),
`strix-glm5-model-description-tests` (`-DSTRIX_FAMILY_GLM5=ON`) and `strix-gemma4-description-tests`
(`-DSTRIX_FAMILY_GEMMA4=ON`).

The isolated host gate needs only a C++20 compiler:

```sh
cmake -S tests/engine_profile_host -B host-gate
cmake --build host-gate
ctest --test-dir host-gate
```

It builds the profile, defaults, receipt and route-identity tests and the switch-table check.

If you add or change a switch row in an engine's defaults table, regenerate the table and commit it:

```sh
host-gate/strix-optimization-table > docs/engine/optimization-table.md
```

(multi-config generators put the executable in a `Release/` subfolder). `strix-optimization-table-current` fails while
the committed file differs from the generated one.

If the change touches `src/minimax/`, `src/dspark_common/`, `apps/minimax_*` or `CMakeLists.txt`, the reviewed source
receipt `tests/minimax_route_identity_sources.inc` no longer matches. Do not update the hashes to silence the check:
audit the changed or new route selectors, update the allow-table in `tests/minimax_route_identity_tests.cpp` if
needed, then recompute the FNV-1a hash of each listed file (CR bytes ignored, as the test does) and run
`strix-minimax-route-identity-tests --audit-sources`.

### Before calling the model supported

Host checks do not prove a numerical result. On the device:

1. Run the engine's built-in exactness gates for the default configuration and compare complete token ids and
   logits/state digests against a reference established independently (`AGENTS.md`). `tools/strix-gate` compares command
   outputs and digests between two builds; it is not an engine gate by itself.
2. Check the printed configuration, route and profile lines and the execution receipt ([receipts.md](receipts.md),
   schema [receipt-v1.schema.json](receipt-v1.schema.json)): identity fields and digests must be filled in, not empty.
3. For a family with a load-time check (Qwen, GLM, MiniMax), confirm that a deliberately mismatching file is refused and
   that the check's OFF switch restores the previous admission; the printed configuration, route and receipt must be
   equal with and without it.
4. Record each shipped switch with its measurement and receipt in `docs/measured-gains.md`.

## New architecture

Follow the [engine binding guide](../adding-an-engine.md) for the family adapter and registration. Keep tensor names,
layouts and stage bindings in the family. Extend the shared mechanism through a trait when the family needs a new
operation variant. The shared implementations include operators and kernels (`src/ops`, `src/core`, `src/cuda/core`),
pair exchange (`src/xport`, `include/strix/pair`) and speculative drafting (`src/spec`, `include/strix/spec`).

When a binding consumes generated constants, use `strix-model-gen` after changing its TOML and run
`strix-model-constants-tests` to verify that the checked-in header matches the source. Shared graph and executor
bindings live in `src/graph` and `src/exec`. The environment switch census in `tools/env_switch_census.py` checks
raw switch reads and conflicting defaults.
