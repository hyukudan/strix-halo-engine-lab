# Strix command line

The `strix` command is in `strix-tools`. Install `strix-core` alongside it for the model registry and descriptions.
Launch commands also need a matching engine executable in the directory containing `strix`.
The registry describes model contracts. Check the installed executable and support receipts for backend availability.

| Command | Result |
|---|---|
| `strix --version` or `strix version` | Configured build identity as JSON |
| `strix doctor [--json]` | OS, CPU model and hardware ISA bits, RAM, runtime loading, Linux device-node access, installed support receipts and package candidates. Runtime loading leaves device initialization and qualification unchecked. |
| `strix devices [--json]` | Device declarations and observations from the shared discovery sources. Uses the same declaration lookup as engine startup. Discovery may be incomplete. |
| `strix models list [--json]` | Model names and families from the registry |
| `strix inspect MODEL [--json]` | Registered model description. Reads the descriptor metadata without opening a weight file. |
| `strix plan MODEL --context N [--devices FILE] --json` | Read-only plan using declared budgets and existing deployment providers. It performs no device work. |
| `strix update` or `strix rollback` | Prints “not available in this release” and exits 2 |

Doctor JSON contains a `checks` array. Each check has `name`, `status` (`ok`, `warn`, `fail`, or `skip`) and `detail`.
CPU flags report hardware capabilities. The engine decides whether a given ISA is usable.

For a memory plan, use a registered asset path and supply a device declaration. This slice can read MiniMax
safetensors headers and the registered Gemma-family GGUF header through their deployment providers.
The CLI treats declared budgets as planning ceilings. The result is a plan and carries no measurement or runtime
admission verdict. MiniMax layer-cut selection requires stage costs and qualification evidence. An unavailable
provider, missing inputs, or a refused plan returns exit 2 with a reason.

## Launch an installed engine

Use the name or ID from `strix models list` and put the model asset path first after `--`.
If the asset basename exactly matches a registry name or ID after stripping `.gguf`, you can use the asset path
directly as `MODEL`. Everything else after `--` goes to the engine in the supplied order.
`--dry-run` prints the resolved command without starting the engine.

```sh
strix run qwen3.8-flash-next --dry-run -- MODEL.gguf 1 2 3 --generate 256
strix serve qwen3.8-flash-next --host 127.0.0.1 --port 8080 -- MODEL.gguf
strix run deepseek-v4.1-flash -- MODEL.gguf --rank 0 --transport usb4auto
strix run minimax-m3.1-flash -- MODEL_DIR 0 usb4-auto OUTPUT_DIR
strix run glm-5.3-flash -- MODEL_DIR --world 2 --rank 0 --transport usb4-auto --prompt PROMPT.u32 --decode 64
```

Launch one command per rank for a pair. DeepSeek pair serving enables HTTP on rank 0 on port 8080 by default.
DeepSeek RTX `run` accepts the [standalone assets and arguments](../models/deepseek-v4.1/rtx-standalone.md#run-and-persistent-worker).
The RTX worker spool uses its own `--serve` argument after the CLI separator.

Unknown model names, unavailable family executables, missing pair launch arguments and unsupported
command/family combinations exit 2 before the engine opens weights. The native engine owns model validation,
remaining engine-specific arguments and execution defaults.
