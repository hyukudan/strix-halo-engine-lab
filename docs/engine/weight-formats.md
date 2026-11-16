# Weight formats and numerical contracts

The modular weight layer keeps a tensor's storage interpretation separate from
its operand image, backend and arithmetic identity. A container parser recognising
a type does not make that type executable. The initial implementation preserves
all existing production choices and numerical contracts.

The [binding census](../../tests/wfmt/bindings.json) records the audited source
revision, every recognised GGUF geometry, typed-only and parse-only entries,
legacy consumer sites and the owned migration work items P02–P14. Safetensors
recognition is recorded separately from model admission. Derived CB4/U5/U6,
Q6P32 and prepared operands are images; they are not additional GGUF wire types.
MXFP4 and native NVFP4 remain distinct storage contracts.

`wfmt.census` checks the reader's complete geometry switch and inventories
format-related operations in each declared source file. Adding or removing an
operation requires reviewing its binding and migration assignment. The keyword
scan is a closure guard for the declared files, not a substitute for reviewing
new consumer files. Include each new consumer in both the binding entries and
`closure_files`. Architecture scaffolds remain unavailable until implemented and
qualified; a source appearing in the census is not a qualification receipt.
DS41 decoder declarations and both CUDA Qwen geometry/decoder switches are also
checked against per-site storage lists. Derived representations use `images`,
separately from `formats`; Engram denotes custom E4M3 codes/E8M0 group-scale storage.
Other sites still require manual per-operation review: a name scan also sees calls
and comments and does not establish exhaustive numerical-variant coverage.

Arithmetic identities remain separate for CPU integer products/tree32, Intel
F16 operands/ascending reduction, HIP staged F16 products, CUDA four-chain IMMA,
native IQ FP32, DS41 group products and affine correction, transformed trellis,
Gemma EXL3 variants, H3 transformed operands and tolerance-qualified NVFP4 MMA.
An existing public identity is preserved; a new registry name does not change
its arithmetic or qualification status.
The current descriptors are an inventory, not complete arithmetic specifications.
Before migration, freeze unspecified rounding, reduction, tiny-input, accumulation
and transform boundaries and qualify them with operation-specific byte captures.
Registry/source entrypoints are declarations at this stage, not a usable weight API.
Borrowed strings/spans require stable backing storage that outlives every copy;
strings remain immutable, while spans permit mutation according to their types.
Dynamic registries must own stable storage for them.

Use the [weight-format comparison kit](../../scripts/bench/wfmt/README.md) to
build host cases and collect versioned byte records before and after each
migration. Receipts identify the exact source, compiler, flags, architecture,
shape, source bytes and arithmetic. Comparison requires equal lengths and byte
contents; hashes alone are insufficient. Intermediate hooks are test-only and
introduced with the corresponding migration. Store generated captures, binaries,
logs and private model inputs outside Git. Host-only census/contract changes do
not claim device execution or new numerical qualification.

Home: shared (host shared contracts and test infrastructure). The shared layer
serves all model families; vendor operand sources and instruction specialisations
are introduced in their assigned migration steps.
