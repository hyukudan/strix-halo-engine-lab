# Persisted offline tuning preparation

`--offline-tuning-cache FILE` explicitly reads the bounded versioned codec once
before GPU setup. The same byte buffer is hashed and decoded. Preparation owns
the records and per-step keys; only compilation queries the existing registry.
There is no file load, model hash or tuning search in the token loop.

Live keys use the actual selected baseline registry variant, selected tensor
formats, resolved model/layout/context/submission configuration, the existing
mapped-model content SHA, one executable SHA and runtime-reported SKU/processor
count/HIP versions. The legacy key field named compute_units contains the HIP
reported multiprocessor count; its domain is explicitly labelled and must not
be read as a physical CU proof. Runtime environment settings are part of the
resolved preparation context; they are not assertions about effective overlap.

The build embeds DECLARED CMake toolchain/global flag context, not observed or
attested compiler argv. Heterogeneous per-source HIP options are not completely
represented; therefore the live compiler-flags SHA is currently unknown. It
stays empty for selection and null in the receipt. The consumer consequently
reports invalid_key fallback instead of borrowing a digest from the cache. A
future complete build representation and independently qualified numerical
classes are needed before this source can select such records.

Registry qualification remains the sole authority: numerical class zero stays
unknown, experimental variants cannot be promoted by records, and a comparable
qualified baseline measurement is mandatory. Source receipts expose exact
selection/fallback reasons and do not invent costs, qualification or speedups.
Defaults do not read a cache, query extra metadata or hash the executable.

## Optional evaluated compile-input provenance

A source implementation now accepts `--offline-compile-provenance FILE` together
with `--offline-tuning-cache`. Without that explicit input, the unknown flags key
and fallback above are unchanged. No manifest is generated or admitted by the
application. The representation supplies a conservative versioned evaluated-argument/declared-input digest
(arguments plus target/source/header/object/toolchain fingerprints), not merely
a hash of one generic HIP flag string. Registry numerical qualification remains
independent; this option never qualifies a kernel or changes numerical class0.

The supported description tool is
`tools/qualification/prepare_compile_provenance.py`. It supports single-config
Ninja POSIX and CMake FileAPI codemodel-v2/toolchains-v1 only. Configure with
`STRIX_EXPORT_OFFLINE_TUNING_PROVENANCE=ON` to request the evaluated compile
command database and FileAPI replies. After the actual build is complete, an
operator must separately export valid Ninja dependencies for that build. Neither
the description tool nor the decoder launches Ninja, compilers or probes.

The tool takes `--build`, exact `--codemodel` and `--toolchains` reply paths,
`--binary`, `--build-context-header` (the generated header),
`--ninja-deps-export`, exact `--linker`, and a fresh `--output`. Target defaults to
`strix-hip-decode-token`, configuration to `Release`. It closes FileAPI target
dependencies and matches each compiled source to one target-specific evaluated
command/object. Explicit dependency exports must be VALID and match object
nanosecond timestamps; missing headers/objects, ambiguous commands, launchers,
response files, unsupported generators/configurations or changed inputs refuse
publication. Inputs are rehashed before atomic no-overwrite publication. The
manifest contains arguments, paths and fingerprints; keep it as build evidence,
not model assets or secrets. No actual manifest has been generated in this stage.

The bounded setup-only loader checks the running executable inode SHA once,
exact embedded DECLARED build context, typed project-target/unit/dependency closure,
counts/extents and every referenced current file's bytes/hash before GPU setup.
The binary record is checked against that already hashed process image; it is not
rehashed through a replaceable executable path. Explicit malformed/stale input
is rejected. Absent or unsupported provenance stays unknown instead of obtaining
facts from a tuning record. Existing installed artifacts without their build
inputs therefore cannot use this optional path.

This is an evaluated-input provenance description, NOT observed compiler argv,
proof that a compiler ran with those inputs, or build attestation. A matching
binary fingerprint binds the description to that exact executable; source/object
fingerprints describe the retained build inputs without inventing history.
The argument/dependency digest is also not a correctness, cost or quality receipt.

Scope is evaluated per-TU argv plus declared input fingerprints bound to the
binary, not transitive runtime/toolchain closure. Explicit library/object paths
in link fragments are fingerprinted. `-l`/`-L` resolution, compiler-driver resource
subcommands and dynamically loaded runtime dependencies are not invented from
those fragments; their literal flags are preserved and the receipt states this
limit. Independent runtime library/model admission remains mandatory. The loader
caps a single referenced file/read at256MiB and the declared total at16GiB;
unsupported larger inputs cannot be represented by this version. No model bytes
are included or rehashed. The current-source fingerprint does not attest history.
