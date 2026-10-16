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
