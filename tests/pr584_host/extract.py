"""Compile production host orchestration with fake allocation/stream operations.

Only HIP launch syntax is replaced; kernel arithmetic and native event semantics
are deliberately outside this host fixture's coverage.
"""
from pathlib import Path
import re
import sys

root, out = map(Path, sys.argv[1:3])
def read(path):
    return (root / path).read_text()
source = read("src/hip_deepseek41_prefill.hip")



def function(marker):
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


methods = function("bool flash_attn_at(") + "\n"
if "void prepare_flash_mirrors()" in source:
    methods += function("void prepare_flash_mirrors()")
(out / "methods.inc").write_text(methods)
(out / "setter.inc").write_text(
    function("void DevicePrefill::set_precision_for_new_sequence(bool") + "\n" +
    function("void DevicePrefill::set_precision_for_new_sequence(const PrefillOptions&"))
start = source.index("                    fe.ring_pos = m.ppos64.p + pk0;")
end = source.index("                    if (carry_words && !last_local)", start)
unpack = source[start:end]
unpack = re.sub(r"final_entries_kernel<false><<<[^\n]+>>>\(fe\);", "unpack_entries(fe);", unpack)
if "<<<" in unpack:
    raise RuntimeError("unrecognized launch syntax")
# The common block must follow both receive paths and precede lane publication.
assert source.index("entries_exchange->finish(st)") < start
assert source.index("exchange_rows(m.pack_send.p, own_words") < start
assert end < source.index('hipEventRecord(m.chunk_ready, st)', end)
(out / "unpack.inc").write_text(unpack)

source = read("src/core/f16_dense_q8a8.hpp")
(out / "parts.inc").write_text(function("inline void enqueue_parts("))
source = read("src/hip_ds41_layer.hip")
(out / "dense.inc").write_text(function("static bool enqueue_fast_dense("))
