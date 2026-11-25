"""Compile the production host replay and IU4 dispatch without HIP or device work."""
import pathlib
import re
import sys

root, out = map(pathlib.Path, sys.argv[1:])
source = (root / "src/hip_deepseek41_prefill.hip").read_text(encoding="utf-8")
setter = source.split("void DevicePrefill::set_precision_for_new_sequence(const PrefillOptions& p)", 1)[1].split("\n}", 1)[0]
copies = "\n".join(re.findall(r"    o\.decoder_replay_\w+ = p\.decoder_replay_\w+;", setter))
resolution = re.search(r"    const std::uint32_t bounded = .*;", source).group(0)
resolution = resolution.replace("m.options", "o").replace("prefill_decoder_replay_rows()", "inherited")
report = "\n".join(re.findall(r"    result\.decoder_replay_\w+ = \w+;", source))
iu4_copies = "\n".join(m.group(0) for m in re.finditer(r"    o\.(fast_moe(?:_f16)?|fast_dense|fast_i4) = p\.\1;", setter))
scope = re.search(r"    const ds41::PrefillPrecisionScope precision_scope[^;]+;", source).group(0)
dispatch = (root / "src/hip_ds41_prefill.hip").read_text(encoding="utf-8")
iu4_dispatch = re.search(r"int fast_i4_mode = -1;\n.*?\nbool fast_i4\(\) \{.*?\n\}", dispatch, re.S).group(0)
(out / "iu4_host.hpp").write_text(
    "namespace strix::ds41 {\n" + iu4_dispatch + "\n}\n"
    "inline void apply_iu4(PrefillOptions& o, const PrefillOptions& p) {\n" + iu4_copies + "\n}\n"
    "inline bool resolve_iu4(const PrefillOptions& o) {\n"
    "    namespace ds41 = strix::ds41;\n    const struct { const PrefillOptions& options; } m{o};\n"
    + scope + "\n    return ds41::fast_i4();\n}\n", encoding="utf-8")
(out / "replay_host.hpp").write_text(
    "inline void apply_replay(PrefillOptions& o, const PrefillOptions& p) {\n" + copies + "\n}\n"
    "inline PrefillResult resolve_replay(const PrefillOptions& o, std::uint32_t inherited, std::uint32_t P0 = 0U) {\n"
    + resolution + "\n    const bool bounded_oracle = false;\n    PrefillResult result;\n"
    + report + "\n    return result;\n}\n", encoding="utf-8")
