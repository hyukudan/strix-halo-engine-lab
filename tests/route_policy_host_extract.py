"""Compile production host policy/receipt bodies without HIP or device initialization.

Only GPU types and launch code are excluded. Resolver and receipt bodies are copied
verbatim from the source tree under test, not reimplemented. --source can point at
an archived PR head to demonstrate that the same behavioral regressions fail there.
"""
import argparse
from pathlib import Path


def block(text, start):
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def generate(source, output):
    def read(name):
        return (source / name).read_text(encoding="utf-8")

    exact = read("src/glm/prefill/glm_pf_exact.hip")
    selectors = exact[exact.index("unsigned x_auto_rows() {"):exact.index("unsigned vector_splits(")]
    begin = exact.index("    const auto selection = select_route(rows);") if "const auto selection = select_route(rows);" in exact else exact.index("    const Route selected = route(rows);")
    selected = exact[begin:exact.index("    const unsigned tm =", begin)]
    hook = exact[exact.index("    const unsigned core =", begin):exact.index("    if (core == 5U && aux::", begin)]
    exact_header = read("src/glm/prefill/glm_pf_exact.hpp")
    enum_start = exact_header.index("enum class Route")
    enum = exact_header[enum_start:exact_header.index(";", enum_start) + 1]
    stub = output / "stubs/glm/prefill/glm_pf_exact.hpp"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text("#pragma once\nnamespace strix::glm::pf::exact {\n" + enum +
                    "\nunsigned x_auto_rows();\nRoute route(unsigned rows);\n}\n", encoding="utf-8")
    minimax = read("src/minimax/engine/engine.hip")
    iu4 = read("src/minimax/moe/mmx_prefill_iu4.inc")
    qwen = read("apps/hip_decode_token_main.cpp")
    qwen_start = qwen.rfind("    {", 0, qwen.index("// Shared route receipt: the prefill route"))
    options = read("apps/hip_decode_options.hpp")
    pieces = {
        "GLM_SELECTORS": selectors,
        "GLM_HOOK": selected + "    const unsigned tm = 16;\n" + hook,
        "IU4_RESOLVER": block(iu4, iu4.index("bool prefill_iu4() {")),
        "MINIMAX_LEVERS": block(minimax, minimax.index("    void route_levers() const {")),
        "QWEN_LEVERS": block(qwen, qwen_start),
        "QWEN_PAIR": block(options, options.index("inline strix::NumericalProfilePair numerical_profile_pair(")),
    }
    template = Path(__file__).with_name("route_policy_host_tests.cpp.in").read_text(encoding="utf-8")
    for name, value in pieces.items():
        template = template.replace("@" + name + "@", value)
    (output / "route_policy_host.cpp").write_text(template, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.output)
