// Model constants derived from the model TOML (strix/model/constants.hpp, strix/model/bind.hpp).
// 1. Drift: every registry model has include/strix/model_data/<ident>.hpp equal, byte for byte, to the header the
//    generator derives from its TOML. The TOML is the single source; a stale or hand-edited header fails here.
// 2. Emission rules: identifiers, reserved words, exact reals, string escapes, skipped arrays, nested tables.
// 3. Bound engine constants keep their previous values exactly (the binding does not change machine code).
#include "strix/model/bind.hpp"
#include "strix/model/constants.hpp"
#include "strix/model_description.hpp"
#include "strix/model_registry.hpp"

#include "../src/gemma4/contracts/g4_identity.hpp"
#include "../tools/model_gen_levers.hpp"
#include "strix/qwen35/config.hpp"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>

#ifndef STRIX_SOURCE_DIR
#error "STRIX_SOURCE_DIR must name the repository root"
#endif

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL %s\n", what.c_str());
    }
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

bool has(const std::string& text, const std::string& needle) { return text.find(needle) != std::string::npos; }

void drift() {
    const std::string root = STRIX_SOURCE_DIR;
    const auto entries = strix::model_registry::load(root + "/models/registry.toml");
    check(entries.size() >= 7, "registry lists the declared models");
    for (const auto& entry : entries) {
        const auto description = strix::model_description::load(root + "/" + entry.description);
        const auto header = root + "/include/strix/model_data/" + strix::model::identifier(entry.name) + ".hpp";
        const auto levers = strix::model::profile_lever_names(entry.name);
        check(read_file(header) == strix::model::emit_constants(description, entry.description, levers),
              header + " equals the generator output (strix-model-gen --write " + entry.description + " " + header +
                  " " + entry.description + ")");
    }
}

void emission() {
    check(strix::model::identifier("nemotron-3.5-bolt-e4b") == "nemotron_3_5_bolt_e4b", "identifier");
    check(strix::model::identifier("3d") == "k3d", "identifier leading digit");
    const auto text = strix::model::emit_constants(
        strix::model_description::parse("schema = \"strix-model/v1\"\nname = \"e\"\n[t]\ndefault = 1\nr = -0.5\n"
                                        "big = 9007199254740993\ns = \"q\\\"\\\\\"\nempty = []\nmixed = [1, \"a\"]\n"
                                        "v = [1.5, 2.0]\n[t.u]\nflag = false\n"),
        "models/e.toml");
    check(has(text, "namespace strix::model_data::e {"), "model namespace");
    check(has(text, "inline constexpr std::string_view name = \"e\";"), "root string");
    check(has(text, "inline constexpr std::int64_t default_ = 1;"), "reserved word suffixed");
    check(has(text, "inline constexpr std::int64_t big = 9007199254740993;"), "integers are not routed via double");
    check(has(text, "inline constexpr double r = -0x1p-1 /* -0.5 */;"), "exact hexadecimal real");
    check(has(text, "inline constexpr double v[] = {0x1.8p+0 /* 1.5 */, 0x1p+1 /* 2 */};"), "real array");
    check(has(text, R"(inline constexpr std::string_view s = "q\"\\";)"), "escaped string");
    check(has(text, "// empty: empty array (no constant)"), "empty array skipped");
    check(has(text, "// mixed: array of tables or mixed array (no constant)"), "mixed array skipped");
    check(has(text, "namespace u {\ninline constexpr bool flag = false;\n} // namespace u"), "nested table");
    bool refused = false;
    try {
        (void)strix::model::emit_constants(strix::model_description::parse("schema = \"strix-model/v1\"\n"), "x.toml");
    } catch (const std::exception& e) {
        refused = has(e.what(), "x.toml: name is missing");
    }
    check(refused, "a description without name is refused");
}

// The binding helpers are constant expressions; an out-of-range value does not compile.
static_assert(strix::model::narrow<std::uint32_t>(2560) == 2560U);
static_assert(strix::model::narrow<std::int8_t>(-128) == -128);
static_assert(strix::model::exact_float(0.5) == 0.5F);

// Gemma4 (Nemotron 3.5 Bolt E4B) bound to models/nemotron-3.5-bolt-e4b.toml: the previous literals, unchanged.
namespace g4 = strix::gemma4;
static_assert(g4::hidden == 2560U && g4::layers == 42U && g4::inter == 10240U && g4::vocab == 262144U);
static_assert(g4::heads == 8U && g4::kv_heads == 2U && g4::group == 4U && g4::ple_dim == 256U);
static_assert(g4::window == 512U && g4::max_positions == 65536U && g4::attention_chunk == 64U);
static_assert(sizeof(g4::tap_layers) == 5 * sizeof(std::uint32_t) && g4::tap_layers[0] == 5U &&
              g4::tap_layers[1] == 11U && g4::tap_layers[2] == 17U && g4::tap_layers[3] == 23U &&
              g4::tap_layers[4] == 35U);
static_assert(g4::norm_eps == 1e-6F && g4::attention_scale == 1.0F && g4::logit_cap == 30.0F);
static_assert(g4::embedding_scale == 50.5F && g4::ple_embedding_scale == 16.0F);
static_assert(g4::model_ple_scale == 0.019764235376052371F && g4::ple_combine_scale == 0.707106781186547524F);

void qwen35_defaults() {
    // Qwen 27B Config defaults bound to models/qwen3.8-27b.toml: the previous literals, unchanged.
    const strix::qwen35::Config c{};
    check(c.layers == 64 && c.hidden == 5120 && c.intermediate == 17408 && c.vocab == 248320 && c.context == 262144,
          "qwen35 dims");
    check(c.rms_eps == 1e-6f && c.heads == 24 && c.kv_heads == 4 && c.head_dim == 256 && c.rotary_dim == 64 &&
              c.rope_theta == 1.0e7,
          "qwen35 attention");
    check(c.gdn_key_heads == 16 && c.gdn_key_dim == 128 && c.gdn_value_heads == 48 && c.gdn_value_dim == 128 &&
              c.gdn_conv_kernel == 4,
          "qwen35 gated delta net");
}

} // namespace

int main() {
    qwen35_defaults();
    drift();
    emission();
    if (failures) {
        std::fprintf(stderr, "strix-model-constants-tests: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("strix-model-constants-tests: PASS\n");
    return 0;
}
