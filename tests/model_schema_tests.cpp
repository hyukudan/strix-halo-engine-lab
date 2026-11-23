#include "inkling2/inkling2_mini_config.hpp"
#include "inkling2/inkling2_mini_schedule.hpp"
#include "model/constants_profiles.hpp"
#include "strix/model/bind.hpp"
#include "strix/model_data/inkling2_mini.hpp"
#include "strix/profiles/qwen_cuda.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

bool has(std::string_view text, std::string_view part) { return text.find(part) != std::string_view::npos; }

constexpr auto qwen_lever_names = [] {
    constexpr auto& hip = strix::qwen::profile::table;
    constexpr auto& cuda = strix::qwen::cuda_profile::table;
    std::array<std::string_view, std::size(hip) + std::size(cuda)> names{};
    auto next = names.begin();
    for (const auto& row : hip) *next++ = row.name;
    for (const auto& row : cuda) *next++ = row.name;
    return names;
}();

std::string generate(std::string_view sections, std::span<const std::string_view> lever_names = qwen_lever_names) {
    const auto text = "schema = \"strix-model/v1\"\nname = \"qwen3.8-flash-next\"\n" + std::string(sections);
    return strix::model::emit_constants(strix::model_description::parse(text), "fixture.toml", lever_names);
}

void refused(std::string_view sections, std::string_view diagnosis,
             std::span<const std::string_view> lever_names = qwen_lever_names) {
    try {
        (void)generate(sections, lever_names);
    } catch (const std::invalid_argument& error) {
        const auto expected = "fixture.toml: " + std::string(diagnosis);
        require(std::string_view(error.what()).starts_with(expected), error.what());
        return;
    }
    throw std::runtime_error("expected refusal for " + std::string(diagnosis));
}

void schema() {
    const auto header = generate(R"(
[drafter]
kind = "block"
block = 8
taps = [0, 5, 11]
layers = 3
windows = [128, 1024]
rank = 64
head_screen = "markov-256"
weights = "q8_0"
max_window = 8
rollback = "position_rings"
[weights.format]
attention = "q8_0"
head = "BF16"
dense = "plain.f16@1"
experts = ["nvfp4", "mxfp8"]
drafter = "gguf.q8_0"
[fusion]
groups = ["qkv", "gate_up", "conv4.clip.kv_append"]
[dims]
layers = 24
[schedule]
core_first = 6
core_last = 18
core_passes = 3
residual_block_layers = 6
[profile.fast]
gdn_fp16_chunk = "1"
moe_route_bf16 = "1"
qsa_budget = "1536"
[profile.turbo]
gdn_fp16_chunk = "off"
moe_route_bf16 = ""
moe_route_prune = "rel0.02"
)");
    require(has(header, "namespace drafter {\ninline constexpr std::int64_t block = 8;"), "drafter block");
    require(has(header, "taps[] = {0, 5, 11};"), "drafter taps");
    require(has(header, "windows[] = {128, 1024};"), "drafter windows");
    require(has(header, "max_window = 8;"), "drafter staging window capacity");
    require(has(header, "rollback = \"position_rings\";"), "drafter ring rollback");
    require(has(header, "experts[] = {\"nvfp4\", \"mxfp8\"};"), "mixed expert role formats");
    require(has(generate("[weights.format]\ndense = \"exl3\"\n"),
                "inline constexpr std::string_view dense = \"exl3\";"),
            "registered EXL3 short storage name is accepted");
    require(has(header, "groups[] = {\"qkv\", \"gate_up\", \"conv4.clip.kv_append\"};"), "fusion groups");
    require(has(header, "core_last = 18;"), "exclusive loop bound");
    require(has(header, "namespace turbo {\n"
                        "inline constexpr std::string_view gdn_fp16_chunk = \"off\";\n"
                        "inline constexpr std::string_view moe_route_bf16 = \"1\";\n"
                        "inline constexpr std::string_view moe_route_prune = \"rel0.02\";\n"
                        "inline constexpr std::string_view qsa_budget = \"1536\";\n"), "turbo override and inheritance");
    require(has(generate("[profile.fast]\nqsa_budget = \"1536\"\n"),
                "namespace turbo {\ninline constexpr std::string_view qsa_budget = \"1536\";"),
            "absent turbo table inherits fast");
    require(has(generate("[profile.turbo]\nqsa_budget = \"\"\n"),
                "namespace turbo {\ninline constexpr std::string_view qsa_budget = \"\";"),
            "empty turbo cell without model fast defers to the catalogue");
    require(has(generate("[drafter]\nkind = \"chain\"\ntaps = [\"trunk.hidden\"]\nrollback = \"replay\"\n"),
                "rollback = \"replay\";"), "drafter replay rollback");

    refused("[drafter]\nkind = \"unregistered\"\n", "drafter.kind: unknown drafter kind 'unregistered'");
    refused("[drafter]\nkind = \"block\"\nblock = 0\n", "drafter.block: integer out of range");
    refused("[drafter]\nkind = \"block\"\n", "drafter.block: missing field");
    refused("[drafter]\nkind = \"chain\"\nrollback = \"undo\"\n", "drafter.rollback: unknown rollback mode 'undo'");
    refused("[drafter]\nkind = \"chain\"\nweights = \"bad\"\n", "drafter.weights: unknown weight format 'bad'");
    refused("[drafter]\nkind = \"chain\"\nwindows = [-1]\n", "drafter.windows: integer out of range");
    refused("[weights.format]\nrouter = \"bf16\"\n", "weights.format.router: unknown field");
    refused("[weights.format]\nhead = \"bad\"\n", "weights.format.head: unknown weight format 'bad'");
    refused("[fusion]\ngroups = [\"qkv\", \"bad\"]\n", "fusion.groups: unknown fusion group 'bad'");
    refused("[fusion]\ngroups = [\"qkv\", \"qkv\"]\n", "fusion.groups: duplicate fusion group 'qkv'");
    refused("[dims]\nlayers = 24\n[schedule]\ncore_first = 6\ncore_last = 25\ncore_passes = 3\n",
            "schedule.core_last: invalid exclusive loop bound");
    refused("[profile.fast]\nbad = \"1\"\n", "profile.fast.bad: unknown profile lever");
    refused("[profile.turbo]\nbad = \"\"\n", "profile.turbo.bad: unknown profile lever");
    refused("[profile.fast]\nmsa_flash = \"1\"\n", "profile.fast.msa_flash: unknown profile lever");
    refused("[profile.fast]\nqsa_budget = 1536\n", "profile.fast.qsa_budget: expected a profile cell string");
    refused("[profile.fast]\nqsa_budget = \"1536\"\n", "profile.fast.qsa_budget: unknown profile lever", {});
    constexpr std::array<std::string_view, 1> supplied{"custom_lever"};
    require(has(generate("[profile.fast]\ncustom_lever = \"on\"\n", supplied),
                "namespace turbo {\ninline constexpr std::string_view custom_lever = \"on\";"),
            "caller-supplied lever inherits into turbo");
    // One lever's row (profile.<lever>: tier cells plus the lever's own attributes) for a known lever.
    require(has(generate("[profile.custom_lever]\nturbo = \"fp8\"\nidentity = \"x-v1\"\n", supplied),
                "namespace custom_lever {\ninline constexpr std::string_view identity = \"x-v1\";\n"
                "inline constexpr std::string_view turbo = \"fp8\";"),
            "known lever row is emitted as is");
    refused("[profile.bad]\nturbo = \"on\"\n", "profile.bad: unknown profile lever", supplied);
    refused("[profile.custom_lever]\nturbo = 1\n", "profile.custom_lever.turbo: expected a profile cell string",
            supplied);
}

namespace generated = strix::model_data::inkling2_mini;
namespace ink = strix::inkling2;
namespace m = strix::model;

static_assert(generated::dims::context == ink::kMaxContext);
static_assert(generated::dims::window == ink::kSlidingWindow);
static_assert(generated::dims::hidden == ink::kHiddenWidth);
static_assert(generated::dims::vocab == ink::kVocabulary);
static_assert(generated::dims::vocab_out == ink::kOutputVocabulary);
static_assert(generated::dims::eos_token == ink::kMiniEosToken);
static_assert(generated::dims::head_dim == ink::kHeadWidth);
static_assert(generated::dims::heads == ink::kQueryHeads);
static_assert(generated::dims::kv_heads == ink::kKvHeads);
static_assert(generated::dims::query_width == ink::kQueryWidth);
static_assert(generated::dims::kv_width == ink::kKvWidth);
static_assert(generated::dims::relative_width == ink::kRelativeWidth);
static_assert(generated::dims::relative_extent == ink::kRelativeExtent);
static_assert(generated::dims::conv_taps == ink::kConvTaps);
static_assert(generated::dims::experts == ink::kExperts);
static_assert(generated::dims::top_k == ink::kExpertsPerToken);
static_assert(generated::dims::ffn == ink::kExpertWidth);
static_assert(generated::dims::rank == ink::kControllerRank);
static_assert(generated::moe::mxfp8_first == ink::kMxFp8First);
static_assert(generated::dims::layers == ink::kPhysicalLayers);
static_assert(generated::schedule::core_first == ink::kCoreFirst);
static_assert(generated::schedule::core_last == ink::kCoreLast);
static_assert(generated::schedule::core_passes == ink::kCorePasses);
static_assert(generated::schedule::residual_block_layers == ink::kResidualBlockLayers);

constexpr bool same_bits(double data, float engine) {
    return std::bit_cast<std::uint32_t>(m::exact_float(data)) == std::bit_cast<std::uint32_t>(engine);
}

static_assert(same_bits(generated::head::softcap, ink::kLogitSoftcap));
static_assert(same_bits(generated::moe::route_scale, ink::kRouteScale));
static_assert(same_bits(generated::moe::gate_softcap, ink::kMlpGateSoftcap));
static_assert(same_bits(generated::moe::up_softcap, ink::kMlpUpSoftcap));

void drift() {
    const std::string root = STRIX_SOURCE_DIR;
    const std::string source = "models/inkling2-mini.toml";
    const auto desc = strix::model_description::load(root + "/" + source);
    std::ifstream file(root + "/include/strix/model_data/inkling2_mini.hpp", std::ios::binary);
    require(file.is_open(), "open generated Inkling header");
    const std::string header{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    require(header == m::emit_constants(desc, source), "Inkling TOML/header drift");
}

}  // namespace

int main() {
    try {
        schema();
        drift();
        std::puts("PASS model schema drafter/weights/fusion/schedule/profile");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL model schema: %s\n", error.what());
        return 1;
    }
}
