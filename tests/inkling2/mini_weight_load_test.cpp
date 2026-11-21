#include "inkling2/inkling2_mini_checkpoint.hpp"
#include "inkling2/inkling2_mini_expert_layout.hpp"
#include "inkling2/inkling2_mini_schedule.hpp"
#include "strix/ggml_dequant.hpp"
#include "strix/profiles/inkling2_mini.hpp"
#include "strix/weight/host_q8.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

namespace mini = strix::inkling2;
namespace profile = strix::profile;
namespace weight = strix::weight;

void require(bool passed, const char* detail) {
    if (!passed) throw std::runtime_error(detail);
}

weight::Digest256 digest(std::uint8_t marker) {
    return {{marker}};
}

void test_conversion() {
    // A nonzero source offset, multiple blocks per row, and a staging boundary.
    constexpr std::size_t columns = 96, rows = 345, prefix = 3;
    const weight::Shape shape{{rows, columns}, 2};
    std::vector<weight::Byte> source(prefix + rows * columns * 2);
    std::vector<float> input(rows * columns);
    for (std::size_t i = 0; i < input.size(); ++i) {
        const auto lane = static_cast<int>((i * 7U) % 32U) - 16;
        const auto group = i / 32U;
        const float value = group % 11U == 0 ? 0.0F :
            std::ldexp(static_cast<float>(lane) / 16.0F, static_cast<int>(group % 20U) - 15);
        const auto bits = std::bit_cast<std::uint32_t>(value);
        input[i] = value;
        source[prefix + i * 2] = weight::Byte((bits >> 16U) & 255U);
        source[prefix + i * 2 + 1] = weight::Byte(bits >> 24U);
    }
    weight::Slice slice{weight::memory_source(source), {prefix, source.size() - prefix}};
    std::vector<weight::Byte> exact(slice.range.bytes);
    require(bool(weight::load_bf16_matrix(slice, shape, weight::kBf16WeightStorage, exact)), "BF16 passthrough");
    require(std::ranges::equal(exact, std::span(source).subspan(prefix)), "BF16 bytes changed");

    weight::MatrixLoadGeometry geometry{};
    require(bool(weight::matrix_load_geometry(weight::kQ8WeightStorage, shape, geometry)), "Q8 geometry");
    require(geometry.row_bytes == 102 && geometry.bytes == rows * 102, "Q8 compact block layout");
    std::vector<weight::Byte> packed(geometry.bytes);
    require(bool(weight::load_bf16_matrix(slice, shape, weight::kQ8WeightStorage, packed)), "Q8 conversion");

    const auto half_type = weight::gguf_type({"plain.f16", 1});
    const auto q8_type = weight::gguf_type(weight::kQ8WeightStorage);
    for (std::size_t row = 0; row < rows; ++row) {
        const auto raw = std::span(packed).subspan(row * geometry.row_bytes, geometry.row_bytes);
        std::array<float, columns> decoded{};
        strix::ggml_dequant::dequantize_row(q8_type, raw, decoded);
        for (std::size_t first = 0; first < columns; first += 32) {
            const auto values = std::span(input).subspan(row * columns + first, 32);
            float amax = 0;
            for (const float value : values) amax = std::max(amax, std::abs(value));
            const float scale = amax / 127.0F;
            std::array<float, 1> stored_scale{};
            strix::ggml_dequant::dequantize_row(half_type, raw.subspan(first / 32 * 34, 2), stored_scale);
            const float bound = 0.5001F * scale + 127.0F * std::abs(scale - stored_scale[0]);
            for (std::size_t lane = 0; lane < values.size(); ++lane)
                require(std::abs(decoded[first + lane] - values[lane]) <= bound, "Q8 round trip error bound");
            if (!amax) require(stored_scale[0] == 0, "zero block scale");
        }
    }

    require(weight::matrix_load_geometry(weight::kQ8WeightStorage, {{2, 33}, 2}, geometry).code ==
                weight::Error::unsupported_shape, "partial Q8 block accepted");
    require(weight::matrix_load_geometry(weight::kBf16WeightStorage, {{UINT64_MAX, 64}, 2}, geometry).code ==
                weight::Error::memory_budget, "matrix byte overflow accepted");
    require(weight::load_bf16_matrix(slice, shape, weight::kQ8WeightStorage, std::span(packed).first(1)).code ==
                weight::Error::malformed, "short destination accepted");
    const weight::Shape block{{1, 32}, 2};
    std::array<weight::Byte, 64> edge{};
    std::array<weight::Byte, 34> encoded{};
    edge[0] = weight::Byte{1}; // Smallest positive BF16.
    require(bool(weight::load_bf16_matrix({weight::memory_source(edge), {0, edge.size()}}, block,
                                          weight::kQ8WeightStorage, encoded)) &&
                std::ranges::all_of(encoded, [](auto byte) { return byte == weight::Byte{}; }),
            "Q8 underflow block must decode to zero");
    for (const auto high : {weight::Byte{0x7f}, weight::Byte{0x70}}) {
        edge[0] = weight::Byte{0x80};
        edge[1] = high;
        require(weight::load_bf16_matrix({weight::memory_source(edge), {0, edge.size()}}, block,
                                        weight::kQ8WeightStorage, encoded).code == weight::Error::malformed,
                "infinite input or overflowing scale accepted");
    }
}

void test_selection() {
    const auto& registry = weight::builtin_registry();
    const auto& policy = mini::kMiniWeightLoadPolicy;
    constexpr std::array formats{weight::kBf16WeightStorage, weight::kQ8WeightStorage};
    for (const auto backend : {weight::Backend::hip, weight::Backend::cuda}) {
        const weight::WeightLoadScope scope{backend, "test-device", "test-compiler", "decode", digest(1), digest(2)};
        const std::array receipts{weight::WeightLoadDecision{policy.id, scope, digest(3), digest(4)}};
        weight::WeightLoadRequest request{profile::Name::standard, scope, formats, true, true};
        const auto resolve = [&] { return weight::resolve_weight_load(registry, policy, request, receipts); };
        require(resolve().storage == weight::kBf16WeightStorage, "explicit qualification opt-in changed EXACT");
        request.tier = profile::Name::fast;
        request.allow_pending = false;
        require(resolve().status == profile::Status::set && resolve().storage == weight::kQ8WeightStorage &&
                    resolve().gate_v2_pass == digest(3) && resolve().measured_gain == digest(4), "qualified FAST selection");
        request.tier = profile::Name::turbo;
        require(resolve().storage == weight::kQ8WeightStorage, "TURBO omitted FAST storage");
        request.tier = profile::Name::fast;
        request.enabled = false;
        require(resolve().status == profile::Status::off && resolve().storage == weight::kBf16WeightStorage, "off switch");
        request.enabled = true;
        auto no_speed = receipts;
        no_speed.front().measured_gain = {};
        require(weight::resolve_weight_load(registry, policy, request, no_speed).storage ==
                    weight::kBf16WeightStorage, "missing measured gain");
        request.scope.checkpoint = digest(5);
        require(resolve().status == profile::Status::pending && resolve().storage == weight::kBf16WeightStorage,
                "unqualified checkpoint used FAST storage");
        request.allow_pending = true;
        require(resolve().status == profile::Status::unqualified && resolve().storage == weight::kQ8WeightStorage &&
                    resolve().storage_for(std::nullopt) == weight::kBf16WeightStorage, "qualification arm selection");
        request.consumer_formats = std::span(formats).first(1);
        require(resolve().status == profile::Status::pending && resolve().storage == weight::kBf16WeightStorage,
                "qualification arm requires a shared consumer");
    }
}

void test_token_bytes() {
    const auto specs = mini::mini_text_tensor_schema();
    std::array<std::uint64_t, mini::kPhysicalLayers> visits{};
    std::uint64_t expert_bytes = 0;
    for (const auto visit : mini::mini_schedule()) {
        ++visits[visit.physical_layer];
        for (const auto projection : {mini::MiniExpertProjection::gate_up, mini::MiniExpertProjection::down}) {
            const auto [inputs, outputs] = mini::mini_expert_geometry(projection);
            const auto layout = mini::mini_expert_format(visit.physical_layer);
            using weight::microscaled::MatrixFormat;
            const std::uint64_t row_bytes = layout == MatrixFormat::nvfp4 ?
                std::uint64_t{inputs} / 2U + std::uint64_t{inputs} / weight::microscaled::kNvfp4BlockColumns :
                std::uint64_t{inputs} + std::uint64_t{inputs} / weight::microscaled::kMxfp8BlockColumns;
            expert_bytes += row_bytes * outputs * mini::kExpertsPerToken;
        }
    }
    for (const auto format : {weight::kBf16WeightStorage, weight::kQ8WeightStorage}) {
        std::uint64_t dense = 0, head = 0, head_storage = 0;
        unsigned tagged = 0;
        for (const auto& spec : specs) {
            if (!spec.load_matrix_role) continue;
            ++tagged;
            require(spec.resident && spec.wire_dtype == "BF16" && spec.extents.size() == 2,
                    "load role requires BF16 matrix");
            weight::MatrixLoadGeometry geometry{};
            const weight::Shape shape{{spec.extents[0], spec.extents[1]}, 2};
            require(bool(weight::matrix_load_geometry(format, shape, geometry)), "tagged matrix geometry");
            if (*spec.load_matrix_role == weight::Role::head) {
                require(spec.name == "model.llm.unembed.weight", "head tensor mapping");
                head = geometry.row_bytes * mini::kOutputVocabulary;
                head_storage = geometry.bytes;
            } else {
                const auto layer_name = std::string_view(spec.name).substr(sizeof("model.llm.layers.") - 1);
                const auto layer = std::stoul(std::string(layer_name.substr(0, layer_name.find('.'))));
                dense += visits.at(layer) * geometry.bytes;
            }
        }
        require(tagged == 97, "weight plan must tag 24 sets of Q/K/V/O and one LM head");
        require(expert_bytes == 1839366144, "selected packed expert bytes");
        const bool bf16 = format == weight::kBf16WeightStorage;
        require(dense == (bf16 ? 1811939328U : 962592768U), "per-token Q/K/V/O bytes");
        require(head == (bf16 ? 921867264U : 489741984U), "per-token LM head bytes");
        require(head_storage == (bf16 ? 926318592U : 492106752U), "padded LM head storage");
        require(dense + head + expert_bytes == (bf16 ? 4573172736ULL : 3291700896ULL), "per-token total bytes");
    }
}

}  // namespace

int main() {
    try {
        test_conversion();
        test_selection();
        test_token_bytes();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
