#include "core/resident_row_plan.hpp"
#include "inkling2/inkling2_mini_checkpoint.hpp"
#include "inkling2/inkling2_mini_expert_layout.hpp"
#include "inkling2/inkling2_mini_schedule.hpp"
#include "inkling2/inkling2_mini_weight_options.hpp"
#include "strix/ggml_dequant.hpp"
#include "strix/profiles/inkling2_mini.hpp"
#include "strix/weight/host_q8.hpp"
#include "strix/weight/resident_slices.hpp"

#include <algorithm>
#include <cstdlib>
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

    weight::RawTensor tensor{};
    tensor.wire_dtype = "BF16";
    tensor.shape = shape;
    tensor.payload = slice;
    const auto resident = weight::ResidentSlicePlan::make({{"projection", &tensor, weight::kQ8WeightStorage},
                                                          {"exact", &tensor, weight::kBf16WeightStorage}}, 16);
    const auto ranges = resident.ranges();
    require(ranges.size() == 2 && ranges[0].bytes == packed.size() && ranges[1].source.source == slice.source &&
                ranges[1].source.range.offset == prefix && resident.bytes() < 2U * exact.size(),
            "resident arena did not place compact and exact images");
    std::vector<std::uint8_t> allocation(resident.bytes() + 15U);
    const auto misalignment = reinterpret_cast<std::uintptr_t>(allocation.data()) % 16U;
    auto device = std::span(allocation).subspan((16U - misalignment) % 16U, resident.bytes());
    for (const auto& range : ranges) {
        auto destination = std::as_writable_bytes(device.subspan(range.destination_offset, range.bytes));
        for (std::size_t first = 0; first < destination.size();) {
            const auto count = std::min(std::size_t{1003}, destination.size() - first);
            require(bool(range.source.source->read({range.source.range.offset + first, count},
                                                   destination.subspan(first, count))), "staged resident read");
            first += count;
        }
    }
    const auto images = weight::ResidentSlices::bind(resident, device);
    const auto q8 = images.matrix("projection");
    const auto bf16 = images.matrix("exact");
    require(q8.storage == weight::kQ8WeightStorage && q8.row_bytes == geometry.row_bytes &&
                q8.rows == rows && q8.columns == columns && std::ranges::equal(std::as_bytes(q8.bytes), packed),
            "Q8 binding geometry or staged bytes differ");
    require(bf16.storage == weight::kBf16WeightStorage && bf16.row_bytes == columns * 2U &&
                std::ranges::equal(std::as_bytes(images.bf16("exact")), exact), "BF16 resident bytes changed");
    require(ranges[0].source.source->identity() == weight::digest_bytes(packed), "converted source content digest");
    bool rejected = false;
    try {
        (void)images.bf16("projection");
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "packed bytes must never bind as BF16");

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
    const auto& policy = mini::kMiniWeightLoadPolicy;
    static_assert(weight::same_id(mini::kMiniWeightLoadPolicy.id, weight::kAttentionHeadQ8Policy.id) &&
                  weight::same_id(mini::kMiniWeightLoadPolicy.target, weight::kAttentionHeadQ8Policy.target),
                  "Inkling binds the shared policy");
    static_assert(weight::find_weight_load_policy("inkling2-mini-attention-head-bf16-q8_0@1",
                                                  std::span(&mini::kMiniWeightLoadPolicy, 1)) ==
                  &weight::kAttentionHeadQ8Policy, "old Inkling receipts resolve to the shared policy");
    constexpr std::array formats{weight::kBf16WeightStorage, weight::kQ8WeightStorage};
    for (const auto backend : {weight::Backend::hip, weight::Backend::cuda}) {
        const weight::WeightLoadScope scope{backend, "test-device", "test-compiler", "decode", digest(1), digest(2)};
        const std::array receipts{weight::WeightLoadDecision{policy.id, scope, digest(3), digest(4)}};
        weight::WeightLoadRequest request{profile::Name::standard, scope, formats, true, true};
        const auto resolve = [&] {
            return mini::resolve_mini_weight_load(request.tier, request.scope, request.consumer_formats,
                                                  request.enabled, request.allow_pending, receipts);
        };
        require(resolve().storage == weight::kBf16WeightStorage, "explicit qualification opt-in changed EXACT");
        request.tier = profile::Name::fast;
        request.allow_pending = false;
        require(mini::resolve_mini_weight_load(request.tier, scope, formats, true, false).storage ==
                    weight::kBf16WeightStorage, "FAST default before gate and speed receipts");
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
        require(mini::resolve_mini_weight_load(request.tier, scope, formats, true, false, no_speed).storage ==
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

// Every committed decision row names the current consumer arithmetic: bumping a resident Q8_0 route's revision
// retires the row, and this test then says so instead of FAST silently falling back to BF16.
// Low-bit qualification arms: one shared policy per mix, HIP consumers only, FAST or TURBO only, EXACT untouched.
void test_lowbit_arms() {
    weight::WeightLoadScope hip{weight::Backend::hip, "gfx1151", "hip-7.1", mini::kMiniResidentRegime, digest(1), digest(2)};
    auto cuda = hip;
    cuda.backend = weight::Backend::cuda;
    mini::MiniWeightOptions options;
    const auto arg = [&]() { return std::string_view("weights.attention_q4_g64.head_q6_g64@1"); };
    require(mini::parse_mini_weight_option(options, "--qualify-weights", arg, true) && !options.qualify_q8 &&
                options.qualify == &weight::kAttentionQ4HeadQ6Policy, "--qualify-weights selects the shared mix");
    for (const auto tier : {profile::Name::fast, profile::Name::turbo}) {
        const auto plan = mini::resolve_mini_resident_weight_load(tier, options, hip);
        require(plan.storage == weight::kQ4G64WeightStorage && plan.head_storage == weight::kQ6G64WeightStorage &&
                    plan.status == profile::Status::unqualified && plan.storage_for(weight::Role::head) == plan.head_storage,
                "q4 attention / q6 head arm");
        bool refused = false;
        try {
            static_cast<void>(mini::resolve_mini_resident_weight_load(tier, options, cuda));
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        require(refused, "a low-bit arm is refused without low-bit consumers");
    }
    const mini::MiniWeightOptions none;
    const auto exact = mini::resolve_mini_resident_weight_load(profile::Name::standard, none, hip);
    require(exact.storage == weight::kBf16WeightStorage && exact.head_storage == weight::kBf16WeightStorage,
            "EXACT stays BF16");
    auto exact_arm = options;
    exact_arm.tier.profile = profile::Name::standard;
    bool exact_refused = false;
    try {
        static_cast<void>(mini::resolve_mini_profile(exact_arm, true));
    } catch (const std::invalid_argument&) {
        exact_refused = true;
    }
    require(exact_refused, "a weight arm is refused under EXACT");
    const auto fast = mini::resolve_mini_resident_weight_load(profile::Name::fast, none, hip);
    require(fast.storage == weight::kBf16WeightStorage && fast.policy == mini::kMiniFastWeightPolicy.id,
            "no arm, no device scope: BF16 under the FAST mix binding");
    mini::MiniWeightOptions q8;
    const auto q8_key = [&]() { return std::string_view("inkling2-mini-attention-head-bf16-q8_0@1"); };
    require(mini::parse_mini_weight_option(q8, "--qualify-weights", q8_key, true) && q8.qualify_q8,
            "legacy Q8_0 key selects the Q8_0 arm");
    require(mini::resolve_mini_resident_weight_load(profile::Name::fast, q8, cuda).storage == weight::kQ8WeightStorage,
            "Q8_0 arm on CUDA");
    for (const char* key : {"weights.attention_head.q6_g64@1", "weights.attention_head.q4_g64@1"}) {
        mini::MiniWeightOptions arm;
        const auto value = [&]() { return std::string_view(key); };
        require(mini::parse_mini_weight_option(arm, "--qualify-weights", value, true), "arm key");
        const auto plan = mini::resolve_mini_resident_weight_load(profile::Name::turbo, arm, hip);
        require(plan.storage == arm.qualify->target && plan.head_storage == arm.qualify->target, "uniform low-bit arm");
    }
    for (const char* key : {"weights.head.q6_g64@1", "weights.head.q4_g64@1"}) {
        mini::MiniWeightOptions arm;
        const auto value = [&]() { return std::string_view(key); };
        require(mini::parse_mini_weight_option(arm, "--qualify-weights", value, true), "head-only arm key");
        const auto plan = mini::resolve_mini_resident_weight_load(profile::Name::fast, arm, hip);
        require(plan.storage == weight::kBf16WeightStorage && plan.head_storage == arm.qualify->head_target(),
                "head-only arm: BF16 attention, low-bit head");
        bool refused = false;
        try {
            static_cast<void>(mini::resolve_mini_resident_weight_load(profile::Name::fast, arm, cuda));
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        require(refused, "head-only low-bit arm refused without a low-bit consumer");
    }
    bool unknown = false;
    try {
        const auto bad = [&]() { return std::string_view("weights.head.q8_0@1"); };
        static_cast<void>(mini::parse_mini_weight_option(options, "--qualify-weights", bad, true));
    } catch (const std::invalid_argument&) {
        unknown = true;
    }
    require(unknown, "an unbound shared policy is refused");
}

// A TURBO row (stated quality, no gate v2 PASS): the low-bit mix for TURBO only, bound to the low-bit arithmetic.
void check_turbo_row(const weight::WeightLoadDecision& row) {
    constexpr std::array formats{weight::kBf16WeightStorage, weight::kQ8WeightStorage, weight::kQ4G64WeightStorage,
                                 weight::kQ6G64WeightStorage};
    const auto& turbo = mini::kMiniTurboWeightPolicy;
    require(row.scope.backend == weight::Backend::hip &&
                row.scope.build == weight::implementation_identity(strix::core::kResidentLowBitArithmeticHip),
            "TURBO row is stale: resident low-bit arithmetic revision changed, re-measure or remove the row");
    require(weight::policy_has_id(turbo, row.policy) && row.scope.regime == mini::kMiniResidentRegime &&
                row.gate_v2_pass == weight::Digest256{} && !(row.measured_gain == weight::Digest256{}),
            "TURBO row policy/regime/evidence");
    const auto plan = mini::resolve_mini_weight_load(profile::Name::turbo, row.scope, formats, true, false,
                                                     mini::kMiniWeightLoadDecisions, turbo);
    require(plan.status == profile::Status::set && plan.storage == turbo.target &&
                plan.head_storage == turbo.head_target() && plan.turbo_quality == row.turbo_quality,
            "TURBO default did not select its row or state its quality");
    require(mini::resolve_mini_weight_load(profile::Name::fast, row.scope, formats, true, false,
                                           mini::kMiniWeightLoadDecisions, turbo).storage == weight::kBf16WeightStorage,
            "FAST took a TURBO row");
    require(mini::resolve_mini_weight_load(profile::Name::fast, row.scope, formats, true, false).storage ==
                weight::kBf16WeightStorage, "FAST default changed by a TURBO row");
    require(mini::resolve_mini_weight_load(profile::Name::standard, row.scope, formats, true, false,
                                           mini::kMiniWeightLoadDecisions, turbo).storage == weight::kBf16WeightStorage,
            "EXACT changed by a TURBO row");
    auto other = row.scope;
    other.device_class = "gfx1100";
    require(mini::resolve_mini_weight_load(profile::Name::turbo, other, formats, true, false,
                                           mini::kMiniWeightLoadDecisions, turbo).storage == weight::kBf16WeightStorage,
            "TURBO row admitted on another device");
    other = row.scope;
    other.checkpoint.bytes[0] = static_cast<std::uint8_t>(other.checkpoint.bytes[0] ^ 1U);
    require(mini::resolve_mini_weight_load(profile::Name::turbo, other, formats, true, false,
                                           mini::kMiniWeightLoadDecisions, turbo).storage == weight::kBf16WeightStorage,
            "TURBO row admitted on another checkpoint");
}

// The FAST row (gate v2 PASS, measured gain): the calibrated q4 head for FAST and TURBO's FAST base, bound to the
// low-bit arithmetic; it applies only when its calibrated pack is present.
void check_fast_lowbit_row(const weight::WeightLoadDecision& row) {
    constexpr std::array formats{weight::kBf16WeightStorage, weight::kQ8WeightStorage, weight::kQ4G64WeightStorage,
                                 weight::kQ6G64WeightStorage};
    const auto& fast = mini::kMiniFastWeightPolicy;
    require(row.scope.backend == weight::Backend::hip &&
                row.scope.build == weight::implementation_identity(strix::core::kResidentLowBitArithmeticHip),
            "FAST low-bit row is stale: resident low-bit arithmetic revision changed, re-qualify or remove the row");
    require(weight::policy_has_id(fast, row.policy) && row.scope.regime == mini::kMiniResidentRegime &&
                !(row.gate_v2_pass == weight::Digest256{}) && !(row.measured_gain == weight::Digest256{}) &&
                row.turbo_quality.empty(),
            "FAST low-bit row policy/regime/evidence");
    for (const auto tier : {profile::Name::fast, profile::Name::turbo}) {
        const auto plan = mini::resolve_mini_weight_load(tier, row.scope, formats, true, false,
                                                         mini::kMiniWeightLoadDecisions, fast);
        require(plan.status == profile::Status::set && plan.storage == weight::kBf16WeightStorage &&
                    plan.head_storage == weight::kQ4G64WeightStorage && plan.gate_v2_pass == row.gate_v2_pass &&
                    mini::mini_row_pack(plan) == &mini::kMiniLowBitPackIdentity,
                "FAST low-bit row not selected (or its pack not required)");
    }
    require(mini::resolve_mini_weight_load(profile::Name::standard, row.scope, formats, true, false,
                                           mini::kMiniWeightLoadDecisions, fast).head_storage == weight::kBf16WeightStorage,
            "EXACT changed by the FAST low-bit row");
    auto other = row.scope;
    other.device_class = "gfx1100";
    require(mini::resolve_mini_weight_load(profile::Name::fast, other, formats, true, false,
                                           mini::kMiniWeightLoadDecisions, fast).head_storage == weight::kBf16WeightStorage,
            "FAST low-bit row admitted on another device");
    // No calibrated pack next to the checkpoint (and STRIX_LOWBIT_PACK unset): BF16, and the plan says why.
    if (const char* env = std::getenv("STRIX_LOWBIT_PACK"); env == nullptr || *env == '\0') {
        const mini::MiniWeightOptions none;
        const auto absent = mini::resolve_mini_resident_weight_load(profile::Name::fast, none, row.scope,
                                                                    "/nonexistent-strix-dir/model.safetensors");
        require(absent.storage == weight::kBf16WeightStorage && absent.head_storage == weight::kBf16WeightStorage &&
                    absent.status == profile::Status::pending &&
                    absent.reason.find("calibrated low-bit pack absent") != std::string_view::npos &&
                    mini::mini_row_pack(absent) == nullptr,
                "FAST low-bit row applied without its pack");
    }
}

void test_committed_rows() {
    constexpr std::array formats{weight::kBf16WeightStorage, weight::kQ8WeightStorage};
    for (const auto& row : mini::kMiniWeightLoadDecisions) {
        if (weight::policy_has_id(mini::kMiniFastWeightPolicy, row.policy)) {
            check_fast_lowbit_row(row);
            continue;
        }
        if (!row.turbo_quality.empty()) {
            check_turbo_row(row);
            continue;
        }
        const auto& consumers = row.scope.backend == weight::Backend::hip ?
            std::span<const weight::ArithmeticId>(strix::core::kResidentQ8ArithmeticHip) :
            std::span<const weight::ArithmeticId>(strix::core::kResidentQ8ArithmeticCuda);
        require(row.scope.build == weight::implementation_identity(consumers),
                "decision row is stale: resident Q8_0 arithmetic revision changed, re-qualify or remove the row");
        require(row.policy.name == mini::kMiniWeightLoadPolicy.id.name &&
                    row.policy.revision == mini::kMiniWeightLoadPolicy.id.revision &&
                    row.scope.regime == mini::kMiniResidentRegime, "decision row policy/regime");
        auto plan = mini::resolve_mini_weight_load(profile::Name::fast, row.scope, formats, true, false);
        require(plan.status == profile::Status::set && plan.storage == weight::kQ8WeightStorage &&
                    plan.gate_v2_pass == row.gate_v2_pass, "FAST default did not select the qualified Q8_0 row");
        require(mini::resolve_mini_weight_load(profile::Name::standard, row.scope, formats, true, false).storage ==
                    weight::kBf16WeightStorage, "EXACT changed by a FAST row");
        require(mini::resolve_mini_weight_load(profile::Name::fast, row.scope, formats, false, false).storage ==
                    weight::kBf16WeightStorage, "--no-weight-q8 ignored");
        auto other = row.scope;
        other.device_class = "gfx1100";
        plan = mini::resolve_mini_weight_load(profile::Name::fast, other, formats, true, false);
        require(plan.status == profile::Status::pending && plan.storage == weight::kBf16WeightStorage,
                "row admitted on another device");
        other = row.scope;
        other.checkpoint.bytes[0] = static_cast<std::uint8_t>(other.checkpoint.bytes[0] ^ 1U);
        require(mini::resolve_mini_weight_load(profile::Name::fast, other, formats, true, false).storage ==
                    weight::kBf16WeightStorage, "row admitted on another checkpoint");
        other = row.scope;
        other.checkpoint = {};
        require(mini::resolve_mini_weight_load(profile::Name::fast, other, formats, true, false).storage ==
                    weight::kBf16WeightStorage, "empty scope admitted");
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
    for (const auto format : {weight::kBf16WeightStorage, weight::kQ8WeightStorage, weight::kQ4G64WeightStorage,
                              weight::kQ6G64WeightStorage}) {
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
        if (format == weight::kQ4G64WeightStorage || format == weight::kQ6G64WeightStorage) {
            // 128-element superblocks: 68 B (q4_g64) / 100 B (q6_g64); every tagged row length is a multiple of 128.
            const std::uint64_t block = format == weight::kQ4G64WeightStorage ? 68U : 100U;
            require(dense == 905969664ULL / 128U * block, "per-token low-bit Q/K/V/O bytes");
            require(head == 460933632ULL / 128U * block && head_storage >= head, "per-token low-bit LM head bytes");
            continue;
        }
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
        test_committed_rows();
        test_lowbit_arms();
        test_token_bytes();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
