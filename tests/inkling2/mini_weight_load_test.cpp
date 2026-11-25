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
#include <filesystem>
#include <memory>
#include <string>
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
        // One image serves prefill and decode: a full decode row selects it once prefill has its gate row.
        auto prefill = scope;
        prefill.regime = weight::kPrefillRegime;
        const weight::WeightLoadEvidence one_lane{"fixture", "2026-11-23", 800, 65536, 1};
        const std::array receipts{weight::WeightLoadDecision{policy.id, scope, digest(3), digest(4), one_lane},
                                  weight::WeightLoadDecision{policy.id, prefill, digest(6), {}}};
        weight::WeightLoadRequest request{profile::Name::standard, scope, formats, true, true, {}, {8192U, 1U}};
        const auto resolve = [&] {
            return mini::resolve_mini_weight_load(request.tier, request.scope, request.consumer_formats,
                                                  request.enabled, request.allow_pending, receipts, request.extent);
        };
        require(resolve().storage == weight::kBf16WeightStorage, "explicit qualification opt-in changed EXACT");
        request.tier = profile::Name::fast;
        request.allow_pending = false;
        require(mini::resolve_mini_weight_load(request.tier, scope, formats, true, false).storage ==
                    weight::kBf16WeightStorage, "FAST default before gate and speed receipts");
        require(resolve().status == profile::Status::set && resolve().storage == weight::kQ8WeightStorage &&
                    resolve().gate_v2_pass == digest(3) && resolve().measured_gain == digest(4), "qualified FAST selection");
        request.extent = {65536U, 1U};
        require(resolve().storage == weight::kQ8WeightStorage, "one lane at the band edge");
        for (const weight::WeightLoadExtent extent : {weight::WeightLoadExtent{8192U, 8U}, {8192U, 0U}, {65537U, 1U}}) {
            request.extent = extent;
            require(resolve().storage == weight::kBf16WeightStorage, "outside the measured gain bands");
        }
        request.extent = {8192U, 1U};
        request.tier = profile::Name::turbo;
        require(resolve().storage == weight::kQ8WeightStorage, "TURBO omitted FAST storage");
        request.tier = profile::Name::fast;
        request.enabled = false;
        require(resolve().status == profile::Status::off && resolve().storage == weight::kBf16WeightStorage, "off switch");
        request.enabled = true;
        auto no_speed = receipts;
        no_speed.front().measured_gain = {};
        require(mini::resolve_mini_weight_load(request.tier, scope, formats, true, false, no_speed, request.extent).storage ==
                    weight::kBf16WeightStorage, "missing measured gain");
        require(mini::resolve_mini_weight_load(request.tier, scope, formats, true, false, std::span(receipts).first(1),
                                              request.extent).storage == weight::kBf16WeightStorage, "a regime without a gate row needs a second image");
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
// Low-bit qualification arms: one shared policy per mix, HIP consumers only, FAST or TURBO only, EXACT untouched.
void test_lowbit_arms() {
    weight::WeightLoadScope hip{weight::Backend::hip, "gfx1151", "hip-7.1", mini::kMiniWeightRegime, digest(1), digest(2)};
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
    require(fast.storage == weight::kBf16WeightStorage && fast.policy == mini::kMiniWeightLoadPolicy.id,
            "no arm: the incumbent FAST weight policy");
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
    const auto head_q4 = [] { return std::string_view("weights.head.q4_g64@1"); };
    require(mini::parse_mini_weight_option(options, "--qualify-decode-weights", head_q4, true) &&
                options.qualify_decode_only, "explicit decode-only weight arm");
}

void test_lowbit_receipts() {
    const auto& turbo = mini::kMiniLowBitDecisions.front();
    const auto extent = mini::kMiniLowBitMeasuredExtent;
    const auto pick = [&](profile::Name tier, weight::WeightLoadScope scope, weight::WeightLoadExtent shape, unsigned rows,
                          std::span<const weight::WeightLoadDecision> decisions = mini::kMiniLowBitDecisions) {
        return mini::mini_lowbit_receipt_plan(tier, scope, shape, rows, mini::kMiniLowBitPackIdentity, decisions);
    };
    require(pick(profile::Name::turbo, turbo.scope, extent, 8U).head_storage == weight::kQ4G64WeightStorage,
            "TURBO evidence covers decode at the measured capacity and width");
    require(pick(profile::Name::fast, turbo.scope, extent, 8U).head_storage == weight::kBf16WeightStorage,
            "current evidence has no qualifying FAST row");
    require(mini::resolve_mini_resident_weight_load(profile::Name::turbo, {}, turbo.scope, extent).head_storage ==
                weight::kBf16WeightStorage, "incumbent TURBO stays on the plain head");

    // A synthetic passing receipt exercises FAST admission without publishing a gate result.
    auto fast = turbo;
    fast.policy = mini::kMiniFastWeightPolicy.id;
    fast.gate_v2_pass = digest(3);
    fast.turbo_quality = {};
    const auto fixture = std::span(&fast, 1U);
    const auto expect = [&](weight::WeightLoadExtent shape, unsigned rows, weight::FormatId expected) {
        const auto plan = pick(profile::Name::fast, fast.scope, shape, rows, fixture);
        require(plan.storage == weight::kBf16WeightStorage && plan.head_storage == expected,
                "FAST evidence uses Q4 only for a measured decode regime");
    };
    expect(extent, 1U, weight::kQ4G64WeightStorage);
    expect(extent, 8U, weight::kQ4G64WeightStorage);
    expect(extent, 16U, weight::kQ4G64WeightStorage);
    expect(extent, 2U, weight::kBf16WeightStorage);
    expect(extent, 17U, weight::kBf16WeightStorage);
    expect(extent, 0U, weight::kBf16WeightStorage);
    expect({8192U, 16U}, 8U, weight::kBf16WeightStorage);
    expect({32768U, 16U}, 8U, weight::kBf16WeightStorage);
    expect({0U, 16U}, 8U, weight::kBf16WeightStorage);
    expect({16384U, 0U}, 8U, weight::kBf16WeightStorage);
    auto prefill = fast.scope;
    prefill.regime = weight::kPrefillRegime;
    require(pick(profile::Name::fast, prefill, extent, 8U, fixture).head_storage == weight::kBf16WeightStorage,
            "decode receipt does not cover prefill");
    require(pick(profile::Name::standard, fast.scope, extent, 8U, fixture).head_storage == weight::kBf16WeightStorage,
            "EXACT ignores the FAST receipt");
    require(pick(profile::Name::fast, fast.scope, extent, 8U, {}).head_storage == weight::kBf16WeightStorage,
            "missing receipt keeps the incumbent head");
    require(mini::mini_lowbit_receipt_plan(profile::Name::fast, fast.scope, extent, 8U, {}, fixture).head_storage ==
                weight::kBf16WeightStorage, "missing or mismatched pack identity keeps the incumbent head");
}

void test_pack_admission() {
    const auto path = std::filesystem::temp_directory_path() / "strix-inkling-mini-weight-pack-test";
    const auto checkpoint = digest(7);
    require(static_cast<bool>(weight::LowbitPackWriter(checkpoint).write(path.string())), "write mock pack");
    std::shared_ptr<const weight::LowbitPack> pack;
    require(mini::admit_mini_lowbit_pack(path.string(), checkpoint, pack).empty() && pack != nullptr,
            "open checkpoint-bound pack");
    const auto identity = pack->identity();
    require(mini::admit_mini_lowbit_pack(path.string(), checkpoint, pack, &identity).empty() && pack != nullptr,
            "open measured pack");
    require(!mini::admit_mini_lowbit_pack(path.string(), digest(8), pack).empty() && pack == nullptr,
            "refuse another checkpoint");
    const auto wrong_identity = digest(9);
    require(!mini::admit_mini_lowbit_pack(path.string(), checkpoint, pack, &wrong_identity).empty() && pack == nullptr,
            "refuse changed pack content");
    std::filesystem::remove(path);
    require(!mini::admit_mini_lowbit_pack(path.string(), checkpoint, pack).empty() && pack == nullptr,
            "refuse missing pack");
    mini::MiniWeightOptions arm;
    arm.qualify = &weight::kHeadQ4Policy;
    const auto& scope = mini::kMiniLowBitDecisions.front().scope;
    const auto fallback = mini::mini_plain_weight_fallback(profile::Name::fast, arm, scope, {}, "sidecar declined");
    require(fallback.policy == mini::kMiniWeightLoadPolicy.id && fallback.head_storage == weight::kBf16WeightStorage &&
                fallback.reason == "sidecar declined", "fallback resolves incumbent weights");
}

// Changing Q8_0 consumer arithmetic retires its receipt.
void test_committed_rows() {
    constexpr std::array formats{weight::kBf16WeightStorage, weight::kQ8WeightStorage};
    const auto& own = mini::kMiniWeightLoadDecisions[0].scope;
    for (const auto tier : {profile::Name::fast, profile::Name::turbo}) {
        require(mini::resolve_mini_weight_load(tier, own, formats, true, false,
                    mini::kMiniWeightLoadDecisions, {8192U, 1U}).storage == weight::kBf16WeightStorage,
                "an unqualified decode arm must preserve the incumbent BF16 default");
    }
    for (const auto& row : mini::kMiniWeightLoadDecisions) {
        const auto& consumers = row.scope.backend == weight::Backend::hip ?
            std::span<const weight::ArithmeticId>(strix::core::kResidentQ8ArithmeticHip) :
            std::span<const weight::ArithmeticId>(strix::core::kResidentQ8ArithmeticCuda);
        require(row.scope.build == weight::implementation_identity(consumers),
                "decision row is stale: resident Q8_0 arithmetic revision changed, re-qualify or remove the row");
        require(row.policy.name == mini::kMiniWeightLoadPolicy.id.name &&
                    row.policy.revision == mini::kMiniWeightLoadPolicy.id.revision &&
                    std::ranges::find(mini::kMiniWeightLoadRegimes, row.scope.regime) != mini::kMiniWeightLoadRegimes.end(),
                "decision row policy/regime");
        require(!(row.gate_v2_pass == weight::Digest256{}) && !row.evidence.receipt.empty() &&
                    row.evidence.receipt.starts_with("docs/") && row.evidence.date.size() == 10U && row.evidence.rows > 0U,
                "decision row without its gate receipt id, date and rows");
        require(row.scope.backend != weight::Backend::hip || row.scope.compiler_id.starts_with("hip-clang-"),
                "HIP row toolkit must be the device-compiler id (weight::hip_clang_compiler_id)");
        const weight::WeightLoadExtent extent{row.evidence.max_context ? row.evidence.max_context : 8192U,
                                              row.evidence.max_lanes ? row.evidence.max_lanes : 1U};
        const auto context = extent.context;
        auto plan = mini::resolve_mini_weight_load(profile::Name::fast, row.scope, formats, true, false,
                                                   mini::kMiniWeightLoadDecisions, extent);
        require(plan.regime_count == mini::kMiniWeightLoadRegimes.size(), "every regime reported");
        for (const auto& entry : plan.regime_rows())
            if (entry.regime == row.scope.regime)
                require(entry.row == &row, "the committed row is the one its regime reports");
        require(weight::weight_load_regime_receipt(plan).find(row.evidence.receipt) != std::string::npos,
                "receipt line names the row's receipt");
        require(mini::resolve_mini_weight_load(profile::Name::standard, row.scope, formats, true, false,
                    mini::kMiniWeightLoadDecisions, extent).storage == weight::kBf16WeightStorage,
                "EXACT changed by a FAST row");
        require(mini::resolve_mini_weight_load(profile::Name::fast, row.scope, formats, false, false,
                    mini::kMiniWeightLoadDecisions, extent).status == profile::Status::off, "--no-weight-q8 ignored");
        if (row.evidence.max_context) {
            plan = mini::resolve_mini_weight_load(profile::Name::fast, row.scope, formats, true, false,
                                                  mini::kMiniWeightLoadDecisions, {row.evidence.max_context + 1U, 1U});
            for (const auto& entry : plan.regime_rows())
                if (entry.regime == row.scope.regime)
                    require(!entry.row && entry.outside_band == &row, "a context outside the band is reported");
        }
        auto other = row.scope;
        other.device_class = "gfx1100";
        plan = mini::resolve_mini_weight_load(profile::Name::fast, other, formats, true, false,
                                              mini::kMiniWeightLoadDecisions, extent);
        require(plan.storage == weight::kBf16WeightStorage && plan.regimes[0].row == nullptr &&
                    plan.regimes[1].row == nullptr, "row admitted on another device");
        other = row.scope;
        other.checkpoint.bytes[0] = static_cast<std::uint8_t>(other.checkpoint.bytes[0] ^ 1U);
        require(weight::find_weight_load_gate(mini::kMiniWeightLoadPolicy, other, mini::kMiniWeightLoadDecisions,
                                              {context, 1U}) == nullptr, "row admitted on another checkpoint");
        other = row.scope;
        other.checkpoint = {};
        require(weight::find_weight_load_gate(mini::kMiniWeightLoadPolicy, other, mini::kMiniWeightLoadDecisions,
                                              {context, 1U}) == nullptr, "empty scope admitted");
    }
    require(weight::registered_regime_rows(mini::kMiniWeightLoadPolicy, weight::Backend::hip, mini::kMiniWeightLoadRegimes,
                                           mini::kMiniWeightLoadDecisions) == "prefill=gate-only@gfx1151,decode=none",
            "only the prefill gate row belongs in the registered table");
    const weight::WeightLoadExtent one_lane{8192U, 1U};
    mini::MiniWeightOptions options;
    const auto fast = [&] {
        return mini::resolve_mini_resident_weight_load(profile::Name::fast, options, own, one_lane);
    };
    const auto plan = fast();
    require(plan.status == profile::Status::pending &&
                weight::weight_load_regime_receipt(plan) ==
                    "strix-weight-load-regimes/v1 weights.attention_head.q8_0@1 plain.bf16@1 pending "
                    "prefill=gate-only:docs/decisions/evidence-register.md#inkling-2-mini-q8_0-attention-and-head-on-gfx1151 "
                    "decode=none",
            "the resident FAST plan must report the admitted prefill row and missing decode qualification");
    options.qualify_q8 = true;
    const auto qualification = fast();
    require(qualification.storage == weight::kQ8WeightStorage && qualification.status == profile::Status::unqualified,
            "the candidate must remain available under explicit qualification");
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
        test_lowbit_receipts();
        test_pack_admission();
        test_token_bytes();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
