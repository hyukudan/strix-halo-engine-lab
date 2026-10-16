#include "strix/offline_tuning_cache.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace {
using namespace strix::kernels;
int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
template <class Function> void rejects(Function&& function, const char* message) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, message);
}
bool same(const OfflineTuningRecord& a, const OfflineTuningRecord& b) {
    return a.key == b.key && a.kernel_id == b.kernel_id &&
        a.qualification_receipt_sha256 == b.qualification_receipt_sha256 &&
        std::bit_cast<std::uint64_t>(a.median_complete_seconds) == std::bit_cast<std::uint64_t>(b.median_complete_seconds) &&
        a.samples == b.samples && a.complete_workload == b.complete_workload &&
        a.owners_and_resources_qualified == b.owners_and_resources_qualified;
}
OfflineTuningRecord fixture() {
    OfflineTuningRecord r;
    auto& k = r.key;
    k.gfx_target = "gfx1151"; k.device_sku = "device-1";
    k.compiler_identity = std::string("compiler\0literal", 16); k.hip_identity = "runtime-identity";
    k.binary_sha256 = std::string(64, 'a'); k.model_content_sha256 = std::string(64, 'b');
    k.compiler_flags_sha256 = std::string(64, 'c');
    k.input_precision = "bf16"; k.accumulator_precision = "fp32"; k.output_precision = "bf16";
    k.weight_format = "Q8_0"; k.input_layout = "row-major"; k.output_layout = "row-major";
    k.state_layout = "state-physical-v1"; k.context_bucket = "512"; k.memory_regime = "registered";
    k.compute_units = 40; k.wave = WaveSize::wave32; k.family = OperatorFamily::recurrent;
    k.phase = ExecutionPhase::decode; k.shape = {1, 2, 2560, 6144};
    k.m = 1; k.n = 2560; k.k = 6144; k.numerical_contract_identity = 0xfedcba9876543210ULL;
    r.kernel_id = "fixture.kernel"; r.qualification_receipt_sha256 = std::string(64, 'd');
    r.median_complete_seconds = 0.000125; r.samples = 17;
    r.complete_workload = true; r.owners_and_resources_qualified = true;
    return r;
}
std::uint32_t u32(const std::vector<std::byte>& b, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) value |= std::to_integer<std::uint32_t>(b.at(offset + i)) << (8U * i);
    return value;
}
void put(std::vector<std::byte>& b, std::size_t offset, std::uint64_t value, unsigned width) {
    for (unsigned i = 0; i < width; ++i) b.at(offset + i) = static_cast<std::byte>((value >> (i * 8U)) & 255U);
}
std::size_t skip_string(const std::vector<std::byte>& b, std::size_t offset) { return offset + 4 + u32(b, offset); }
} // namespace

int main() {
    const auto original = fixture();
    const std::array records{original};
    const auto bytes = encode_offline_tuning_cache(records);
    const auto decoded = decode_offline_tuning_cache(bytes);
    expect(decoded.size() == 1 && same(decoded.front(), original), "all fields and provenance survive exact binary roundtrip");
    expect(encode_offline_tuning_cache(decoded) == bytes, "roundtrip retains canonical serialized bytes");
    expect(bytes[0] == std::byte{'S'} && bytes[7] == std::byte{0} && u32(bytes, 8) == 1 && u32(bytes, 12) == 1 &&
        u32(bytes, 16) == bytes.size() - 20, "wire header and framed record use explicit little endian version/count/length");
    const auto empty = encode_offline_tuning_cache({});
    expect(empty.size() == 16 && decode_offline_tuning_cache(empty).empty(), "empty cache has a complete versioned header");
    OfflineTuningRecord unmeasured;
    const std::array unknown_records{unmeasured};
    const auto unknown_bytes = encode_offline_tuning_cache(unknown_records);
    expect(unknown_bytes.size() == 170 && same(decode_offline_tuning_cache(unknown_bytes).front(), unmeasured),
        "codec preserves unmeasured and incomplete records without inventing provenance");
    for (const double cost : {0.0, -0.0, std::numeric_limits<double>::denorm_min(), 1.25}) {
        auto r = original; r.median_complete_seconds = cost;
        const std::array arm{r};
        expect(same(decode_offline_tuning_cache(encode_offline_tuning_cache(arm)).front(), r),
            "finite timing preserves exact binary64 bits including negative zero and subnormals");
    }
    for (unsigned family = 0; family <= static_cast<unsigned>(OperatorFamily::lm_head); ++family)
        for (unsigned phase = 0; phase <= static_cast<unsigned>(ExecutionPhase::rollback); ++phase) {
            auto r = original; r.key.family = static_cast<OperatorFamily>(family); r.key.phase = static_cast<ExecutionPhase>(phase);
            r.key.wave = phase % 2 ? WaveSize::wave32 : WaveSize::wave64;
            r.complete_workload = phase % 2 != 0; r.owners_and_resources_qualified = family % 2 != 0;
            const std::array arm{r};
            expect(same(decode_offline_tuning_cache(encode_offline_tuning_cache(arm)).front(), r),
                "every known enum and independent boolean combination roundtrips");
        }
    for (std::size_t cut = 0; cut < bytes.size(); ++cut)
        rejects([&] { static_cast<void>(decode_offline_tuning_cache(std::span(bytes).first(cut))); },
            "every truncation rejects the entire cache");
    auto malformed = bytes; malformed.push_back(std::byte{0});
    rejects([&] { static_cast<void>(decode_offline_tuning_cache(malformed)); }, "top-level trailing byte rejects");
    malformed = bytes; put(malformed, 16, u32(bytes, 16) + 1, 4); malformed.push_back(std::byte{0});
    rejects([&] { static_cast<void>(decode_offline_tuning_cache(malformed)); }, "record trailing byte rejects");
    for (const auto offset : {std::size_t{0}, std::size_t{8}, std::size_t{12}, std::size_t{16}, std::size_t{20}}) {
        malformed = bytes; put(malformed, offset, 0xffffffffU, 4);
        rejects([&] { static_cast<void>(decode_offline_tuning_cache(malformed)); },
            "bad magic/version/count/frame/string prefixes reject without allocating declared payload");
    }
    std::size_t scalars = 20;
    for (unsigned i = 0; i < 16; ++i) scalars = skip_string(bytes, scalars);
    for (const auto delta : {4U, 8U, 12U}) {
        malformed = bytes; put(malformed, scalars + delta, 0x7fffffffU, 4);
        rejects([&] { static_cast<void>(decode_offline_tuning_cache(malformed)); }, "unknown wave/family/phase tags reject");
    }
    auto cost_offset = skip_string(bytes, skip_string(bytes, scalars + 64));
    for (const auto bits : {0x7ff0000000000000ULL, 0xfff0000000000000ULL, 0x7ff8000000000042ULL, 0xbff0000000000000ULL}) {
        malformed = bytes; put(malformed, cost_offset, bits, 8);
        rejects([&] { static_cast<void>(decode_offline_tuning_cache(malformed)); }, "nonfinite and negative serialized costs reject");
        auto r = original; r.median_complete_seconds = std::bit_cast<double>(static_cast<std::uint64_t>(bits));
        const std::array arm{r};
        rejects([&] { static_cast<void>(encode_offline_tuning_cache(arm)); }, "serializer also rejects invalid costs");
    }
    for (const auto offset : {cost_offset + 12, cost_offset + 13}) {
        malformed = bytes; malformed[offset] = std::byte{2};
        rejects([&] { static_cast<void>(decode_offline_tuning_cache(malformed)); }, "noncanonical boolean rejects");
    }
    for (unsigned fault = 0; fault < 4; ++fault) {
        OfflineTuningCacheLimits limits;
        if (fault == 0) limits.max_bytes = bytes.size() - 1;
        if (fault == 1) limits.max_records = 0;
        if (fault == 2) limits.max_string_bytes = 63;
        if (fault == 3) limits.max_total_string_bytes = 1;
        rejects([&] { static_cast<void>(decode_offline_tuning_cache(bytes, limits)); }, "parser enforces each resource limit");
        rejects([&] { static_cast<void>(encode_offline_tuning_cache(records, limits)); }, "serializer enforces each resource limit before output allocation");
    }
    OfflineTuningCacheLimits exact; exact.max_bytes = bytes.size(); exact.max_records = 1; exact.max_string_bytes = 64;
    exact.max_total_string_bytes = bytes.size() - 170;
    expect(encode_offline_tuning_cache(records, exact) == bytes && same(decode_offline_tuning_cache(bytes, exact).front(), original),
        "exact byte/count/string limits are inclusive");
    auto short_total = exact; --short_total.max_total_string_bytes;
    rejects([&] { static_cast<void>(decode_offline_tuning_cache(bytes, short_total)); }, "aggregate string limit rejects one byte below the exact boundary");
    OfflineTuningCacheLimits no_header; no_header.max_bytes = 15;
    rejects([&] { static_cast<void>(encode_offline_tuning_cache({}, no_header)); }, "even empty serialization obeys the header byte budget");
    for (unsigned field = 0; field < 3; ++field) {
        auto r = original;
        if (field == 0) r.key.wave = static_cast<WaveSize>(0);
        if (field == 1) r.key.family = static_cast<OperatorFamily>(-1);
        if (field == 2) r.key.phase = static_cast<ExecutionPhase>(99);
        const std::array arm{r};
        rejects([&] { static_cast<void>(encode_offline_tuning_cache(arm)); }, "serializer refuses unknown enum values");
    }

    // Persistence cannot authorize a new kernel or reconcile conflicting
    // timings. The existing registry retains both decisions after a roundtrip.
    KernelVariant baseline;
    baseline.id = original.kernel_id; baseline.family = original.key.family; baseline.phase = original.key.phase;
    baseline.shape = original.key.shape; baseline.wave = original.key.wave; baseline.resources.workgroup_threads = 256;
    baseline.numerics = {original.key.input_precision, original.key.accumulator_precision, original.key.output_precision, true, true};
    baseline.input_layout = original.key.input_layout; baseline.output_layout = original.key.output_layout;
    baseline.compiler_identity = original.key.compiler_identity; baseline.numerical_contract_identity = original.key.numerical_contract_identity;
    baseline.qualification = QualificationStatus::target_qualified; baseline.measured_seconds = 0.1;
    KernelRegistry registry; registry.add(baseline);
    KernelQuery query; query.family = baseline.family; query.phase = baseline.phase; query.shape = baseline.shape;
    query.minimum_qualification = QualificationStatus::target_qualified;
    expect(registry.select_tuned(query, original.key, decoded).reason == TuningSelectionReason::qualified_record,
        "qualified exact record remains subject to the existing selector");
    auto faster_duplicate = original; faster_duplicate.median_complete_seconds /= 10;
    const std::array duplicates{original, faster_duplicate};
    const auto loaded_duplicates = decode_offline_tuning_cache(encode_offline_tuning_cache(duplicates));
    expect(loaded_duplicates.size() == 2 && registry.select_tuned(query, original.key, loaded_duplicates).reason == TuningSelectionReason::ambiguous_records,
        "duplicates stay ambiguous rather than selecting their fastest measurement");
    baseline.id = "unqualified"; baseline.qualification = QualificationStatus::experimental;
    KernelRegistry unqualified; unqualified.add(baseline);
    auto alleged = original; alleged.kernel_id = baseline.id;
    const std::array alleged_records{alleged};
    expect(unqualified.select_tuned(query, original.key,
        decode_offline_tuning_cache(encode_offline_tuning_cache(alleged_records))).reason == TuningSelectionReason::no_qualified_record,
        "loaded complete records cannot promote an experimental registry variant");
    if (!failures) std::cout << "offline tuning cache tests passed\n";
    return failures ? 1 : 0;
}
