#include "strix/offline_tuning_cache.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace strix::kernels {
namespace {
constexpr std::array<std::byte, 8> magic{std::byte{'S'}, std::byte{'T'}, std::byte{'R'}, std::byte{'X'},
    std::byte{'T'}, std::byte{'U'}, std::byte{'N'}, std::byte{0}};
constexpr std::uint32_t version = 1;
constexpr std::uint64_t header_bytes = 16;
constexpr std::uint64_t fixed_record_bytes = 150; // Includes eighteen string lengths.
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);

[[noreturn]] void invalid(const char* reason) { throw std::invalid_argument(reason); }

std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        invalid("offline tuning cache size overflows");
    return a + b;
}

void valid_tags(std::uint32_t wave, std::uint32_t family, std::uint32_t phase) {
    if (wave != 32 && wave != 64) invalid("offline tuning cache has unknown wave tag");
    if (family > static_cast<std::uint32_t>(OperatorFamily::lm_head))
        invalid("offline tuning cache has unknown family tag");
    if (phase > static_cast<std::uint32_t>(ExecutionPhase::rollback))
        invalid("offline tuning cache has unknown phase tag");
    switch (static_cast<OperatorFamily>(family)) {
    case OperatorFamily::rms_norm: case OperatorFamily::residual_add: case OperatorFamily::partial_rope:
    case OperatorFamily::argmax: case OperatorFamily::top_k: case OperatorFamily::packed_projection:
    case OperatorFamily::recurrent: case OperatorFamily::moe_route: case OperatorFamily::moe_expert:
    case OperatorFamily::sparse_lookup: case OperatorFamily::embedding_lookup: case OperatorFamily::indexed_attention:
    case OperatorFamily::hyper_mix: case OperatorFamily::hyper_expand: case OperatorFamily::elementwise_inject:
    case OperatorFamily::shared_expert: case OperatorFamily::moe_combine: case OperatorFamily::lm_head: break;
    default: invalid("offline tuning cache has unknown family tag");
    }
    switch (static_cast<ExecutionPhase>(phase)) {
    case ExecutionPhase::prefill: case ExecutionPhase::decode: case ExecutionPhase::speculative_draft:
    case ExecutionPhase::speculative_verify: case ExecutionPhase::commit: case ExecutionPhase::rollback: break;
    default: invalid("offline tuning cache has unknown phase tag");
    }
}

void valid_cost(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0)
        invalid("offline tuning cache has nonfinite or negative cost");
}

template <class Record, class Visitor>
void strings(Record& r, Visitor&& visit) {
    auto& k = r.key;
    visit(k.gfx_target); visit(k.device_sku); visit(k.compiler_identity); visit(k.hip_identity);
    visit(k.binary_sha256); visit(k.model_content_sha256); visit(k.compiler_flags_sha256);
    visit(k.input_precision); visit(k.accumulator_precision); visit(k.output_precision);
    visit(k.weight_format); visit(k.input_layout); visit(k.output_layout); visit(k.state_layout);
    visit(k.context_bucket); visit(k.memory_regime);
}

struct StringBudget {
    OfflineTuningCacheLimits limits;
    std::uint64_t total = 0;
    void consume(std::uint64_t bytes) {
        if (bytes > limits.max_string_bytes || bytes > std::numeric_limits<std::uint32_t>::max())
            invalid("offline tuning cache string exceeds limit");
        total = add(total, bytes);
        if (total > limits.max_total_string_bytes)
            invalid("offline tuning cache total strings exceed limit");
    }
};

struct Reader {
    std::span<const std::byte> bytes;
    std::size_t offset = 0;
    std::span<const std::byte> take(std::uint64_t count) {
        if (count > bytes.size() - offset) invalid("offline tuning cache is truncated");
        const auto result = bytes.subspan(offset, static_cast<std::size_t>(count));
        offset += static_cast<std::size_t>(count);
        return result;
    }
    std::uint64_t uint(unsigned width) {
        std::uint64_t value = 0;
        const auto data = take(width);
        for (unsigned i = 0; i < width; ++i)
            value |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(data[i])) << (i * 8U);
        return value;
    }
    bool boolean() {
        const auto value = uint(1);
        if (value > 1) invalid("offline tuning cache has noncanonical boolean");
        return value != 0;
    }
    void string(std::string* output, StringBudget& budget) {
        const auto count = uint(4);
        budget.consume(count);
        const auto data = take(count);
        if (output) output->assign(reinterpret_cast<const char*>(data.data()), data.size());
    }
    void finished() const {
        if (offset != bytes.size()) invalid("offline tuning cache has trailing bytes");
    }
};

void read_record(Reader& input, StringBudget& budget, OfflineTuningRecord* output) {
    // The first pass constructs no strings or record vector, even for a huge
    // forged count/length prefix. The second pass runs the identical grammar.
    if (output) strings(*output, [&](std::string& value) { input.string(&value, budget); });
    else for (unsigned i = 0; i < 16; ++i) input.string(nullptr, budget);
    const auto compute_units = static_cast<std::uint32_t>(input.uint(4));
    const auto wave = static_cast<std::uint32_t>(input.uint(4));
    const auto family = static_cast<std::uint32_t>(input.uint(4));
    const auto phase = static_cast<std::uint32_t>(input.uint(4));
    valid_tags(wave, family, phase);
    ShapeClass shape;
    shape.logical_rows = static_cast<std::uint32_t>(input.uint(4));
    shape.request_concurrency = static_cast<std::uint32_t>(input.uint(4));
    shape.hidden_size = static_cast<std::uint32_t>(input.uint(4));
    shape.auxiliary_size = static_cast<std::uint32_t>(input.uint(4));
    const auto m = input.uint(8), n = input.uint(8), k = input.uint(8), numerical = input.uint(8);
    input.string(output ? &output->kernel_id : nullptr, budget);
    input.string(output ? &output->qualification_receipt_sha256 : nullptr, budget);
    const auto seconds = std::bit_cast<double>(input.uint(8));
    valid_cost(seconds);
    const auto samples = static_cast<std::uint32_t>(input.uint(4));
    const auto complete = input.boolean(), owners = input.boolean();
    if (output) {
        output->key.compute_units = compute_units;
        output->key.wave = static_cast<WaveSize>(wave);
        output->key.family = static_cast<OperatorFamily>(family);
        output->key.phase = static_cast<ExecutionPhase>(phase);
        output->key.shape = shape;
        output->key.m = m; output->key.n = n; output->key.k = k;
        output->key.numerical_contract_identity = numerical;
        output->median_complete_seconds = seconds; output->samples = samples;
        output->complete_workload = complete; output->owners_and_resources_qualified = owners;
    }
    input.finished();
}

void write_uint(std::vector<std::byte>& output, std::uint64_t value, unsigned width) {
    for (unsigned i = 0; i < width; ++i) output.push_back(static_cast<std::byte>((value >> (i * 8U)) & 255U));
}
void write_string(std::vector<std::byte>& output, std::string_view value) {
    write_uint(output, value.size(), 4);
    for (const unsigned char c : value) output.push_back(static_cast<std::byte>(c));
}
std::uint64_t record_size(const OfflineTuningRecord& record, StringBudget& budget) {
    valid_tags(static_cast<std::uint32_t>(record.key.wave), static_cast<std::uint32_t>(record.key.family),
        static_cast<std::uint32_t>(record.key.phase));
    valid_cost(record.median_complete_seconds);
    auto bytes = fixed_record_bytes;
    const auto measure = [&](std::string_view value) { budget.consume(value.size()); bytes = add(bytes, value.size()); };
    strings(record, measure); measure(record.kernel_id); measure(record.qualification_receipt_sha256);
    if (bytes > std::numeric_limits<std::uint32_t>::max()) invalid("offline tuning cache record exceeds frame limit");
    return bytes;
}

std::uint32_t read_header(Reader& input, OfflineTuningCacheLimits limits) {
    if (input.bytes.size() > limits.max_bytes) invalid("offline tuning cache exceeds byte limit");
    const auto prefix = input.take(magic.size());
    for (std::size_t i = 0; i < magic.size(); ++i)
        if (prefix[i] != magic[i]) invalid("offline tuning cache magic mismatch");
    if (input.uint(4) != version) invalid("offline tuning cache version unsupported");
    const auto count = static_cast<std::uint32_t>(input.uint(4));
    if (count > limits.max_records) invalid("offline tuning cache record count exceeds limit");
    if (count > (input.bytes.size() - input.offset) / (4 + fixed_record_bytes))
        invalid("offline tuning cache record count exceeds payload");
    return count;
}
} // namespace

std::vector<std::byte> encode_offline_tuning_cache(
    std::span<const OfflineTuningRecord> records, OfflineTuningCacheLimits limits) {
    if (records.size() > limits.max_records || records.size() > std::numeric_limits<std::uint32_t>::max())
        invalid("offline tuning cache record count exceeds limit");
    StringBudget budget{limits};
    auto bytes = header_bytes;
    for (const auto& record : records) bytes = add(bytes, add(4, record_size(record, budget)));
    if (bytes > limits.max_bytes || bytes > std::numeric_limits<std::size_t>::max())
        invalid("offline tuning cache exceeds byte limit");
    std::vector<std::byte> output;
    output.reserve(static_cast<std::size_t>(bytes));
    output.insert(output.end(), magic.begin(), magic.end());
    write_uint(output, version, 4); write_uint(output, records.size(), 4);
    for (const auto& r : records) {
        StringBudget frame_budget{limits};
        write_uint(output, record_size(r, frame_budget), 4);
        strings(r, [&](std::string_view value) { write_string(output, value); });
        write_uint(output, r.key.compute_units, 4); write_uint(output, static_cast<std::uint32_t>(r.key.wave), 4);
        write_uint(output, static_cast<std::uint32_t>(r.key.family), 4); write_uint(output, static_cast<std::uint32_t>(r.key.phase), 4);
        write_uint(output, r.key.shape.logical_rows, 4); write_uint(output, r.key.shape.request_concurrency, 4);
        write_uint(output, r.key.shape.hidden_size, 4); write_uint(output, r.key.shape.auxiliary_size, 4);
        write_uint(output, r.key.m, 8); write_uint(output, r.key.n, 8); write_uint(output, r.key.k, 8);
        write_uint(output, r.key.numerical_contract_identity, 8);
        write_string(output, r.kernel_id); write_string(output, r.qualification_receipt_sha256);
        write_uint(output, std::bit_cast<std::uint64_t>(r.median_complete_seconds), 8);
        write_uint(output, r.samples, 4); write_uint(output, r.complete_workload ? 1 : 0, 1);
        write_uint(output, r.owners_and_resources_qualified ? 1 : 0, 1);
    }
    return output;
}

std::vector<OfflineTuningRecord> decode_offline_tuning_cache(
    std::span<const std::byte> bytes, OfflineTuningCacheLimits limits) {
    Reader scan{bytes};
    const auto count = read_header(scan, limits);
    StringBudget scan_budget{limits};
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto size = scan.uint(4);
        Reader frame{scan.take(size)};
        read_record(frame, scan_budget, nullptr);
    }
    scan.finished();
    std::vector<OfflineTuningRecord> records(count);
    Reader input{bytes};
    static_cast<void>(read_header(input, limits));
    StringBudget budget{limits};
    for (auto& record : records) {
        const auto size = input.uint(4);
        const auto payload = input.take(size);
        Reader frame{payload};
        read_record(frame, budget, &record);
    }
    return records;
}
} // namespace strix::kernels
