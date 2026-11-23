#include "exec/cuda_decode_wr2.hpp"
#include "strix/cuda_core/attention.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using namespace strix;
using namespace strix::exec::cuda_decode;
namespace conv = core::causal_conv;
namespace packed = moe_core::microscaled_r1;
namespace weight_pack = weight::microscaled;
namespace state_rows = core::decode_state;
using weight_pack::MatrixFormat;

void check(cudaError_t status) {
    device::native::check(status, "WR2 fixture");
}

void require(bool condition, std::string_view name) {
    if (!condition) throw std::runtime_error(std::string(name));
}

struct Stream {
    cudaStream_t value{};
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { (void)cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

template <class T>
struct Buffer {
    T* data{};
    std::size_t size;
    explicit Buffer(std::size_t count) : size(count) { check(cudaMallocManaged(&data, count * sizeof(T))); }
    ~Buffer() { (void)cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void fill(T value) { std::fill_n(data, size, value); }
    void copy(const T* source, std::size_t count, cudaStream_t stream) {
        check(cudaMemcpyAsync(data, source, sizeof(T) * count, cudaMemcpyDeviceToDevice, stream));
    }
};

template <class T>
struct Pair {
    Buffer<T> wide, serial;
    Pair(std::size_t size, T sentinel) : wide(size), serial(size) {
        wide.fill(sentinel);
        serial.fill(sentinel);
    }
    void compare(std::string_view name) const {
        require(std::memcmp(wide.data, serial.data, wide.size * sizeof(T)) == 0, name);
    }
};

void launch(BoundStepLeaf leaf, unsigned rows, cudaStream_t stream) {
    require(leaf.rows == rows && leaf.binding.predicted.launches == 1U && leaf.binding.capturable,
        "recipe width or launch count");
    leaf.binding.fn(leaf.binding.args, leaf.binding.grid, stream);
}

constexpr unsigned slots = 2U;
constexpr std::array<unsigned, slots> frontiers{31U, 65U};

exec::StepBlock descriptor(unsigned rows, bool padded) {
    exec::StepBlock step{};
    step.h.rows = rows;
    for (unsigned r = 0; r < rows; ++r) {
        auto& row = step.row[r];
        if (padded && (r + 1U == rows || r % 4U == 1U)) {
            row = {0U, ~0U, ~0U, ~0U, ~0U, 0U, 0U, 0U};
            continue;
        }
        const unsigned parent = r == 2U || r == 3U ? 0U : r == 4U ? 2U : exec::kStepNoParent;
        row.slot = parent == exec::kStepNoParent ? r % slots : step.row[parent].slot;
        row.parent = parent;
        row.ancestors = (1U << r) | (parent == exec::kStepNoParent ? 0U : step.row[parent].ancestors);
        row.position = parent == exec::kStepNoParent ? frontiers[row.slot] : step.row[parent].position + 1U;
        row.flags = exec::step_valid | exec::step_pending_state;
        step.h.valid_mask |= 1U << r;
    }
    return step;
}

float sample(std::size_t i, unsigned period, float scale) {
    return static_cast<float>(static_cast<int>(i % period) - static_cast<int>(period / 2U)) * scale;
}

// The scalar arm reconstructs each branch independently using the incumbent
// append/flash and convolution. A sibling never mutates the other arm's prefix.
void state_fixture(const exec::StepBlock& step, const exec::StepBlock* device_step, bool round, cudaStream_t stream) {
    constexpr unsigned capacity = 128U, ring = 48U, window = 40U, head = 128U;
    constexpr unsigned heads = 2U, qwidth = heads * head, rel = 45U;
    constexpr unsigned channels = 264U, taps = 4U, groups = 3U, history = (taps - 1U) * channels;
    constexpr std::uint16_t untouched_bf16 = 0x6badU;
    constexpr float untouched_float = 7'654.5F;
    const auto rows = step.h.rows;
    Buffer<std::uint16_t> committed_k(slots * capacity * head), committed_v(slots * capacity * head);
    Buffer<std::uint16_t> ring_v(slots * ring * head);
    ring_v.fill(untouched_bf16);
    Buffer<std::uint64_t> pages(slots * (capacity / state::page_rows));
    Buffer<float> committed_h(slots * history);
    Buffer<state::SlotEntry> entries(3U * slots);
    for (unsigned s = 0; s < slots; ++s) {
        for (unsigned p = 0; p < capacity; ++p) {
            for (unsigned c = 0; c < head; ++c) {
                const auto i = (s * capacity + p) * head + c;
                committed_k.data[i] = static_cast<std::uint16_t>(0x3e80U + (i * 13U + p) % 0x81U);
                committed_v.data[i] = static_cast<std::uint16_t>(0x3d80U + (i * 31U + p) % 0xa1U);
                if (p < frontiers[s]) ring_v.data[(s * ring + p % ring) * head + c] = committed_v.data[i];
            }
        }
        const auto page = s * (capacity / state::page_rows);
        for (unsigned i = 0; i < capacity / state::page_rows; ++i) {
            pages.data[page + i] = reinterpret_cast<std::uint64_t>(
                committed_k.data + (s * capacity + i * state::page_rows) * head);
        }
        const auto entry = [&](const void* data) {
            return state::SlotEntry{reinterpret_cast<std::uint64_t>(data), frontiers[s], 0U, capacity, s};
        };
        entries.data[s] = entry(pages.data + page);
        entries.data[slots + s] = entry(ring_v.data + s * ring * head);
        entries.data[2U * slots + s] = entry(committed_h.data + s * history);
    }
    for (std::size_t i = 0; i < committed_h.size; ++i) committed_h.data[i] = sample(i, 17U, 0.021F);
    // Compare the committed payloads as well as pending images and outputs.
    const std::vector original_k(committed_k.data, committed_k.data + committed_k.size);
    const std::vector original_h(committed_h.data, committed_h.data + committed_h.size);
    const std::vector original_entries(entries.data, entries.data + entries.size);
    const std::vector original_ring(ring_v.data, ring_v.data + ring_v.size);
    Buffer<float> query(rows * qwidth), key(rows * head), value(rows * head), bias(rows * heads * rel), sink(rows * heads);
    for (std::size_t i = 0; i < query.size; ++i) query.data[i] = sample(i, 73U, 0.011F);
    for (std::size_t i = 0; i < key.size; ++i) {
        key.data[i] = sample(i, 59U, 0.033F);
        value.data[i] = sample(i, 79U, 0.025F);
    }
    for (std::size_t i = 0; i < bias.size; ++i) bias.data[i] = sample(i, 11U, 0.042F);
    for (std::size_t i = 0; i < sink.size; ++i) sink.data[i] = sample(i, 13U, 0.111F);
    Buffer<float> input(rows * channels), weights(taps * channels), dynamic(rows * groups * taps);
    for (std::size_t i = 0; i < input.size; ++i) input.data[i] = sample(i, 397U, 0.009F);
    for (std::size_t i = 0; i < weights.size; ++i) weights.data[i] = sample(i, 29U, 0.016F);
    for (std::size_t i = 0; i < dynamic.size; ++i) dynamic.data[i] = sample(i, 47U, 0.003F);

    Pair<std::uint16_t> pending_k(rows * head, untouched_bf16), pending_v(rows * head, untouched_bf16);
    Pair<float> attention(rows * qwidth, untouched_float), convolution(rows * channels, untouched_float);
    Pair<float> pending_h(rows * history, untouched_float);
    Pair<unsigned> failure(rows, 0xcafeU);
    for (unsigned r = 0; r < rows; ++r) {
        if (r % 4U == 2U || !exec::step_row_valid(step, r)) {
            bias.data[(r * heads + 1U) * rel - 1U] = std::numeric_limits<float>::quiet_NaN();
        }
        if (exec::step_row_valid(step, r)) failure.wide.data[r] = failure.serial.data[r] = 0U;
    }
    Buffer<std::uint16_t> serial_k(capacity * head), serial_v(capacity * head);
    Buffer<float> serial_h(history), ignored(channels);
    const state::StateTable table{entries.data, 3U, slots};
    const attn_core::PreparedStepCache cache{
        {{table, 0U}, head * sizeof(std::uint16_t), 0U, true},
        {{table, 1U}, head * sizeof(std::uint16_t), ring, false}, pending_k.wide.data, pending_v.wide.data};
    const core::ShortConvolution plan{channels, taps, round ? groups : 0U, 0.685F, 1.075F};
    launch(bind_step(attn_core::PreparedStepKv{device_step, cache, key.data, value.data, 1U, head}, rows), rows, stream);
    launch(bind_step(attn_core::PreparedStepFlash{device_step, cache, query.data, bias.data, sink.data,
        attention.wide.data, heads, 1U, rel, window, 0.087F}, rows), rows, stream);
    launch(bind_step(attn_core::PreparedStepFinite{device_step, {query.data, key.data, value.data, bias.data, sink.data,
        attention.wide.data, 0U, rows}, heads, 1U, head, rel, failure.wide.data}, rows), rows, stream);
    launch(bind_step(conv::StepArgs{device_step, {table, 2U}, plan, input.data, weights.data, round ? dynamic.data : nullptr,
        pending_h.wide.data, convolution.wide.data, round}, rows), rows, stream);

    const auto transfer = [&](auto* to, const auto* from, std::size_t count) {
        check(cudaMemcpyAsync(to, from, count * sizeof(*to), cudaMemcpyDeviceToDevice, stream));
    };
    const cuda_core::AttentionDims dims{heads, 1U, head, 0.087F, capacity};
    auto* keys = reinterpret_cast<__nv_bfloat16*>(serial_k.data);
    auto* values = reinterpret_cast<__nv_bfloat16*>(serial_v.data);
    for (unsigned r = 0; r < rows; ++r) {
        if (!exec::step_row_valid(step, r)) continue;
        const auto& row = step.row[r];
        serial_k.copy(committed_k.data + row.slot * capacity * head, capacity * head, stream);
        serial_v.copy(committed_v.data + row.slot * capacity * head, capacity * head, stream);
        serial_h.copy(committed_h.data + row.slot * history, history, stream);
        for (unsigned ancestor = 0; ancestor <= r; ++ancestor) {
            if ((row.ancestors & (1U << ancestor)) == 0U) continue;
            check(cuda_core::kv_store_prepared_bf16(dims, key.data + ancestor * head, value.data + ancestor * head,
                keys, values, step.row[ancestor].position, 1U, false, stream));
            const conv::Args args{1U, input.data + ancestor * channels, weights.data,
                round ? dynamic.data + ancestor * groups * taps : nullptr, serial_h.data,
                ancestor == r ? convolution.serial.data + r * channels : ignored.data};
            check(round ? conv::launch_with_store<true>(plan, args, stream) : conv::launch_with_store<false>(plan, args, stream));
        }
        const auto* row_bias = bias.data + r * heads * rel;
        const auto* row_sink = sink.data + r * heads;
        check(cuda_core::attention_prefill_prepared_bf16(dims, {row_bias, row_sink, rel, window, false},
            query.data + r * qwidth, keys, values, row.position, 1U, attention.serial.data + r * qwidth, stream));
        check(attn_core::enqueue_validate_prepared_finite(
            {query.data + r * qwidth, key.data + r * head, value.data + r * head, row_bias, row_sink,
             attention.serial.data + r * qwidth, row.position, 1U}, heads, 1U, head, rel, failure.serial.data + r, stream));
        transfer(pending_k.serial.data + r * head, serial_k.data + row.position * head, head);
        transfer(pending_v.serial.data + r * head, serial_v.data + row.position * head, head);
        transfer(pending_h.serial.data + r * history, serial_h.data, history);
    }
    check(cudaStreamSynchronize(stream));
    pending_k.compare("kv_append keys");
    pending_v.compare("kv_append values");
    attention.compare("prepared_attention");
    failure.compare("prepared_finite");
    convolution.compare("causal_convolution output");
    pending_h.compare("causal_convolution history");
    require(std::equal(original_k.begin(), original_k.end(), committed_k.data) &&
            std::equal(original_h.begin(), original_h.end(), committed_h.data) &&
            std::equal(original_ring.begin(), original_ring.end(), ring_v.data) &&
            std::memcmp(original_entries.data(), entries.data, entries.size * sizeof(state::SlotEntry)) == 0,
            "committed state was modified");
}

constexpr unsigned experts = 32U, top = packed::kStepTopK;

struct RoutePolicy {
    static constexpr moe::RoutePolicy value{moe::RouteScore::sigmoid, true, moe::RouteKey::ordered_bits,
        moe::RouteOrder::selection, experts, top};
};
struct RouteMath { __host__ __device__ static float exp(float x) { return expf(x); } };
using RouteTraits = moe::RouteWaveTraits<true, true>;

template <MatrixFormat Format>
struct Matrices {
    using Operand = weight_pack::PackedOperand<Format>;
    unsigned rows, width;
    std::size_t codes_per, scales_per;
    unsigned offset;
    Buffer<std::uint8_t> codes, scales;

    Matrices(unsigned rows_, unsigned width_, bool unaligned)
        : rows(rows_), width(width_), codes_per(std::size_t{rows} * width / (Operand::kNvfp4 ? 2U : 1U)),
          scales_per(std::size_t{rows} * width / Operand::kBlockColumns), offset(unaligned ? 1U : 0U),
          codes(experts * codes_per + offset), scales(experts * scales_per + offset) {
        for (std::size_t i = 0; i < codes.size; ++i) {
            const auto code = static_cast<std::uint8_t>((i * 141U + i / 103U) & 0xffU);
            codes.data[i] = (code & 0x7fU) == 0x7fU && !Operand::kNvfp4 ? static_cast<std::uint8_t>(code - 1U) : code;
        }
        for (std::size_t i = 0; i < scales.size; ++i) {
            scales.data[i] = static_cast<std::uint8_t>(Operand::kNvfp4 ? 44U + i % 9U : 119U + i % 7U);
        }
    }
    weight_pack::PackedMatrix matrix(unsigned expert, weight_pack::RowOrder order) const {
        return {codes.data + offset + expert * codes_per, scales.data + offset + expert * scales_per,
            rows, width, 0.027F * static_cast<float>(expert + 3U), order};
    }
};

template <MatrixFormat Format>
void moe_fixture(const exec::StepBlock& step, const exec::StepBlock* device_step, bool padded, cudaStream_t stream) {
    constexpr unsigned width = Format == MatrixFormat::nvfp4 ? 272U : 160U;
    const auto rows = step.h.rows;
    const Matrices<Format> gate(2U * width, width, padded), down(width, width, padded);
    Buffer<moe_core::device_grouped::ExpertWeights> matrices(experts);
    for (unsigned e = 0; e < experts; ++e) {
        matrices.data[e] = {gate.matrix(e, weight_pack::RowOrder::interleaved_halves),
                            down.matrix(e, weight_pack::RowOrder::contiguous)};
    }
    Buffer<float> logits(rows * experts), bias(experts), input(rows * width);
    for (unsigned e = 0; e < experts; ++e) bias.data[e] = sample(e, 9U, 0.17F);
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned e = 0; e < experts; ++e) logits.data[r * experts + e] = sample(e + (r % 3U), 13U, 0.33F);
        if (r % 5U == 3U || !exec::step_row_valid(step, r)) logits.data[r * experts + 5U] = std::numeric_limits<float>::quiet_NaN();
    }
    for (std::size_t i = 0; i < input.size; ++i) input.data[i] = sample(i, 97U, 0.019F);
    Pair<unsigned> ids(rows * top, 0xfeadU), status(rows, 0xbeefU);
    Pair<float> weights(rows * top, 3'456.5F), output(rows * width, 3'456.5F);
    Pair<std::uint16_t> activation(rows * top * width, 0x6badU);
    launch(bind_step_route<RoutePolicy, RouteMath, RouteTraits>(
        {device_step, logits.data, bias.data, ids.wide.data, weights.wide.data, status.wide.data}, rows,
        "cuda-sm120-checked-wave-route-fixture"), rows, stream);
    const packed::StepArgs args{device_step, {matrices.data, ids.wide.data, weights.wide.data, status.wide.data,
        input.data, activation.wide.data, output.wide.data, experts, width, width, 3.0F, 2.0F}};
    launch(bind_step(StepPackedGateUp{args, Format}, rows), rows, stream);
    launch(bind_step(StepPackedDownCombine{args, Format}, rows), rows, stream);
    for (unsigned r = 0; r < rows; ++r) {
        if (!exec::step_row_valid(step, r)) continue;
        auto* row_ids = ids.serial.data + r * top;
        auto* row_weights = weights.serial.data + r * top;
        auto* row_status = status.serial.data + r;
        check(moe_core::enqueue_route_wave<RoutePolicy, RouteMath, RouteTraits>(
            logits.data + r * experts, bias.data, row_ids, row_weights, row_status, 1U, stream));
        const packed::Args scalar{matrices.data, row_ids, row_weights, row_status, input.data + r * width,
            activation.serial.data + r * top * width, output.serial.data + r * width, experts, width, width, 3.0F, 2.0F};
        check(packed::launch_gate_up<Format, top>(scalar, stream));
        check(packed::launch_down_combine<Format, top>(scalar, stream));
    }
    check(cudaStreamSynchronize(stream));
    ids.compare("checked_wave_route IDs");
    weights.compare("checked_wave_route weights");
    status.compare("checked_wave_route status");
    activation.compare("packed_gate_up");
    output.compare("packed_down_combine");
}

}  // namespace

int main() {
    try {
        const Stream stream;
        Buffer<exec::StepBlock> step(1U);
        for (const unsigned rows : {1U, 2U, 4U, 8U, 16U}) {
            for (const bool padded : {false, true}) {
                const auto host = descriptor(rows, padded);
                *step.data = host;
                state_fixture(host, step.data, padded, stream.value);
                moe_fixture<MatrixFormat::nvfp4>(host, step.data, padded, stream.value);
                moe_fixture<MatrixFormat::mxfp8>(host, step.data, padded, stream.value);
            }
        }
        std::cout << "wr2_state PASS widths=1,2,4,8,16 kv_append prepared_attention prepared_finite causal_convolution\n"
                     "wr2_moe PASS widths=1,2,4,8,16 checked_wave_route packed_gate_up packed_down_combine NVFP4 MXFP8\n"
                     "wr2_exact PASS scalar_bitwise=all padding_untouched=all\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "wr2 FAIL " << error.what() << '\n';
        return 1;
    }
}
