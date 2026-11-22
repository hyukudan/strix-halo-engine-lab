#include "cuda/core/attn_decode.cuh"

#include "core/row_transforms.hpp"
#include "strix/cuda_core/attention.hpp"
#include "strix/cuda_core/attention_row_walk.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
namespace attn = strix::cuda::core::attn;
namespace incumbent = strix::cuda_core;
namespace transform = strix::core::row_transforms;

void check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "k4c %s: %s\n", operation, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

template <typename T>
class Device {
public:
    explicit Device(std::size_t count) : count_(count) {
        if (count != 0) check(cudaMalloc(&ptr_, count * sizeof(T)), "cudaMalloc");
    }
    explicit Device(const std::vector<T>& data) : Device(data.size()) {
        check(cudaMemcpy(ptr_, data.data(), count_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    ~Device() { if (ptr_) (void)cudaFree(ptr_); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    [[nodiscard]] T* data() const { return ptr_; }
    void poison() const { check(cudaMemset(ptr_, 0xff, count_ * sizeof(T)), "poison output"); }
    [[nodiscard]] std::vector<T> download() const {
        std::vector<T> data(count_);
        check(cudaMemcpy(data.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return data;
    }

private:
    std::size_t count_;
    T* ptr_ = nullptr;
};

class Values {
public:
    explicit Values(unsigned seed) : state_(seed) {}
    float next(float scale) {
        state_ = state_ * 1664525U + 1013904223U;
        return static_cast<float>(static_cast<int>(state_ >> 16) - 32768) * (scale / 32768.0F);
    }
    template <typename T>
    std::vector<T> make(std::size_t count, float scale) {
        std::vector<T> result(count);
        for (auto& v : result) v = T(next(scale));
        return result;
    }

private:
    std::uint32_t state_;
};

void compare(const char* name, unsigned context, const std::vector<float>& want,
             const std::vector<float>& got) {
    for (std::size_t i = 0; i < want.size(); ++i) {
        const auto expected = std::bit_cast<std::uint32_t>(want[i]);
        const auto actual = std::bit_cast<std::uint32_t>(got[i]);
        if (std::isfinite(want[i]) && actual == expected) continue;
        std::fprintf(stderr, "k4c %s-%u FAIL element %zu: %08x vs %08x\n", name, context, i, actual, expected);
        std::exit(EXIT_FAILURE);
    }
}

void compare(const char* name, unsigned context, const Device<float>& reference,
             const Device<float>& candidate) {
    compare(name, context, reference.download(), candidate.download());
}

template <typename T>
void compare_bytes(const char* name, const std::vector<T>& want, const Device<T>& candidate) {
    const auto got = candidate.download();
    if (std::memcmp(want.data(), got.data(), want.size() * sizeof(T)) == 0) return;
    std::fprintf(stderr, "k4c %s FAIL (byte mismatch)\n", name);
    std::exit(EXIT_FAILURE);
}

attn::DecodePlan plan_for(const attn::DecodeSpec& spec) {
    const auto plan = attn::make_decode_plan(spec);
    if (!plan) {
        std::fputs("k4c fixture plan rejected\n", stderr);
        std::exit(EXIT_FAILURE);
    }
    return *plan;
}

void dense(unsigned context) {
    attn::DecodeSpec spec{256, 24, 4, 1, context + 1U, context + 1U};
    spec.gate = attn::OutputGate::sigmoid_interleaved;
    spec.scale = 0.0625F;
    const auto plan = plan_for(spec);
    const auto n = std::size_t{spec.heads} * spec.head_dim;
    const auto cache = std::size_t{spec.cache_slots} * spec.kv_heads * spec.head_dim;
    Values values(context);
    Device<float> qg(values.make<float>(2 * n, 2.75F));
    Device<__nv_bfloat16> k(values.make<__nv_bfloat16>(cache, 2.0F));
    Device<__nv_bfloat16> v(values.make<__nv_bfloat16>(cache, 0.75F));
    Device<std::uint32_t> position(std::vector<std::uint32_t>{context - 1U});
    const incumbent::AttentionDims dims{spec.heads, spec.kv_heads, spec.head_dim, spec.scale, spec.cache_slots};
    const incumbent::GroupedView view{1, spec.heads, spec.head_dim, spec.heads * spec.head_dim * 2U, spec.head_dim * 2U};
    Device<float> ref(n), got(n), ref_scratch(incumbent::attention_scratch_floats(dims, 1)), scratch(plan.scratch_floats);
    ref.poison();
    got.poison();
    check(incumbent::attention_decode(dims, qg.data(), view, qg.data() + spec.head_dim, view, k.data(), v.data(),
                                      position.data(), 1, ref.data(), ref_scratch.data(), nullptr), "dense incumbent");
    attn::DecodeInput input{qg.data(), k.data(), v.data()};
    input.device_position = position.data();
    check(attn::decode(plan, input, got.data(), scratch.data(), plan.scratch_floats, nullptr), "dense family");
    compare("qwen27b", context, ref, got);
    input.device_position = nullptr;
    input.first_position = context - 1U;
    check(attn::decode(plan, input, got.data(), scratch.data(), plan.scratch_floats, nullptr), "dense host position");
    compare("qwen27b", context, ref, got);
    Device<std::uint32_t> outside(std::vector<std::uint32_t>{spec.max_context});
    input.device_position = outside.data();
    check(attn::decode(plan, input, got.data(), scratch.data(), plan.scratch_floats, nullptr), "dense invalid replay");
    compare("qwen27b invalid replay leaves output", context, ref, got);
    spec.max_context = context;
    const auto fallback = plan_for(spec);
    input.device_position = position.data();
    got.poison();
    check(attn::decode(fallback, input, got.data(), scratch.data(), plan.scratch_floats, nullptr), "dense distinct extents");
    compare("qwen27b fallback", context, ref, got);
    std::printf("k4c qwen27b-%u PASS\n", context);
}

void prepared(const char* name, unsigned context, unsigned heads, unsigned kv_heads,
              unsigned window, unsigned rows, incumbent::FlashPipe pipe) {
    const unsigned slots = window == 0 ? context : window;
    const unsigned first = context - rows;
    attn::DecodeSpec spec{128, heads, kv_heads, rows, context, slots, window, 1024, 0.08838834764831844F,
        attn::DecodeOrder::online_bf16, window ? attn::KvLayout::ring : attn::KvLayout::slot_major,
        attn::OutputGate::none, true};
    const auto n = std::size_t{rows} * heads * spec.head_dim;
    const auto cache = std::size_t{slots} * kv_heads * spec.head_dim;
    Values values(context + heads);
    const auto q_values = values.make<float>(n, 2.75F);
    Device<float> q(q_values), g(values.make<float>(n, 4.0F));
    Device<float> rel(values.make<float>(std::size_t{rows} * heads * spec.relative_extent, 1.25F));
    Device<float> sinks(values.make<float>(std::size_t{rows} * heads, 3.0F));
    Device<__nv_bfloat16> k(values.make<__nv_bfloat16>(cache, 1.75F));
    Device<__nv_bfloat16> v(values.make<__nv_bfloat16>(cache, 0.75F));
    Device<std::uint32_t> position(std::vector<std::uint32_t>{first});
    const auto* device_position = rows == 1 ? position.data() : nullptr;
    const incumbent::AttentionDims dims{heads, kv_heads, spec.head_dim, spec.scale, slots};
    const incumbent::PreparedBf16Flash logits{rel.data(), sinks.data(), spec.relative_extent, window, window != 0};
    Device<float> ref(n), gated(n), got(n);
    ref.poison();
    got.poison();
    check(incumbent::attention_prefill_prepared_bf16(dims, logits, q.data(), k.data(), v.data(), first,
        rows, ref.data(), nullptr, pipe, device_position), "prepared incumbent");
    attn::DecodeInput input{q.data(), k.data(), v.data(), rel.data(), sinks.data(),
        g.data(), first, device_position};
    check(attn::decode(plan_for(spec), input, got.data(), nullptr, 0, nullptr), "prepared family");
    compare(name, context, ref, got);
    const auto other_pipe = pipe == incumbent::FlashPipe::pad2 ? incumbent::FlashPipe::swz3 : incumbent::FlashPipe::pad2;
    check(incumbent::attention_prefill_prepared_bf16(dims, logits, q.data(), k.data(), v.data(), first,
        rows, got.data(), nullptr, other_pipe, device_position), "prepared alternate pipe");
    compare(name, context, ref, got);

    gated.poison();
    got.poison();
    check(transform::elements(n, transform::Bf16SigmoidProduct{ref.data(), g.data(), gated.data()}, nullptr),
          "prepared incumbent gate");
    spec.gate = attn::OutputGate::sigmoid_bf16;
    check(attn::decode(plan_for(spec), input, got.data(), nullptr, 0, nullptr), "prepared gated family");
    compare(name, context, gated, got);
    std::vector<float> unaligned{0.0F};
    unaligned.insert(unaligned.end(), q_values.begin(), q_values.end());
    Device<float> strided_q(unaligned);
    input.q = strided_q.data() + 1;
    got.poison();
    check(attn::decode(plan_for(spec), input, got.data(), nullptr, 0, nullptr), "prepared unaligned fallback");
    compare(name, context, gated, got);
    std::printf("k4c %s-%u PASS\n", name, context);
}

void mapped_split(unsigned group, bool tree) {
    constexpr unsigned rows = 4, dim = 256, context = 272;
    const attn::DecodeSpec spec{dim, group, 1, rows, context, context, 0, 0, 0.0625F,
        attn::DecodeOrder::global_max, attn::KvLayout::head_major, attn::OutputGate::sigmoid_strided};
    const auto plan = plan_for(spec);
    const incumbent::AttentionDims dims{group, 1, dim, spec.scale, context};
    const std::size_t plane = context * dim, slot_stride = plane + 16;
    const std::vector<std::uint32_t> positions{255, 256, 255, 256}, slots{0, 0, 1, 1};
    const std::vector<std::uint32_t> index = tree ? std::vector<std::uint32_t>{255, 258, 255, 259} : positions;
    Device<std::uint32_t> pos(positions), slot(slots), self(index);
    const incumbent::RowMap map{pos.data(), slot.data(), slot_stride, tree ? self.data() : nullptr};
    Values values(group + 11U);
    const auto packed = group * dim, gate_stride = 2U * dim, gate_row = group * gate_stride + 16U;
    const incumbent::GroupedView qv{rows, group, dim}, gv{rows, group, dim, gate_row, gate_stride};
    Device<float> q(values.make<float>(rows * packed, 2.0F)), gate(values.make<float>(rows * gate_row, 2.0F));
    const auto key = values.make<float>(rows * dim, 1.0F), value = values.make<float>(rows * dim, 1.0F);
    Device<float> kd(key), vd(value);
    auto want_k = values.make<__nv_bfloat16>(2 * slot_stride, 1.5F);
    auto want_v = values.make<__nv_bfloat16>(2 * slot_stride, 0.75F);
    Device<__nv_bfloat16> k(want_k), v(want_v);
    check(incumbent::kv_store(dims, kd.data(), vd.data(), k.data(), v.data(), nullptr, rows, nullptr, map), "mapped append");
    for (unsigned r = 0; r < rows; ++r) {
        const auto dst = slots[r] * slot_stride + index[r] * dim;
        for (unsigned d = 0; d < dim; ++d) {
            want_k[dst + d] = __nv_bfloat16(key[r * dim + d]);
            want_v[dst + d] = __nv_bfloat16(value[r * dim + d]);
        }
    }
    compare_bytes("mapped append K", want_k, k);
    compare_bytes("mapped append V", want_v, v);

    Device<float> got(rows * packed), scratch(plan.scratch_floats);
    attn::DecodeInput input{q.data(), k.data(), v.data()};
    input.gate = gate.data();
    input.q_view = qv;
    input.gate_view = gv;
    input.row_map = map;
    got.poison();
    incumbent::set_decode_row_share(incumbent::DecodeRowShare::all);
    check(attn::decode(plan, input, got.data(), scratch.data(), plan.scratch_floats, nullptr), "mapped split plan");
    incumbent::set_decode_row_share(incumbent::DecodeRowShare::off);
    std::vector<float> want;
    for (unsigned r = 0; r < rows; ++r) {
        // Put the row's own K/V at its logical position in an independent single-cache view.
        const auto start = static_cast<std::ptrdiff_t>(slots[r] * slot_stride);
        const auto end = start + static_cast<std::ptrdiff_t>(plane);
        auto single_k = std::vector<__nv_bfloat16>(want_k.begin() + start, want_k.begin() + end);
        auto single_v = std::vector<__nv_bfloat16>(want_v.begin() + start, want_v.begin() + end);
        if (index[r] != positions[r]) {
            std::copy_n(single_k.data() + index[r] * dim, dim, single_k.data() + positions[r] * dim);
            std::copy_n(single_v.data() + index[r] * dim, dim, single_v.data() + positions[r] * dim);
        }
        Device<__nv_bfloat16> rk(single_k), rv(single_v);
        Device<float> row(packed);
        row.poison();
        check(incumbent::attention_decode(dims, q.data() + r * packed, qv, gate.data() + r * gate_row, gv,
            rk.data(), rv.data(), pos.data() + r, 1, row.data(), scratch.data(), nullptr), "mapped split reference");
        const auto copy = row.download();
        want.insert(want.end(), copy.begin(), copy.end());
    }
    compare("mapped split", group, want, got.download());
    std::printf("k4c production-split-g%u-%s PASS\n", group, tree ? "tree" : "slot");
}

void kv_placement() {
    constexpr unsigned slots = 8, kv_heads = 2, dim = 128, row_size = kv_heads * dim;
    const incumbent::AttentionDims dims{4, kv_heads, dim, 0.0625F, slots};
    Values values(27);
    const auto hk = values.make<float>(3 * row_size, 2.0F), hv = values.make<float>(3 * row_size, 1.0F);
    Device<float> kd(hk), vd(hv);
    std::vector<__nv_bfloat16> want_k(slots * row_size, __nv_bfloat16(-4.0F)), want_v = want_k;
    Device<__nv_bfloat16> k(want_k), v(want_v);
    Device<std::uint32_t> positions(std::vector<std::uint32_t>(3, 0));
    check(incumbent::iota_u32(positions.data(), 3, 7, 2, nullptr), "position iota");
    compare_bytes("iota", std::vector<std::uint32_t>{7, 9, 11}, positions);
    std::puts("k4c production-iota PASS");
    const auto out_size = 2U * dims.heads * dim;
    Device<float> q(values.make<float>(out_size, 2.0F)), out(out_size);
    Device<unsigned> failure(std::vector<unsigned>{0});
    const incumbent::PreparedBf16Flash prepared{nullptr, nullptr, 0, 4, true};
    check(incumbent::append_attention_prepared_bf16(dims, prepared, q.data(), kd.data(), vd.data(), k.data(),
        v.data(), 7, 2, out.data(), nullptr, incumbent::FlashPipe::pad2, failure.data()), "prepared host append");
    compare_bytes("prepared finite validation", std::vector<unsigned>{0}, failure);
    check(incumbent::kv_store_prepared_bf16(dims, kd.data() + 2 * row_size, vd.data() + 2 * row_size,
        k.data(), v.data(), 0, 1, true, nullptr, positions.data() + 1), "prepared step append");
    for (unsigned r = 0; r < 3; ++r) {
        const auto start = ((7U + r) % slots) * row_size;
        for (unsigned i = 0; i < row_size; ++i) {
            want_k[start + i] = __nv_bfloat16(hk[r * row_size + i]);
            want_v[start + i] = __nv_bfloat16(hv[r * row_size + i]);
        }
    }
    compare_bytes("prepared append K", want_k, k);
    compare_bytes("prepared append V", want_v, v);
    std::puts("k4c production-kv-prepared PASS");

    std::array<std::vector<__nv_bfloat16>, 4> planes;
    for (auto& plane : planes) plane = values.make<__nv_bfloat16>(slots * row_size, 1.0F);
    Device<__nv_bfloat16> k0(planes[0]), v0(planes[1]), k1(planes[2]), v1(planes[3]);
    const incumbent::KvLayerCaches caches{{k0.data(), k1.data()}, {v0.data(), v1.data()}};
    check(incumbent::kv_move_row(dims, caches, 2, 6, 2, nullptr), "kv move");
    for (auto& plane : planes) {
        for (unsigned h = 0; h < kv_heads; ++h) {
            auto* head = plane.data() + h * slots * dim;
            std::copy_n(head + 6 * dim, dim, head + 2 * dim);
        }
    }
    compare_bytes("move K0", planes[0], k0);
    compare_bytes("move V0", planes[1], v0);
    compare_bytes("move K1", planes[2], k1);
    compare_bytes("move V1", planes[3], v1);
    std::puts("k4c production-kv-move PASS");
}

void flash_and_shadow() {
    constexpr unsigned rows = 3, dim = 256, heads = 4, context = 2051, first = context - rows;
    const incumbent::AttentionDims dims{heads, 1, dim, 0.0625F, context};
    const incumbent::GroupedView qv{rows, heads, dim}, gv{rows, heads, dim, heads * 2U * dim, 2U * dim};
    const std::size_t output_size = rows * heads * dim;
    Values values(81);
    Device<float> q(values.make<float>(output_size, 1.0F)), gate(values.make<float>(2 * output_size, 2.0F));
    Device<__nv_bfloat16> k(values.make<__nv_bfloat16>(context * dim, 1.0F));
    Device<__nv_bfloat16> v(values.make<__nv_bfloat16>(context * dim, 0.5F));
    Device<float> ref(output_size), got(output_size);
    ref.poison();
    got.poison();
    check(incumbent::attention_prefill_flash(dims, q.data(), qv, gate.data(), gv, k.data(), v.data(), first, rows,
        ref.data(), nullptr), "pad2 prefill");
    check(incumbent::attention_prefill_flash(dims, q.data(), qv, gate.data(), gv, k.data(), v.data(), first, rows,
        got.data(), nullptr, incumbent::FlashPipe::swz3), "swz3 prefill");
    compare("production-flash", context, ref, got);
    std::puts("k4c production-flash-bf16 PASS");

    const auto capacity = incumbent::attn8_capacity(dims);
    const std::vector<std::uint8_t> bzero(capacity * dim);
    const std::vector<float> fzero(capacity);
    Device<std::uint8_t> k8(bzero), v8(bzero), rk8(bzero), rv8(bzero);
    Device<float> ks(fzero), vs(fzero), rks(fzero), rvs(fzero);
    const incumbent::KvShadowE4m3 shadow{k8.data(), ks.data(), v8.data(), vs.data()};
    const incumbent::KvShadowE4m3 reference{rk8.data(), rks.data(), rv8.data(), rvs.data()};
    check(incumbent::kv_shadow_e4m3(dims, k.data(), v.data(), reference, 0, context, nullptr), "full shadow oracle");
    check(incumbent::kv_shadow_e4m3(dims, k.data(), v.data(), shadow, 0, first, nullptr), "shadow prefix");
    Device<std::uint32_t> pos(std::vector<std::uint32_t>{first, first + 1, first + 2});
    const incumbent::RowMap map{pos.data()};
    const attn::DecodeSpec spec{dim, heads, 1, rows, context, context, 0, 0, dims.scale,
        attn::DecodeOrder::online_e4m3, attn::KvLayout::head_major, attn::OutputGate::sigmoid_strided};
    const auto plan = plan_for(spec);
    Device<float> scratch(plan.scratch_floats);
    attn::DecodeInput input{q.data(), k.data(), v.data()};
    input.gate = gate.data();
    input.q_view = qv;
    input.gate_view = gv;
    input.row_map = map;
    input.shadow = &shadow;
    got.poison();
    check(attn::decode(plan, input, got.data(), scratch.data(), plan.scratch_floats, nullptr), "FAST8 mapped decode");
    compare_bytes("shadow K", rk8.download(), k8);
    compare_bytes("shadow V", rv8.download(), v8);
    compare_bytes("shadow K scale", rks.download(), ks);
    compare_bytes("shadow V scale", rvs.download(), vs);
    std::puts("k4c production-shadow-e4m3 PASS");

    ref.poison();
    const auto row_size = heads * dim;
    for (unsigned r = 0; r < rows; ++r) {
        check(incumbent::attention_decode8(dims, q.data() + r * row_size, qv, gate.data() + r * 2U * row_size,
            gv, reference, pos.data() + r, 1, ref.data() + r * row_size, scratch.data(), nullptr), "FAST8 serial decode");
    }
    compare("FAST8 decode", context, ref, got);
    std::puts("k4c production-decode-e4m3 PASS");

    ref.poison();
    got.poison();
    check(incumbent::attention_prefill_flash8(dims, q.data(), qv, gate.data(), gv, reference, first, rows,
        ref.data(), nullptr), "FAST8 prefill bulk shadow");
    check(incumbent::attention_prefill_flash8(dims, q.data(), qv, gate.data(), gv, shadow, first, rows,
        got.data(), nullptr), "FAST8 prefill mapped shadow");
    compare("FAST8 prefill", context, ref, got);
    std::puts("k4c production-flash-e4m3 PASS");
}
}  // namespace

int main() {
    for (unsigned context : {1U, 17U, 4096U, 8191U}) {
        prepared("inkling2-window", context, 24, 8, 1024, 1, incumbent::FlashPipe::pad2);
        prepared("prepared-full", context, 12, 3, 0, context < 3U ? context : 3U, incumbent::FlashPipe::swz3);
        dense(context);
    }
    for (unsigned group : {3U, 6U, 8U}) {
        mapped_split(group, false);
        mapped_split(group, true);
    }
    kv_placement();
    flash_and_shadow();
}
