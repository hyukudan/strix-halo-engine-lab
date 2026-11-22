#include "cuda/core/chain/launch.cuh"

#include "core/a4_tiles.hpp"
#include "core/residual_mix.hpp"
#include "core/row_transforms.hpp"
#include "core/short_convolution.hpp"
#include "cuda/models/ds41/ds41_cuda_common.cuh"
#include "strix/cuda/ds41_core.hpp"
#include "strix/cuda_core/norms.hpp"
#include "strix/cuda_core/rope.hpp"

#include <cuda_fp8.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

// The fixture links only the incumbent arithmetic translation unit. Its
// launch-status adapter avoids linking the engine's device-discovery code.
void strix::ds41::cuda::check(cudaError_t status, const char* what, const char*, int) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + " " + cudaGetErrorString(status));
}

namespace ch = strix::cuda::core::chain;
namespace cc = strix::cuda_core;
namespace ha = strix::head_argmax;
namespace head = strix::cuda::core::head_argmax;
namespace sf = strix::weight::scalar_float;
namespace rt = strix::core::row_transforms;
namespace bank = strix::core::residual_mix;
namespace conv = strix::core::causal_conv;
namespace activation = strix::moe_core::expert_activation;
using ch::Form;

namespace {

constexpr unsigned width = 128, stride = width + 5, position = 2, first_index = 19;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T>
struct Managed {
    T* p = nullptr;
    Managed() {
        check(cudaMallocManaged(&p, sizeof(T)));
        std::construct_at(p);
    }
    Managed(const Managed&) = delete;
    Managed& operator=(const Managed&) = delete;
    ~Managed() { cudaFree(p); }
    T* operator->() const { return p; }
};

struct Side {
    float output[stride];
    std::uint16_t shorts[stride];
    std::int8_t codes[stride];
    float scales[stride];
    int sums[stride];
    float state[stride];
    float mix[4 * width + 5];
    float carry[5];
    std::uint32_t ids[5];
    float values[5];
    unsigned flag;
};

struct Inputs {
    float pair[2 * width];
    float gains[width];
    std::uint16_t bgain[width];
    float table[2 * (position + 1U) * (width / 2U)];
    float banks[4 * width];
    float coefficients[24];
    unsigned classes[1]{1};
    unsigned position_on_device = position;
    float route_weight = -0.625f;
    ha::Pick partial;
    ha::Pick2 partial2;
};

struct Bindings {
    ch::ScalarStore store;
    ch::Gate gate;
    ch::Softcaps caps;
    ch::Softcap cap;
    ch::Affine affine;
    ch::ClassBias bias;
    ch::Norm norm;
    ch::Clip clip;
    ch::Rope rope;
    ch::Quant quant;
    ch::Emit emit;
    ch::Mix mix;
    ch::PendingRow pending[2];
    float* blocksum;
    float* lse;
    unsigned* flag;
};

template <Form F>
ch::ChainArgs bind(Bindings& b, Inputs& a, Side& out) {
    ch::ChainArgs args{};
    args.scalar[0] = -0.625f;
    if constexpr (F == Form::store_f32) {
        b.store = {out.state};
        args.aux[0] = &b.store;
    } else if constexpr (F == Form::store_e4m3) {
        b.store = {out.codes};
        args.aux[0] = &b.store;
    } else if constexpr (F == Form::store_bf16 || F == Form::store_f16 || F == Form::store_iu4prep) {
        b.store = {out.shorts};
        args.aux[0] = &b.store;
    } else if constexpr (F == Form::act_swiglu_clamp || F == Form::act_swiglu_oai || F == Form::gate_sigmoid_mul) {
        b.gate = {a.pair + width, 1.75f, F == Form::gate_sigmoid_mul};
        args.aux[0] = &b.gate;
    } else if constexpr (F == Form::act_swiglu_softcap) {
        b.caps = {a.pair + width, 4.0f, 25.0f, true};
        args.aux[0] = &b.caps;
    } else if constexpr (F == Form::softcap) {
        b.cap = {3.0f, true};
        args.aux[0] = &b.cap;
    } else if constexpr (F == Form::norm_rms || F == Form::norm_layer || F == Form::norm_group) {
        b.norm = {F == Form::norm_rms ? 1.0e-6f : 3.0f, a.gains};
        args.aux[0] = &b.norm;
    } else if constexpr (F == Form::clip_l2) {
        b.clip = {0.75f, 1.0e-6f, 2, 0.125f, a.bgain};
        args.aux[0] = &b.clip;
    } else if constexpr (F == Form::rope || F == Form::rope_inverse) {
        b.rope = {a.table, a.table + (position + 1U) * (width / 2U), width / 2U};
        args.aux[0] = &b.rope;
    } else if constexpr (F == Form::store_a8g32 || F == Form::quant_a8g32 || F == Form::quant_a4had ||
                         F == Form::quant_e4m3 || F == Form::quant_i8rot) {
        b.quant = {out.codes, out.scales, out.sums};
        args.aux[0] = &b.quant;
    } else if constexpr (F == Form::emit_argmax || F == Form::emit_top2 || F == Form::emit_topn) {
        b.emit = {out.ids, out.values, 4};
        args.aux[0] = &b.emit;
    } else if constexpr (F == Form::mix_expand) {
        b.mix = {a.banks, a.coefficients, out.mix, out.carry};
        args.aux[0] = &b.mix;
    } else if constexpr (F == Form::mod_affine) {
        b.affine = {a.gains, a.pair + width};
        args.aux[0] = &b.affine;
    } else if constexpr (F == Form::bias_by_class) {
        b.bias = {a.classes, a.pair};
        args.aux[0] = &b.bias;
    } else if constexpr (F == Form::route_weight_mul) {
        args.aux[0] = &a.route_weight;
    } else if constexpr (F == Form::blocksum) {
        b.blocksum = out.scales;
        args.aux[0] = &b.blocksum;
    } else if constexpr (F == Form::emit_lse) {
        b.lse = out.scales;
        args.aux[0] = &b.lse;
    } else if constexpr (F == Form::finite_status) {
        b.flag = &out.flag;
        args.aux[0] = &b.flag;
    } else if constexpr (ch::form_info(F).arg == ch::Arg::state) {
        args.state[0] = 1;
        if constexpr (F == Form::kv_append || F == Form::index_pool_append) b.pending[1] = {out.shorts, &out.flag};
        else b.pending[1] = {out.state};
    } else if constexpr (ch::form_info(F).arg == ch::Arg::aux) {
        args.aux[0] = a.pair + width;
    }
    return args;
}

// Mimic a producer with the input value already in its own accumulator.
template <Form F>
__global__ void embedded(const Inputs* a, Side* out, ch::ChainArgs args, const ch::PendingRow* pending) {
    __shared__ float scratch[ch::kThreads];
    const unsigned i = threadIdx.x;
    const ch::TileCtx ctx{width, 0, 0, position, first_index, scratch, pending};
    float accum = i < width ? a->pair[i] : 0.0f;
    using List = ch::List<ch::Step<F>>;
    if constexpr (ch::form_scope<F> == ch::Scope::element) {
        if (i < width) ch::apply_element(List{}, accum, i, args, ctx);
    } else if constexpr (ch::form_scope<F> == ch::Scope::warp) {
        ch::apply_warp(List{}, accum, i, args, ctx);
    } else {
        ch::RegisterRow<1> tile{{accum}};
        ch::apply(List{}, tile, args, ctx);
        accum = tile.values[0];
    }
    if (i < width) out->output[i] = accum;
}

// Separate reference expressions for forms without a standalone CUDA
// incumbent in this fixture. Conversion tests observe both representations.
template <Form F>
__global__ void scalar_reference(const Inputs* a, Side* out) {
    const unsigned i = threadIdx.x;
    if (i >= width) return;
    const float x = a->pair[i], up = a->pair[width + i];
    if constexpr (F == Form::store_f32) {
        out->state[i] = x;
    } else if constexpr (F == Form::store_bf16) {
        out->shorts[i] = __bfloat16_as_ushort(__float2bfloat16_rn(x));
        out->output[i] = __bfloat162float(__ushort_as_bfloat16(out->shorts[i]));
    } else if constexpr (F == Form::store_f16 || F == Form::store_iu4prep) {
        auto y = __float2half_rn(x);
        out->shorts[i] = __half_as_ushort(y);
        if constexpr (F == Form::store_f16) out->output[i] = __half2float(y);
    } else if constexpr (F == Form::store_e4m3) {
        const auto bits = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
        out->codes[i] = static_cast<std::int8_t>(bits);
        out->output[i] = sf::e4m3(bits);
    } else if constexpr (F == Form::scale_const || F == Form::scale_layer || F == Form::route_weight_mul) {
        out->output[i] = x * -0.625f;
    } else if constexpr (F == Form::bias_add || F == Form::bias_by_class) {
        out->output[i] = x + up;
    } else if constexpr (F == Form::act_silu_mul) {
        out->output[i] = (x / (1.0f + expf(-x))) * up;
    } else if constexpr (F == Form::act_swiglu_clamp) {
        out->output[i] = strix::ds41::swiglu_clamped(x, up, 1.75f);
    } else if constexpr (F == Form::act_sigmoid) {
        out->output[i] = 1.0f / (1.0f + expf(-x));
    } else if constexpr (F == Form::act_gelu_tanh_mul) {
        const float polynomial = x + 0.044715f * ((x * x) * x);
        out->output[i] = (0.5f * x * (1.0f + tanhf(0.7978845608f * polynomial))) * up;
    } else if constexpr (F == Form::act_swiglu_oai) {
        const float g = fminf(x, 1.75f), u = fmaxf(-1.75f, fminf(up, 1.75f));
        out->output[i] = (g * (1.0f / (1.0f + expf(-1.702f * g)))) * (1.0f + u);
    } else if constexpr (F == Form::mod_affine) {
        out->output[i] = x * a->gains[i] + up;
    } else if constexpr (F == Form::emit_lse) {
        if (i == 0) out->scales[0] = logf(static_cast<float>(width));
    } else if constexpr (F == Form::store_a8g32 || F == Form::quant_a8g32) {
        strix::ds41::cuda::detail::a8_row(a->pair, width, out->codes, out->scales, out->sums);
    }
}

__global__ void decode_activation(Side* out) {
    const unsigned i = threadIdx.x;
    if (i < width) out->output[i] = sf::u2f(std::uint32_t{out->shorts[i]} << 16U);
}

struct Pick1 {
    Side* side;
    __device__ void operator()(unsigned, const ha::Pick& pick) const {
        side->ids[0] = pick.index;
        side->values[0] = pick.value;
    }
};
struct Pick2 {
    Side* side;
    __device__ void operator()(unsigned, const ha::Pick2& pick) const {
        side->ids[0] = pick.first.index;
        side->ids[1] = pick.index2;
        side->values[0] = pick.first.value;
        side->values[1] = pick.value2;
    }
};

template <Form F>
constexpr bool incumbent = F == Form::residual_add || F == Form::residual_add_round_bf16 ||
    F == Form::norm_rms || F == Form::clip_l2 || F == Form::rope || F == Form::rope_inverse ||
    F == Form::load_round_bf16 || F == Form::gate_sigmoid_mul || F == Form::softcap || F == Form::act_swiglu_softcap ||
    F == Form::mix_expand || F == Form::conv_tap || F == Form::emit_argmax || F == Form::emit_top2 ||
    F == Form::quant_a8g32 || F == Form::store_a8g32;

template <Form F>
void enqueue_reference(Inputs* a, Side* e) {
    const auto* x = a->pair;
    const auto* rhs = x + width;
    const cc::GroupedView view{1, 1, width};
    if constexpr (F == Form::residual_add) {
        check(cc::add_inplace(e->output, rhs, width, nullptr));
    } else if constexpr (F == Form::residual_add_round_bf16) {
        check(bank::launch_add({x, rhs, e->output, width, true}, nullptr));
    } else if constexpr (F == Form::norm_rms) {
        check(cc::rmsnorm(x, view, a->gains, e->output, view, 1.0e-6f, nullptr));
    } else if constexpr (F == Form::clip_l2) {
        check(rt::l2_clip_bf16({x, a->bgain, e->output, 1, 1, width, 1.0e-6f, 0.75f,
                               {0, 2, 0.125f}}, nullptr, &a->position_on_device));
    } else if constexpr (F == Form::rope || F == Form::rope_inverse) {
        check(cc::rope_apply(e->output, view, width, a->table, a->table + (position + 1U) * (width / 2U),
                             &a->position_on_device, nullptr));
    } else if constexpr (F == Form::load_round_bf16) {
        check(bank::launch_round({x, e->output, width}, nullptr));
    } else if constexpr (F == Form::gate_sigmoid_mul) {
        check(rt::elements(width, rt::Bf16SigmoidProduct{x, rhs, e->output}, nullptr));
    } else if constexpr (F == Form::softcap) {
        check(rt::elements(width, rt::Bf16Softcap{e->output, 3.0f}, nullptr));
    } else if constexpr (F == Form::act_swiglu_softcap) {
        check(activation::launch({x, e->shorts, 1, width, 2U * width, width}, 4.0f, 25.0f, nullptr));
        decode_activation<<<1, width>>>(e);
        check(cudaGetLastError());
    } else if constexpr (F == Form::mix_expand) {
        strix::ds41::cuda::enqueue_hc_expand({x, width, a->banks, a->coefficients, 1, width, e->mix, e->carry}, nullptr);
    } else if constexpr (F == Form::conv_tap) {
        const strix::core::ShortConvolution plan{width, 2};
        check(conv::launch_with_store<false>(plan, {1, x, a->banks, nullptr, e->state, e->mix}, nullptr));
    } else if constexpr (F == Form::emit_argmax || F == Form::emit_top2) {
        const head::View v{x, width, width, 1, first_index, 1, &a->partial};
        if constexpr (F == Form::emit_argmax) check(head::pick_rows(v, Pick1{e}, nullptr));
        else check(head::pick2_rows(v, &a->partial2, Pick2{e}, nullptr));
    } else if constexpr (F == Form::norm_layer || F == Form::norm_group || F == Form::rotate_hadamard ||
                         F == Form::quant_a4had || F == Form::quant_i8rot || F == Form::quant_e4m3 ||
                         F == Form::blocksum || F == Form::emit_topn || F == Form::finite_status ||
                         F == Form::kv_append || F == Form::index_pool_append || F == Form::recur_commit) {
        // Independent host expectations below.
    } else {
        scalar_reference<F><<<1, width>>>(a, e);
        check(cudaGetLastError());
    }
}

template <Form F>
void set_expected(const Inputs& a, Side& expected) {
    const auto* x = a.pair;
    std::copy_n(x, width, expected.output);
    if constexpr (F == Form::norm_layer || F == Form::norm_group) {
        for (unsigned i = 0; i < width; ++i) expected.output[i] = (x[i] * 0.5f) * a.gains[i];
    } else if constexpr (F == Form::rotate_hadamard || F == Form::quant_a4had || F == Form::quant_i8rot) {
        float rotated[width];
        std::copy_n(x, width, rotated);
        strix::core::a4_tiles::fwht128(rotated);
        if constexpr (F == Form::rotate_hadamard) std::copy_n(rotated, width, expected.output);
        else {
            constexpr int limit = F == Form::quant_a4had ? 7 : 127;
            for (unsigned g = 0; g < width / 32U; ++g) {
                float maximum = 0.0f;
                for (unsigned j = 0; j < 32; ++j) maximum = std::max(maximum, std::abs(rotated[g * 32U + j]));
                const float delta = maximum == 0.0f ? 1.0f : maximum / static_cast<float>(limit);
                expected.scales[g] = delta;
                expected.sums[g] = 0;
                for (unsigned j = 0; j < 32; ++j) {
                    const auto i = g * 32U + j;
                    const int code = std::clamp(static_cast<int>(std::rint(rotated[i] / delta)), -limit, limit);
                    expected.codes[i] = static_cast<std::int8_t>(code);
                    expected.sums[g] += code;
                }
            }
        }
    } else if constexpr (F == Form::quant_e4m3) {
        float maximum = 0.0f;
        for (unsigned i = 0; i < width; ++i) maximum = std::max(maximum, std::abs(x[i]));
        expected.scales[0] = maximum / 448.0f;
        const float inv = 1.0f / expected.scales[0];
        for (unsigned i = 0; i < width; ++i) expected.codes[i] = static_cast<std::int8_t>(sf::e4m3_encode(x[i] * inv));
    } else if constexpr (F == Form::blocksum) {
        // All inputs are multiples of 1/16, so different addition trees are exact here.
        for (unsigned g = 0; g < width / 16U; ++g) {
            expected.scales[g] = 0.0f;
            for (unsigned j = 0; j < 16; ++j) expected.scales[g] += x[g * 16U + j];
        }
    } else if constexpr (F == Form::emit_topn) {
        float remaining[width];
        std::copy_n(x, width, remaining);
        for (unsigned rank = 0; rank < 4; ++rank) {
            const auto p = ha::reference(remaining, first_index);
            expected.ids[rank] = p.index;
            expected.values[rank] = p.value;
            remaining[p.index - first_index] = -std::numeric_limits<float>::infinity();
        }
    } else if constexpr (F == Form::finite_status) {
        expected.flag = 1;
    } else if constexpr (F == Form::kv_append || F == Form::index_pool_append) {
        expected.flag = !ch::cache::store_rows_ref(x, width, nullptr, 0, 1, {width, 0}, expected.shorts);
    } else if constexpr (F == Form::recur_commit) {
        std::copy_n(x, width, expected.state);
    }
}

template <Form F>
void fixture(const char* name) {
    Managed<Inputs> a;
    Managed<Side> actual, inline_result, expected;
    Managed<Bindings> b, inline_b;
    for (auto* side : {actual.p, inline_result.p, expected.p}) {
        std::memset(side, 0x75, sizeof(Side));
        side->flag = 0;
    }
    for (unsigned i = 0; i < width; ++i) {
        a->pair[i] = static_cast<float>(static_cast<int>((i * 19U) % 101U) - 50) / 16.0f;
        a->pair[width + i] = static_cast<float>(static_cast<int>((i * 7U) % 107U) - 53) / 19.0f;
        a->gains[i] = 0.75f + static_cast<float>(i % 8U) / 32.0f;
        a->bgain[i] = sf::bf16_encode(a->gains[i]);
    }
    if constexpr (F == Form::norm_layer || F == Form::norm_group)
        for (unsigned i = 0; i < width; ++i) a->pair[i] = i % 2U ? 1.0f : -1.0f;
    if constexpr (F == Form::emit_lse) std::fill_n(a->pair, width, 0.0f);
    if constexpr (F == Form::finite_status || F == Form::kv_append || F == Form::index_pool_append ||
                  F == Form::emit_argmax || F == Form::emit_top2 || F == Form::emit_topn) {
        a->pair[0] = std::numeric_limits<float>::infinity();
        a->pair[1] = 8.0f;
        a->pair[width - 1] = 8.0f;
    }
    constexpr unsigned table_size = (position + 1U) * (width / 2U);
    for (unsigned i = 0; i < table_size; ++i) {
        a->table[i] = static_cast<float>(i % 13U) / 13.0f;
        a->table[table_size + i] = static_cast<float>(i % 17U) / 17.0f;
    }
    for (unsigned i = 0; i < 4U * width; ++i) a->banks[i] = static_cast<float>(i % 17U) / 13.0f;
    for (unsigned i = 0; i < 24; ++i) a->coefficients[i] = static_cast<float>(i) / 29.0f;
    if constexpr (F == Form::conv_tap) std::fill_n(expected->state, width, 0.0f);
    set_expected<F>(*a.p, *expected.p);

    const auto args = bind<F>(*b.p, *a.p, *actual.p);
    const auto inline_args = bind<F>(*inline_b.p, *a.p, *inline_result.p);
    if constexpr (F == Form::rope_inverse)
        for (unsigned i = table_size; i < 2U * table_size; ++i) a->table[i] = -a->table[i];
    enqueue_reference<F>(a.p, expected.p);
    if constexpr (F == Form::rope_inverse) {
        check(cudaDeviceSynchronize());
        for (unsigned i = table_size; i < 2U * table_size; ++i) a->table[i] = -a->table[i];
    }
    using List = ch::List<ch::Step<F>>;
    const ch::Rows rows{a->pair, actual->output, 1, width, 0, stride, 1, 0, &a->position_on_device, first_index,
                        b->pending, 2};
    check(ch::instance<1, List>.enqueue(rows, args, nullptr));
    embedded<F><<<1, ch::kThreads>>>(a.p, inline_result.p, inline_args, inline_b->pending);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    if constexpr (F == Form::act_swiglu_softcap)
        std::memset(expected->shorts, 0x75, sizeof(expected->shorts)); // numeric payload was decoded for comparison
    if constexpr (F == Form::conv_tap)
        std::memset(expected->mix, 0x75, sizeof(expected->mix)); // compare the incumbent's newly written history
    if (std::memcmp(actual.p, inline_result.p, sizeof(Side)) != 0 ||
        std::memcmp(actual.p, expected.p, sizeof(Side)) != 0) {
        throw std::runtime_error(name);
    }
    std::printf("K6c %s %s\n", name, incumbent<F> ? "EXACT" : "REFERENCE");
}

constexpr unsigned wide = 513, wide_stride = wide + 11, wide_groups = 2;

struct Wide {
    float input[wide_groups * wide_stride];
    float residual[wide_groups * wide];
    float gates[wide_groups * wide];
    float gain[wide];
    float actual[wide_groups * wide_stride];
    float embedded[wide_groups * wide_stride];
    float expected[wide_groups * wide_stride];
    float added[wide_groups * wide_stride];
    ch::Norm norm;
    ch::Gate gate;
};

using WideList = ch::List<ch::Step<Form::residual_add, 1>, ch::Step<Form::norm_rms, 2>,
                          ch::Step<Form::gate_sigmoid_mul, 3>>;

__global__ void recompute(const Wide* input, float* result, ch::ChainArgs args) {
    __shared__ float scratch[ch::kThreads];
    const unsigned row = blockIdx.x;
    const ch::TileCtx ctx{wide, row, row, 0, 0, scratch};
    const auto tile = ch::load_prologue<4>(WideList{}, input->input + row * wide_stride, args, ctx);
    for (unsigned r = 0; r < 4; ++r) {
        const unsigned column = threadIdx.x + r * ch::kThreads;
        if (column < wide) result[row * wide_stride + column] = tile.values[r];
    }
}

void compound_fixture() {
    Managed<Wide> a;
    constexpr float epsilon = 1.0e-5f;
    std::fill_n(a->actual, wide_groups * wide_stride, -17.0f);
    std::fill_n(a->embedded, wide_groups * wide_stride, -17.0f);
    std::fill_n(a->expected, wide_groups * wide_stride, -17.0f);
    for (unsigned i = 0; i < wide; ++i) a->gain[i] = 1.0f + static_cast<float>(i % 17U) / 17.0f;
    for (unsigned g = 0; g < wide_groups; ++g) {
        for (unsigned i = 0; i < wide; ++i) {
            const auto raw = static_cast<float>(static_cast<int>((i * 19U + g) % 139U) - 69) / 31.0f;
            a->input[g * wide_stride + i] = raw;
            a->added[g * wide_stride + i] = raw;
            a->residual[g * wide + i] = 0.25f + static_cast<float>(i % 21U) / 21.0f;
            a->gates[g * wide + i] = raw * 2.0f;
        }
    }
    a->norm = {epsilon, a->gain};
    a->gate = {a->gates, 0.0f};
    ch::ChainArgs args{};
    args.aux[1] = a->residual;
    args.aux[2] = &a->norm;
    args.aux[3] = &a->gate;
    const cc::GroupedView padded{1, wide_groups, wide, 0, wide_stride}, tight{1, wide_groups, wide};
    for (unsigned g = 0; g < wide_groups; ++g)
        check(cc::add_inplace(a->added + g * wide_stride, a->residual + g * wide, wide, nullptr));
    check(cc::gated_rmsnorm(a->added, padded, a->gain, a->gates, tight, cc::GateKind::sigmoid, a->expected, padded,
                            epsilon, nullptr));
    const ch::Rows rows{a->input, a->actual, wide_groups, wide, wide_stride, wide_stride, wide_groups};
    check(ch::launch<4>(WideList{}, rows, args, nullptr));
    recompute<<<wide_groups, ch::kThreads>>>(a.p, a->embedded, args);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    if (std::memcmp(a->actual, a->expected, sizeof(a->actual)) != 0 ||
        std::memcmp(a->embedded, a->expected, sizeof(a->embedded)) != 0)
        throw std::runtime_error("residual + open/recompute RMS + sigmoid");
    std::puts("K6c residual_open_recompute_gate EXACT");

    // Two independent bank-norm jobs in one opened launch. Their reduction
    // uses explicit FMA partials and the halving tree, rather than the CUDA
    // warp-tree norm contract checked above.
    a->norm = {epsilon, nullptr, nullptr, nullptr, ch::RmsOrder::bank_halving, true};
    args.aux[2] = &a->norm;
    const bank::NormalizeArgs first{a->input, a->expected, 1, wide, epsilon, wide_stride, wide_stride, true};
    const bank::NormalizeArgs second{a->input + wide_stride, a->expected + wide_stride, 1, wide, epsilon,
                                     wide_stride, wide_stride, true};
    bank::detail::normalize_pair<<<dim3(1, wide_groups), ch::kThreads>>>(first, second);
    check(cudaGetLastError());
    check(ch::launch<4>(ch::List<ch::Step<Form::norm_rms, 2>>{}, rows, args, nullptr));
    check(cudaDeviceSynchronize());
    if (std::memcmp(a->actual, a->expected, sizeof(a->actual)) != 0)
        throw std::runtime_error("paired bank normalization");
    std::puts("K6c normalize_pair EXACT");
}

}  // namespace

int main() {
    try {
#define STRIX_CUDA_CHAIN_FORM(id, scope, arg) fixture<Form::id>(#id);
#include "cuda/core/chain/forms.inc"
#undef STRIX_CUDA_CHAIN_FORM
        compound_fixture();
        std::puts("K6c chain fixtures: 47/47 passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "K6c fixture failed: %s\n", error.what());
        return 1;
    }
}
