#pragma once

#include "core/markov_head_round.hpp"
#include "core/pointwise.hpp"
#include "strix/attn/cache_rows.hpp"
#include "strix/ds41_layer.hpp"
#include "strix/weight/host_q8.hpp"

namespace strix::core::markov_device {

struct Bf16Matrix {
    const std::uint16_t* data;
    unsigned width;
    __device__ float load(std::size_t row, unsigned column) const {
        return attn::cache_rows::decode(data[row * width + column]);
    }
};

struct Q8Matrix {
    const std::uint8_t* data;
    std::size_t stride;
    __device__ float load(std::size_t row, unsigned column) const {
        using Q8 = weight::Q8Storage;
        const auto* block = Q8::block(data + row * stride, column / Q8::elements);
        return __fmul_rn(ds41::f16_to_float(Q8::scale_bits(block)), static_cast<float>(Q8::code(block, column % Q8::elements)));
    }
};

template <class Matrix>
struct Fold {
    Matrix lm_head;
    const std::uint16_t* weight;
    const std::uint16_t* bias;
    std::uint16_t* folded;
    std::uint16_t* folded_bias;
    unsigned width;
    unsigned rank;

    __device__ void operator()(std::size_t i) const {
        const auto row = i / (rank + 1U);
        const auto column = i % (rank + 1U);
        double sum = 0.0;
        for (unsigned k = 0; k < width; ++k) {
            const auto b = column == rank ? bias[k] : weight[std::size_t{k} * rank + column];
            sum += static_cast<double>(lm_head.load(row, k)) * static_cast<double>(attn::cache_rows::decode(b));
        }
        if (column == rank) folded_bias[row] = markov_fold_bf16(sum);
        else folded[row * rank + column] = markov_fold_bf16(sum);
    }
};

struct CopyAffineSlice {
    const std::uint16_t* source;
    std::uint16_t* slice;
    unsigned stride;
    unsigned width;
    unsigned first;

    __device__ void operator()(std::size_t i) const {
        slice[i] = source[i / width * stride + first + i % width];
    }
};

struct PreviousEmbedding {
    const std::uint16_t* table;
    const std::uint32_t* previous;
    std::uint32_t anchor;
    unsigned vocabulary;
    unsigned width;
    float* output;

    __device__ void operator()(std::size_t i) const {
        const auto id = previous ? *previous : anchor;
        output[i] = id < vocabulary ? attn::cache_rows::decode(table[std::size_t{id} * width + i]) : 0.0F;
    }
};

struct Gate {
    const float* context_dot;
    const float* recurrent_dot;
    const std::uint16_t* bias;
    float* state;
    float* latent;
    unsigned rank;

    __device__ void operator()(std::size_t i) const {
        using weight::scalar_float::bf16_round;
        const auto raw = [&](std::size_t j) {
            return bf16_round(__fadd_rn(__fadd_rn(attn::cache_rows::decode(bias[j]), context_dot[j]), recurrent_dot[j]));
        };
        const auto g = bf16_round(__fdiv_rn(1.0F, __fadd_rn(1.0F, expf(-raw(i)))));
        const auto n = bf16_round(tanhf(raw(i + rank)));
        const auto kept = bf16_round(__fmul_rn(g, state[i]));
        const auto added = bf16_round(__fmul_rn(bf16_round(__fsub_rn(1.0F, g)), n));
        state[i] = bf16_round(__fadd_rn(kept, added));
        latent[i] = bf16_round(tanhf(raw(i + 2U * rank)));
    }
};

struct ComposeHead {
    float* base;
    const float* correction;
    const std::uint16_t* bias;
    float cap;

    __device__ void operator()(std::size_t i) const {
        using weight::scalar_float::bf16_round;
        const auto delta = bf16_round(__fadd_rn(correction[i], attn::cache_rows::decode(bias[i])));
        const auto score = bf16_round(__fadd_rn(bf16_round(base[i]), delta));
        base[i] = bf16_round(__fmul_rn(cap, tanhf(__fdiv_rn(score, cap))));
    }
};

}  // namespace strix::core::markov_device
