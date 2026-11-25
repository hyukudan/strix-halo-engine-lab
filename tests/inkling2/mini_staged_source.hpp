#pragma once

// A checkpoint-shaped Mini text source generated on the fly (no model file): the real text schema (every tensor name,
// dtype and extent) as a safetensors-indexed source whose payload is valid for everything the resident upload validates.
//   BF16 tensors          small finite values (so every attention weight and gain is finite and bounded)
//   NVFP4 outer scales    amax = 1.0 and denominator = 1.0 (F32), the sidecars the expert bind reads and checks (> 0, finite)
//   expert block scales   1.0 in their dtype (E4M3 0x38 for NVFP4, UE8M0 0x7F for MXFP8)
//   everything else       zeros (expert codes, F32 queries and router bias: never validated on load)
// plus an optional poisoned BF16 coefficient.

#include "inkling2/inkling2_mini_checkpoint.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mini_staged {

namespace wei = strix::weight;
namespace mini = strix::inkling2;

struct Poison {
    std::string tensor;  // full tensor name; empty = none
    std::uint64_t element = 0;
    std::uint16_t bits = 0;
};

class StagedSource final : public wei::Source {
public:
    explicit StagedSource(Poison poison = {}) : poison_(std::move(poison)) {
        std::string header = "{";
        std::uint64_t offset = 0;
        for (const auto& spec : mini::mini_text_tensor_schema()) {
            std::uint64_t count = 1;
            std::string shape;
            for (const auto dim : spec.extents) {
                count *= dim;
                shape += (shape.empty() ? "" : ",") + std::to_string(dim);
            }
            const std::string_view dtype = spec.wire_dtype, name = spec.name;
            const std::uint64_t width = dtype == "F32" ? 4U : (dtype == "BF16" ? 2U : 1U);
            const std::uint64_t bytes = count * width;
            header += (header.size() > 1U ? "," : "") + std::string("\"") + spec.name + "\":{\"dtype\":\"" +
                      std::string(dtype) + "\",\"shape\":[" + shape + "],\"data_offsets\":[" + std::to_string(offset) +
                      "," + std::to_string(offset + bytes) + "]}";
            Fill fill = Fill::zero;
            std::uint8_t byte = 0;
            if (dtype == "BF16") {
                fill = Fill::bf16;
            } else if (name.ends_with("_amax") || name.ends_with("_outer_scale_denominator")) {
                fill = Fill::f32_one;
            } else if (name.ends_with("_weight_scale")) {
                fill = Fill::byte;
                byte = dtype == "F8_E4M3" ? 0x38U : 0x7FU;  // 1.0 in E4M3 / UE8M0
            }
            tensors_.push_back({offset, offset + bytes, fill, byte, spec.name});
            offset += bytes;
        }
        header += "}";
        payload_ = offset;
        for (unsigned shift = 0; shift < 64; shift += 8)
            header_.push_back(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 0xFFU));
        header_ += header;
        std::uint32_t seed = 0x0DDBA11U;
        for (std::size_t i = 0; i < pattern_.size(); i += 2U) {
            seed = 1664525U * seed + 1013904223U;
            const auto value = static_cast<std::uint16_t>(0x3C00U | ((seed >> 16U) & 0x8000U) | ((seed >> 12U) & 0x01FFU));
            pattern_[i] = static_cast<wei::Byte>(value & 0xFFU);
            pattern_[i + 1U] = static_cast<wei::Byte>(value >> 8U);
        }
    }
    wei::Status read(wei::ByteRange range, std::span<wei::Byte> out) const override {
        if (range.bytes != out.size()) return {wei::Error::malformed, "staged source: size mismatch"};
        if (range.offset < header_.size()) {
            if (range.bytes > header_.size() - range.offset) return {wei::Error::malformed, "staged source: header range"};
            std::memcpy(out.data(), header_.data() + range.offset, out.size());
            return {wei::Error::ok, {}};
        }
        const auto begin = range.offset - header_.size(), end = begin + range.bytes;
        if (end > payload_) return {wei::Error::malformed, "staged source: payload range"};
        std::memset(out.data(), 0, out.size());
        for (const auto& tensor : tensors_) {
            if (tensor.end <= begin || tensor.begin >= end) continue;
            const auto first = std::max(begin, tensor.begin), last = std::min(end, tensor.end);
            if (tensor.fill == Fill::bf16) {
                for (auto at = first; at < last;) {
                    const auto inside = (at - tensor.begin) % pattern_.size();
                    const auto run = std::min<std::uint64_t>(last - at, pattern_.size() - inside);
                    std::memcpy(out.data() + (at - begin), pattern_.data() + inside, static_cast<std::size_t>(run));
                    at += run;
                }
            } else if (tensor.fill == Fill::f32_one) {
                constexpr std::array<wei::Byte, 4> one{wei::Byte{0x00}, wei::Byte{0x00}, wei::Byte{0x80}, wei::Byte{0x3F}};  // 1.0F LE
                for (auto at = first; at < last; ++at) out[static_cast<std::size_t>(at - begin)] = one[(at - tensor.begin) % 4U];
            } else if (tensor.fill == Fill::byte) {
                std::memset(out.data() + (first - begin), tensor.byte, static_cast<std::size_t>(last - first));
            }
            if (!poison_.tensor.empty() && tensor.name == poison_.tensor) {
                const auto at = tensor.begin + 2U * poison_.element;
                if (at >= first && at + 2U <= last) {
                    out[static_cast<std::size_t>(at - begin)] = static_cast<wei::Byte>(poison_.bits & 0xFFU);
                    out[static_cast<std::size_t>(at - begin) + 1U] = static_cast<wei::Byte>(poison_.bits >> 8U);
                }
            }
        }
        return {wei::Error::ok, {}};
    }
    std::uint64_t size() const noexcept override { return header_.size() + payload_; }
    wei::Digest256 identity() const override { throw std::logic_error("staged source has no digest"); }
    std::string shard_name() const override { return std::string(mini::kMiniCheckpointShard); }

private:
    enum class Fill { zero, bf16, f32_one, byte };
    struct Tensor {
        std::uint64_t begin, end;
        Fill fill;
        std::uint8_t byte;
        std::string name;
    };
    std::string header_;
    std::uint64_t payload_ = 0;
    std::vector<Tensor> tensors_;
    Poison poison_;
    std::array<wei::Byte, 4096> pattern_{};
};

}  // namespace mini_staged
