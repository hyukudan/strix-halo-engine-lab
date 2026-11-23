#include <hip/hip_runtime.h>
#include "fp16_mirror.hpp"
#include "attn_core/cache_shadow.hpp"
#include "gemma4/kernels/internal.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace strix::gemma4::gpu {
namespace sc = strix::attn_core::cache_shadow;
static_assert(kv8_group == strix::attn::cache_shadow::group);
void fp16_mirror_sync(DeviceStream::Impl& stream, MirrorFormat format, std::uint32_t* failure, Queue queue) {
    if (format == MirrorFormat::none)
        throw std::invalid_argument("mirror format");
    const auto upto = stream.frontier / fp16_mirror_step * fp16_mirror_step;
    if (!stream.fp16_mirror) {
        if (!upto)
            return;
        auto mirror = std::make_shared<Fp16Mirror>();
        mirror->format = format;
        mirror->layer.resize(stream.cache.size());
        for (std::size_t l = 0; l < stream.cache.size(); ++l) {
            const auto& c = stream.cache[l];
            if (!c.width || !c.global || c.width % kv8_group)
                continue;
            auto& m = mirror->layer[l];
            const auto elements = 2ULL * c.slots * c.width;
            if (format == MirrorFormat::f16)
                m.values.resize(elements * sizeof(_Float16));
            else {
                m.values.resize(elements);
                m.scales.resize(elements / kv8_group * sizeof(std::uint16_t));
            }
            m.width = c.width;
            m.slots = c.slots;
        }
        stream.fp16_mirror = std::move(mirror);
    }
    if (stream.fp16_mirror->format != format)
        throw std::logic_error("stream mirror already holds another format");
    for (std::size_t l = 0; l < stream.cache.size(); ++l) {
        auto& m = stream.fp16_mirror->layer[l];
        m.pending = m.mirrored;
        if (!m.width || m.mirrored >= upto)
            continue;
        const auto& c = stream.cache[l];
        const auto rows = unsigned(upto - m.mirrored);
        const auto plane = std::size_t(m.slots) * m.width;
        const auto stream_q = static_cast<hipStream_t>(queue);
        if (format == MirrorFormat::f16) {
            const auto blocks = (std::size_t(rows) * m.width / 4U + 255U) / 256U;
            auto* halves = m.values.as<std::uint16_t>();
            hipLaunchKernelGGL(sc::mirror_f16_kernel<strix::attn::cache_shadow::Tagged>, dim3(unsigned(blocks)), dim3(256), 0, stream_q, c.k.as<float>(),
                               c.v.as<float>(), c.tags.as<std::uint64_t>(), m.mirrored, rows, m.width, halves,
                               halves + plane, failure);
        } else {
            const auto blocks = (std::size_t(rows) * (m.width / kv8_group) + 255U) / 256U;
            auto* codes = m.values.as<std::int8_t>();
            auto* scales = m.scales.as<std::uint16_t>();
            hipLaunchKernelGGL(sc::mirror_i8g16_kernel<strix::attn::cache_shadow::Tagged>, dim3(unsigned(blocks), 2), dim3(256), 0, stream_q, c.k.as<float>(),
                               c.v.as<float>(), c.tags.as<std::uint64_t>(), m.mirrored, rows, m.width, codes,
                               codes + plane, scales, scales + plane / kv8_group, failure);
        }
        launch_checked();
        m.pending = upto;
    }
}
void fp16_mirror_confirm(DeviceStream::Impl& stream) {
    if (stream.fp16_mirror)
        for (auto& m : stream.fp16_mirror->layer)
            m.mirrored = std::max(m.mirrored, m.pending);
}
Fp16MirrorRead fp16_mirror_read(const DeviceStream::Impl& stream, std::uint32_t producer) {
    if (!stream.fp16_mirror || producer >= stream.fp16_mirror->layer.size())
        return {};
    const auto& m = stream.fp16_mirror->layer[producer];
    if (!m.width)
        return {};
    const auto plane = std::size_t(m.slots) * m.width;
    Fp16MirrorRead out;
    out.mirrored = std::max(m.mirrored, m.pending);
    out.width = m.width;
    out.format = std::uint32_t(stream.fp16_mirror->format);
    if (stream.fp16_mirror->format == MirrorFormat::f16) {
        out.k = m.values.as<std::uint16_t>();
        out.v = m.values.as<std::uint16_t>() + plane;
    } else {
        out.k = m.values.as<std::int8_t>();
        out.v = m.values.as<std::int8_t>() + plane;
        out.k_scale = m.scales.as<std::uint16_t>();
        out.v_scale = m.scales.as<std::uint16_t>() + plane / kv8_group;
    }
    return out;
}
} // namespace strix::gemma4::gpu
