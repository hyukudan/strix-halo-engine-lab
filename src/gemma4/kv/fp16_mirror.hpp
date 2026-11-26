#pragma once

#include "storage.hpp"
#include "strix/attn/cache_shadow.hpp"
#include <cstdint>
#include <vector>

namespace strix::gemma4::gpu {
// Reduced-precision mirror of a device stream's committed GLOBAL-layer KV (one per stream, read by the FAST decode
// attention and, once adopted, the FAST flash prefill). Two formats, fixed per stream when it is first synced:
//  * f16: per global producer K rows [0, admitted) then V rows, row stride `width` halves, each element
//    static_cast<_Float16>(committed FP32 element) (RNE; an overflowing half sets failure bit 4).
//  * kv8: the same rows as int8 codes, groups of kv8_group consecutive features with one FP16 scale each
//    (shared Int8G16 shadow form, strix/attn/cache_shadow.hpp: d = half(amax / 127), code = clamp(rint(x / d), -127, 127); value = d * code).
// Rows [0, mirrored) are valid: committed rows never change (commits only append; forks and prefix-cache restores
// are new streams, whose mirror starts empty). Readers convert any row at or above the descriptor's `mirrored`
// (unsynced committed rows, overlay rows) in registers with the same operations, so the bytes they compute never
// depend on when the mirror was synchronized. The exact routes never read it; it is allocated only when a FAST
// reader syncs it.
// The form and the read descriptor are the shared shadow contract (strix/attn/cache_shadow.hpp; reader trait
// src/attn_core/shadow_kv_read.hpp).
using MirrorFormat = strix::attn::cache_shadow::Form;
inline constexpr std::uint32_t kv8_group = strix::attn::cache_shadow::group;
using Fp16MirrorRead = strix::attn::cache_shadow::ShadowRead;
struct Fp16Mirror {
    struct Layer {
        Buffer values;              // K rows then V rows (halves or codes)
        Buffer scales;              // kv8: K scales then V scales
        std::uint64_t mirrored = 0; // validated rows (a pass that read them completed without failure)
        std::uint64_t pending = 0;  // rows converted for the current pass, trusted only after fp16_mirror_confirm
        std::uint32_t width = 0, slots = 0;
    };
    MirrorFormat format = MirrorFormat::none;
    std::vector<Layer> layer; // by producer layer; width 0 = not a global producer
};
// Sync policy of the FAST readers: rows are converted in whole 64-row steps once the stream's committed frontier is
// at least one step past the mirror (the reader converts the rest itself).
inline constexpr std::uint32_t fp16_mirror_step = 64;
// Converts the committed rows [mirrored, frontier rounded down to a step) of every global producer of the stream on
// `queue` (allocating the mirror in `format` on first use; another format later throws). A position tag that does
// not match sets failure bit 2, an overflow bit 4. The stream must not be in the middle of a commit.
void fp16_mirror_sync(DeviceStream::Impl& stream, MirrorFormat format, std::uint32_t* failure, Queue queue);
// After a pass that synced and read the mirror completed with a zero failure word: the converted rows become valid.
// A failed pass never confirms, so its rows are converted (and checked) again by the next sync.
void fp16_mirror_confirm(DeviceStream::Impl& stream);
// Descriptor of one producer's mirror for the current pass (rows [0, pending), or an empty one when it has none).
[[nodiscard]] Fp16MirrorRead fp16_mirror_read(const DeviceStream::Impl& stream, std::uint32_t producer);
} // namespace strix::gemma4::gpu
