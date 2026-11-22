#pragma once
// Split-K policy of the shared matrix-vector cores: how many K ranges (blocks along K) one output group is split into, and
// therefore which fp32 partial sums the in-order reduce adds. Backend-neutral host logic (no device types): the kernel
// bodies stay one family per backend; the split target is a parameter of that family, chosen here.
//
// Arithmetic contract. ks = factor(target, out, in) depends on the SHAPE and the policy only, never on rows, so rows
// 1..32 of one call stay bitwise row-invariant (a verify row equals the serial row of the same identity). The reduce adds
// the ks partials in ascending K order; a different ks is a different fp32 summation order, i.e. different bits:
//   exact = target kExactTarget for every shape: the EXACT identity (nvfp4-splitk-exact-v1, today's bits).
//   fast  = target from kFastV1 when a row matches (device, out, in), else kExactTarget: FAST identity
//           nvfp4-splitk-fast-v1 (quality-gated; the same partial products, fewer and longer K ranges per block).
// Why the target is per shape: fewer splits cut the partial-sum bytes (2 ks rows out 4 B written then re-read) linearly,
// but also cut resident blocks per SM; below ~1.3 blocks/SM the weight stream loses memory-level parallelism. Where the
// optimum lies depends on out / in / fusion, so the FAST rows are measured, not derived.
// Table rule: rows are keyed by device properties read at run time (SM / CU count + architecture), never by a host name;
// a row is never edited: a new measurement is a new table version (new identity), so one identity has one set of bits per
// device. A device or shape without a row keeps kExactTarget (FAST == EXACT bits there).
// Bound: CUDA NVFP4 GEMV (src/cuda/core/dense/nvfp4.cu nvfp4_split_k; dense_big's exact prefill and the Qwen dense engine's
// workspace sizing call the same function). HIP: the dense IU4 GEMV reduces its K-waves inside the block (no cross-block
// split), so it has no row here; a HIP split core binds this same interface with HIP device keys.
#include <cstdint>
#include <string>
#include <string_view>

namespace strix::split_k {

enum class Profile : std::uint8_t { exact, fast };

// Device properties a FAST row is keyed by (queried from the device, never assumed).
struct Device {
    std::uint32_t units = 0;  // CUDA SMs / HIP CUs
    std::uint32_t arch = 0;   // CUDA: 10 * major + minor; HIP: gfx number
    constexpr bool operator==(const Device&) const = default;
};

// Split unit of the consuming kernel: an output group is tiles_per_block tiles of tile_out outputs; K is split in whole
// chunks of chunk_in inputs. Defaults = the CUDA NVFP4 GEMV (256 outputs per group, 128-input chunks).
struct Geometry {
    std::uint32_t tile_out = 16, tiles_per_block = 16, chunk_in = 128;
};

inline constexpr std::uint32_t kExactTarget = 1000;  // blocks per call aimed at by the EXACT identity
inline constexpr std::string_view kExactName = "nvfp4-splitk-exact-v1";
inline constexpr std::string_view kFastName = "nvfp4-splitk-fast-v1";

struct Rule {
    Device device;
    std::uint32_t out = 0, in = 0, target = 0;
};

// FAST v1: split-target sweep of the NVFP4 GEMV (targets 1000 / 500 / 250 / 2000, bracketed, rows 1..32, cold rotating
// weights, graph timing). Per shape: the target with the lowest rows-16 time whose gain exceeds twice the bracket drift
// and whose rows 1 and 8 are not slower than the EXACT target; shapes where no arm qualified have no row.
//   188 SMs, cc 12.0: out x in 12288 x 5120 / 10240 x 5120 / 6144 x 5120 / 5120 x 6144 -> 250 (rows-16 -5 .. -15 %),
//   17408 x 5120 (gate+up fused pair) -> 500 (-1.5 %); 5120 x 17408 keeps 1000 (250: +1.8 %, 500: +8.8 % at rows 16).
inline constexpr Rule kFastV1[] = {
    {{188, 120}, 12288, 5120, 250},
    {{188, 120}, 10240, 5120, 250},
    {{188, 120}, 6144, 5120, 250},
    {{188, 120}, 5120, 6144, 250},
    {{188, 120}, 17408, 5120, 500},
};

// Split target for one shape under a profile. exact_target is the EXACT value (kExactTarget; a test hook may change it).
[[nodiscard]] constexpr std::uint32_t target(Profile p, Device d, std::uint32_t out, std::uint32_t in,
                                             std::uint32_t exact_target = kExactTarget) {
    if (p == Profile::fast)
        for (const Rule& r : kFastV1)
            if (r.device == d && r.out == out && r.in == in) return r.target;
    return exact_target;
}

// True when the FAST table changes at least one shape on this device (otherwise FAST and EXACT bits are equal).
[[nodiscard]] constexpr bool fast_applies(Device d) {
    for (const Rule& r : kFastV1)
        if (r.device == d) return true;
    return false;
}

// Identity of the effective split policy on a device.
[[nodiscard]] constexpr std::string_view identity(Profile p, Device d) {
    return p == Profile::fast && fast_applies(d) ? kFastName : kExactName;
}

// Effective-policy text for the config print: the identity the device really runs and its key. device_key is the
// backend's wording of d ("188 SMs cc 12.0"; empty = unknown device). FAST requested on a device without rows says that
// FAST was requested and the bits are EXACT.
[[nodiscard]] inline std::string effective_text(Profile requested, Device d, std::string_view device_key) {
    const std::string on = " on " + std::string(device_key.empty() ? std::string_view("an unknown device") : device_key);
    if (requested == Profile::exact)
        return "exact " + std::string(kExactName) + on + " (target " + std::to_string(kExactTarget) + " for every shape)";
    if (fast_applies(d))
        return "fast " + std::string(kFastName) + on + " (per-shape targets of this device's rows; shapes without a row keep " +
               std::to_string(kExactTarget) + ")";
    return "FAST requested, bits EXACT: " + std::string(kExactName) + on + " (no FAST row for this device)";
}

// Split factor for a target: groups = ceil((out / tile_out) / tiles_per_block) (>= 1), ks = round(target / groups)
// clamped to [1, in / chunk_in]. Rows never enter.
[[nodiscard]] constexpr std::uint32_t factor(std::uint32_t tgt, std::uint32_t out, std::uint32_t in, Geometry g = {}) {
    const std::uint32_t nch = in / g.chunk_in;
    const std::uint32_t tiles = out / g.tile_out;
    const std::uint32_t grp = (tiles + g.tiles_per_block - 1) / g.tiles_per_block;
    const std::uint32_t groups = grp < 1 ? 1 : grp;
    const std::uint32_t ks = (tgt + groups / 2) / groups;
    return ks < 1 ? 1 : (nch < ks ? nch : ks);
}

}  // namespace strix::split_k
