#include "strix/env_flags.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/weight/lowbit_pack.hpp"
#include "strix/weight/pack_source.hpp"
#include "strix/weight/registry.hpp"
#include "strix/hip_check.hpp"
#include "strix/hip_qwen_onfly_i4.hpp"
#include "strix/qwen_onfly_i4.hpp"
#include "strix/q6p_registry.hpp"
#include "core/iu4_onfly.hpp"
#include "core/wmma_tiled.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstring>
#include <optional>
#include <span>
#include <vector>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace strix {
namespace {
using strix::hip_check;
unsigned group_index(unsigned group) { return group == 1U ? 0U : group == 2U ? 1U : group == 4U ? 2U : 3U; }
void retire_device_copy(void* device, int allocation_device) noexcept {
    // Snapshots retain the backing through submission. Final retirement fences all
    // streams before release, including launches racing owner destruction/replacement.
    int previous_device = 0;
    if (hipGetDevice(&previous_device) != hipSuccess || hipSetDevice(allocation_device) != hipSuccess) {
        std::fprintf(stderr, "onfly INT4: device selection failed; retaining device allocation\n");
        return;
    }
    if (hipDeviceSynchronize() == hipSuccess) (void)hipFree(device);
    else std::fprintf(stderr, "onfly INT4: completion fence failed; retaining device allocation\n");
    (void)hipSetDevice(previous_device);
}
}  // namespace
HipQwenOnflyI4::~HipQwenOnflyI4() {
    if (!copies.empty())
        std::fprintf(stderr, "turbo-weights onfly: INT4 kernel launches %llu, registry hits %llu\n",
                     static_cast<unsigned long long>(q6p::iu4::launches.load()), static_cast<unsigned long long>(q6p::iu4::hits()));
    for (const auto& c : copies) {
        q6p::iu4::unregister_tensor(c.original, static_cast<const unsigned char*>(c.device));
    }
}
UmaOwnerInventory HipQwenOnflyI4::owner_inventory(std::shared_ptr<const HipQwenOnflyI4> owner) const {
    if (owner.get() != this) throw std::invalid_argument("onfly INT4 inventory requires its actual owner");
    std::vector<UmaOwnerView> views;
    for (const auto& c : copies) {
        UmaOwnerView view;
        view.view_identity = next_uma_owner_view_identity();
        view.backing = {0x5157524f4e464c59ULL, c.lease, 1U};
        view.category = "qwen_onfly_i4";
        view.allocation_bytes = view.view_bytes = c.bytes;
        view.allocation_pin = owner;
        view.address_space = UmaAddressSpace::device;
        view.address = reinterpret_cast<std::uintptr_t>(c.device);
        views.push_back(std::move(view));
    }
    return build_uma_owner_inventory(views);
}
std::string HipQwenOnflyI4::receipt_json() const {
    auto identity = resolved.json();
    identity.pop_back();
    std::string out = "  \"qwen_onfly_i4\":" + identity + ",\"tensors\":" +
        std::to_string(copies.size()) + ",\"source_q8_bytes\":" + std::to_string(source_bytes) + ",\"copy_bytes\":" +
        std::to_string(copy_bytes) + ",\"skipped\":" + std::to_string(skipped) + ",\"convert_seconds\":" +
        std::to_string(convert_seconds) + ",\"source_resident\":true,";
    if (resolved.converted()) {
        static constexpr char hex[] = "0123456789abcdef";
        std::string id;
        for (const auto b : pack_identity) { id += hex[b >> 4U]; id += hex[b & 15U]; }
        out += "\"pack_identity\":\"" + id + "\",\"kept_q8\":" + std::to_string(kept_q8) + ",";
    }
    out += "\"groups\":{";
    for (unsigned i = 0; i < 4U; ++i)
        out += std::string(i ? "," : "") + "\"" + std::string(onfly_i4::group_name(1U << i)) + "\":{\"tensors\":" +
            std::to_string(group_tensors[i]) + ",\"bytes\":" + std::to_string(group_copy_bytes[i]) + "}";
    return out + "}},\n";
}
namespace {
// converted: the calibrated pack (strix-lowbit-calib-pack, strix.q4_g64 matrices) bound to the raw SHA-256 of the model
// file the engine loads (the cached content hash the MTP target pin also reads), and the checkpoint index that names
// each source matrix so the pack's per-matrix source digest is checked against the bytes in the file.
struct ConvertedPack {
    weight::LowbitPack pack;
    weight::PackCheckpoint checkpoint;
};
std::unique_ptr<ConvertedPack> open_converted_pack(const onfly_i4::Resolved& resolved, const std::string& model_path) {
    if (model_path.empty()) throw std::invalid_argument("converted low-bit weights need the model file path");
    auto out = std::make_unique<ConvertedPack>();
    const auto sha = sha256_file_content(model_path, FileHashPolicy::cached);
    weight::Digest256 checkpoint{};
    checkpoint.bytes = sha.bytes;
    if (const auto s = weight::LowbitPack::open(resolved.pack, checkpoint, out->pack); !s)
        throw std::invalid_argument("converted low-bit pack " + resolved.pack + ": " + s.detail);
    if (const auto s = weight::PackCheckpoint::open(model_path, out->checkpoint); !s)
        throw std::invalid_argument("converted low-bit pack: model index " + model_path + ": " + s.detail);
    return out;
}
// The pack's strix.q4_g64 rows of `name` (source digest checked) relaid into the IU4 copy layout; false: not in the pack.
bool converted_copy(const ConvertedPack& converted, const std::string& name, std::uint32_t n, std::uint32_t k,
                    std::vector<std::uint8_t>& copy) {
    const auto* entry = converted.pack.entry(name);
    if (!entry) return false;
    const auto* raw = converted.checkpoint.find(name);
    weight::PackMatrixSource source;
    if (!raw || !weight::PackMatrixSource::bind(converted.checkpoint, *raw, source) || source.rows != n || source.columns != k)
        throw std::invalid_argument("converted low-bit pack: " + name + " has no matching Q8_0 source in the model");
    std::optional<weight::Slice> slice;
    if (const auto s = converted.pack.find(name, weight::kQ4G64WeightStorage, source, slice); !s || !slice)
        throw std::invalid_argument("converted low-bit pack: " + name + ": " + (s ? std::string("not strix.q4_g64") : s.detail));
    std::vector<std::uint8_t> q4(static_cast<std::size_t>(slice->range.bytes));
    if (q4.size() != std::size_t{n} * (k / 128U) * 68U)
        throw std::invalid_argument("converted low-bit pack: " + name + ": payload is not N x K/128 q4_g64 superblocks");
    if (const auto s = weight::read_slice(*slice, std::as_writable_bytes(std::span(q4))); !s)
        throw std::invalid_argument("converted low-bit pack: " + name + ": " + s.detail);
    copy.resize(core::onfly::copy_bytes(n, k));
    core::onfly::q4_g64_to_copy(q4.data(), n, k, copy.data());
    return true;
}
}  // namespace
std::shared_ptr<const HipQwenOnflyI4> load_hip_qwen_onfly_i4(const ModelManifest& manifest, const ModelRuntimeBinding& binding,
                                                             const onfly_i4::Resolved& resolved, const std::string& model_path) {
    auto owner = std::make_shared<HipQwenOnflyI4>(resolved);
    if (!resolved.active()) return owner;
    if (binding.tensors.size() != manifest.tensors.size() || binding.manifest_layout_identity != manifest.layout_identity)
        throw std::invalid_argument("onfly INT4 binding differs from manifest");
    const auto begin = std::chrono::steady_clock::now();
    const auto converted = resolved.converted() ? open_converted_pack(resolved, model_path) : nullptr;
    if (converted) owner->pack_identity = converted->pack.identity().bytes;
    std::vector<std::uint8_t> host_copy;
    // Calibration capture (diagnostic): the copies are named, the run computes with Q8_0 and records the rows.
    if (const char* tap = envcfg::text("STRIX_IU4_TAP_DIR"); tap && *tap) {
        q6p::iu4::tap_open(tap);
        std::fprintf(stderr, "onfly INT4: calibration tap open at %s (rows recorded, Q8_0 arithmetic)\n", tap);
    }
    int allocation_device = 0;
    hip_check(hipGetDevice(&allocation_device), "onfly INT4 allocation device");
    for (std::size_t i = 0; i < manifest.tensors.size(); ++i) {
        const auto& t = manifest.tensors[i];
        if (!qwen_onfly_i4_candidate(t, resolved)) continue;
        const auto* runtime = find_runtime_tensor_binding(binding, static_cast<std::uint32_t>(i));
        if (!runtime || !runtime->address || runtime->bytes < t.byte_size) throw std::invalid_argument("onfly INT4 source binding is incomplete");
        const auto* original = reinterpret_cast<const void*>(runtime->address);
        if (q6p::iu4::find(original)) { ++owner->skipped; continue; }
        const auto k = static_cast<std::uint32_t>(t.dimensions[0]), n = static_cast<std::uint32_t>(t.dimensions[1]);
        const auto padded = core::onfly::padded_outputs(n), bytes = core::onfly::copy_bytes(n, k);
        // Mixed widths: a converted pack holds the tensors coded at 4 bits; every other one keeps its Q8_0 rows.
        if (converted && !converted_copy(*converted, t.name, n, k, host_copy)) { ++owner->kept_q8; continue; }
        void* device = nullptr;
        hip_check(hipMalloc(&device, bytes), "onfly INT4 allocation");
        const std::shared_ptr<void> backing(device, [allocation_device](void* p) { retire_device_copy(p, allocation_device); });
        auto* base = static_cast<unsigned char*>(device);
        owner->copies.push_back({original, device, bytes, next_uma_owner_view_identity(), backing});
        if (converted) {
            hip_check(hipMemcpy(base, host_copy.data(), bytes, hipMemcpyHostToDevice), "converted INT4 upload");
        } else {
            hip_check(strix::core::convert_q8_0_iu4(static_cast<const std::uint8_t*>(original), static_cast<int>(k), static_cast<int>(n),
                                                    base, reinterpret_cast<std::uint16_t*>(base + core::onfly::packed_bytes(padded, k)), nullptr,
                                                    resolved.scale_rule), "onfly INT4 convert");
            hip_check(hipStreamSynchronize(nullptr), "onfly INT4 convert sync");
        }
        q6p::iu4::register_tensor(original, {base, n, k, backing});
        if (q6p::iu4::tap_active()) static_cast<void>(q6p::iu4::tap_name(original, t.name, n, k));
        owner->copy_bytes += bytes;
        owner->source_bytes += t.byte_size;
        const auto g = group_index(onfly_i4::classify(t.name));
        ++owner->group_tensors[g];
        owner->group_copy_bytes[g] += bytes;
    }
    owner->convert_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    return owner;
}
} // namespace strix
