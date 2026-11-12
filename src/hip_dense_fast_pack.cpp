#include "strix/hip_dense_fast_pack.hpp"

#include "strix/dense_fast_registry.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/gsq_a8_groups.hpp"
#include "strix/model_content_hash.hpp"

#include <stdexcept>
#include <unordered_map>

namespace strix {
namespace {
std::string metadata_string(const gguf::FileInfo& info, const std::string& key) {
    const auto* entry = info.find_metadata(key);
    return entry && entry->string_value ? *entry->string_value : std::string{};
}
}  // namespace

HipDenseFastPack::~HipDenseFastPack() { dense_fast::clear(); }

std::shared_ptr<const HipDenseFastPack> load_hip_dense_fast_pack(const std::filesystem::path& path, const MappedModelFile& target,
    const ModelManifest& target_manifest, const ModelRuntimeBinding& target_binding, AllocationGenerationSource& generations) {
    const auto info = gguf::read_file(path);
    if (metadata_string(info, "strix.dense_fast.format") != "DENSE-FAST-v1")
        throw std::invalid_argument("dense FAST pack has an unknown format");
    if (metadata_string(info, "strix.dense_fast.target_content") != model_content_identity_hex(target.content_identity()) ||
        metadata_string(info, "strix.dense_fast.target_file_sha256") != model_content_identity_hex(target.file_sha256()))
        throw std::invalid_argument("dense FAST pack was built for a different target file");
    if (target_binding.tensors.size() != target_manifest.tensors.size())
        throw std::invalid_argument("dense FAST pack target binding differs from its manifest");

    auto pack = std::make_shared<HipDenseFastPack>();
    pack->spec = metadata_string(info, "strix.dense_fast.spec");
    for (auto& c : pack->spec)
        if (c == '"' || c == '\\' || static_cast<unsigned char>(c) < 0x20U) c = '_';  // receipt-safe
    pack->mapped = std::make_shared<const MappedModelFile>(MappedModelFile::open_read_only(path, "dense-fast-pack", generations));
    pack->sha256 = model_content_identity_hex(pack->mapped->file_sha256());
    const auto manifest = build_model_manifest(info);
    HipModelResidencyPolicy policy;
    policy.reservation_limits.retained_bytes = pack->mapped->bytes() + (64ULL << 20U);
    policy.reservation_limits.setup_peak_bytes = pack->mapped->bytes() + (64ULL << 20U);
    pack->residency = std::make_shared<const HipModelResidency>(
        HipModelResidency::create(pack->mapped, manifest, generations, policy));
    const auto& binding = pack->residency->binding();
    if (binding.tensors.size() != manifest.tensors.size())
        throw std::invalid_argument("dense FAST pack binding differs from its manifest");

    std::unordered_map<std::string, std::size_t> by_name;
    for (std::size_t i = 0; i < target_manifest.tensors.size(); ++i) by_name.emplace(target_manifest.tensors[i].name, i);
    dense_fast::clear();
    for (std::size_t i = 0; i < manifest.tensors.size(); ++i) {
        const auto& t = manifest.tensors[i];
        const auto it = by_name.find(t.name);
        if (it == by_name.end()) throw std::invalid_argument("dense FAST pack tensor has no target: " + t.name);
        const auto& source = target_manifest.tensors[it->second];
        // Q4_K / Q5_K / Q6_K only: the types this pack's quantiser writes and its gate measured.
        const bool ok = source.ggml_type == 8U && source.dimensions.size() == 2U && t.dimensions == source.dimensions &&
            (t.ggml_type == 12U || t.ggml_type == 13U || t.ggml_type == 14U) && source.dimensions[0] % 256U == 0U &&
            source.dimensions[0] <= 8192U && source.dimensions[1] < (1ULL << 32U) &&
            t.byte_size == source.dimensions[1] * gsq_a8::dense_row_bytes(t.ggml_type, static_cast<std::uint32_t>(source.dimensions[0])) &&
            binding.tensors[i].address % 4U == 0U;
        if (!ok) throw std::invalid_argument("dense FAST pack tensor shape or type differs from its target: " + t.name);
        dense_fast::register_tensor(reinterpret_cast<const void*>(target_binding.tensors[it->second].address),
            {reinterpret_cast<const unsigned char*>(binding.tensors[i].address), static_cast<std::uint32_t>(source.dimensions[1]),
                static_cast<std::uint32_t>(source.dimensions[0]), t.ggml_type});
        ++pack->tensors;
        pack->packed_bytes += t.byte_size;
        pack->replaced_bytes += source.byte_size;
    }
    if (!pack->tensors) throw std::invalid_argument("dense FAST pack is empty");
    return pack;
}

}  // namespace strix
