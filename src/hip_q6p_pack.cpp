#include "strix/hip_q6p_pack.hpp"

#include "strix/gguf_reader.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/q6p_registry.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace strix {
namespace {

std::string metadata_string(const gguf::FileInfo& info, const std::string& key) {
    const auto* entry = info.find_metadata(key);
    return entry && entry->string_value ? *entry->string_value : std::string{};
}

struct Copy {
    std::size_t target = 0, pack = 0;
    std::uint64_t rows = 0, cols = 0;
    q6p::Kind kind = q6p::Kind::q6p32;
};

// The copy decodes to exactly the target tensor's bytes.
bool copy_matches(const Copy& copy, const unsigned char* target, const unsigned char* pack) {
    if (copy.kind == q6p::Kind::bf16) {
        for (std::uint64_t i = 0; i < copy.rows * copy.cols; ++i) {
            std::uint16_t half;
            std::uint32_t full;
            std::memcpy(&half, pack + 2U * i, 2U);
            std::memcpy(&full, target + 4U * i, 4U);
            if (static_cast<std::uint32_t>(half) << 16U != full) return false;
        }
        return true;
    }
    constexpr std::uint64_t kRows = 1024U;
    std::vector<unsigned char> decoded;
    const std::uint64_t q8_row = copy.cols / 32U * 34U, q6_row = copy.cols / 32U * 26U;
    for (std::uint64_t r = 0; r < copy.rows; r += kRows) {
        const auto rows = std::min(kRows, copy.rows - r);
        decoded.resize(rows * q8_row);
        q6p::to_q8_0(pack + r * q6_row, rows, copy.cols, decoded.data());
        if (std::memcmp(decoded.data(), target + r * q8_row, decoded.size()) != 0) return false;
    }
    return true;
}

}  // namespace

Q6pPackHeader read_q6p_pack_header(const std::filesystem::path& path) {
    const auto info = gguf::read_file(path);
    return {metadata_string(info, "strix.q6p.format"), metadata_string(info, "strix.q6p.source_content"),
        metadata_string(info, "strix.q6p.target_content"), metadata_string(info, "strix.q6p.target_file_sha256")};
}

HipQ6pPack::~HipQ6pPack() { q6p::clear(); }

std::shared_ptr<const HipQ6pPack> load_hip_q6p_pack(const std::filesystem::path& path, const MappedModelFile& target,
    const ModelManifest& target_manifest, const ModelRuntimeBinding& target_binding, AllocationGenerationSource& generations) {
    const auto info = gguf::read_file(path);
    if (metadata_string(info, "strix.q6p.format") != "Q6P32-v1")
        throw std::invalid_argument("dense Q6P32 pack has an unknown format");
    if (metadata_string(info, "strix.q6p.target_content") != model_content_identity_hex(target.content_identity()) ||
        metadata_string(info, "strix.q6p.target_file_sha256") != model_content_identity_hex(target.file_sha256()))
        throw std::invalid_argument("dense Q6P32 pack was built for a different target file");
    if (target_binding.tensors.size() != target_manifest.tensors.size())
        throw std::invalid_argument("dense Q6P32 pack target binding differs from its manifest");

    auto pack = std::make_shared<HipQ6pPack>();
    pack->mapped = std::make_shared<const MappedModelFile>(MappedModelFile::open_read_only(path, "dense-q6p-pack", generations));
    const auto manifest = build_model_manifest(info);
    HipModelResidencyPolicy policy;
    policy.reservation_limits.retained_bytes = pack->mapped->bytes() + (64ULL << 20U);
    policy.reservation_limits.setup_peak_bytes = pack->mapped->bytes() + (64ULL << 20U);
    pack->residency = std::make_shared<const HipModelResidency>(
        HipModelResidency::create(pack->mapped, manifest, generations, policy));
    const auto& binding = pack->residency->binding();
    if (binding.tensors.size() != manifest.tensors.size())
        throw std::invalid_argument("dense Q6P32 pack binding differs from its manifest");

    std::unordered_map<std::string, std::size_t> by_name;
    for (std::size_t i = 0; i < target_manifest.tensors.size(); ++i) by_name.emplace(target_manifest.tensors[i].name, i);
    std::vector<Copy> copies;
    for (std::size_t i = 0; i < manifest.tensors.size(); ++i) {
        const auto& t = manifest.tensors[i];
        const bool q6 = t.name.ends_with(".q6p"), bf16 = t.name.ends_with(".bf16");
        if (!q6 && !bf16) throw std::invalid_argument("dense Q6P32 pack tensor has an unknown suffix: " + t.name);
        const auto it = by_name.find(t.name.substr(0, t.name.size() - (q6 ? 4U : 5U)));
        if (it == by_name.end()) throw std::invalid_argument("dense Q6P32 pack tensor has no target: " + t.name);
        const auto& source = target_manifest.tensors[it->second];
        if (source.dimensions.size() != 2U) throw std::invalid_argument("dense Q6P32 pack target is not a matrix: " + t.name);
        Copy copy{it->second, i, source.dimensions[1], source.dimensions[0], q6 ? q6p::Kind::q6p32 : q6p::Kind::bf16};
        const bool shape = q6 ? source.ggml_type == 8U && copy.cols % 32U == 0U && t.ggml_type == 24U &&
                t.byte_size == q6p::q6p32_bytes(copy.rows, copy.cols) && copy.rows * (copy.cols / 32U) * 26U < (1ULL << 32U)
            : source.ggml_type == 0U && t.ggml_type == 30U && t.dimensions == source.dimensions;
        if (!shape) throw std::invalid_argument("dense Q6P32 pack tensor shape or type differs from its target: " + t.name);
        copies.push_back(copy);
    }

    // Every copy must decode to the target bytes that the original kernels read.
    std::atomic<std::size_t> next{0};
    std::atomic<bool> ok{true};
    std::vector<std::thread> workers;
    const unsigned threads = std::max(1U, std::min(16U, std::thread::hardware_concurrency()));
    for (unsigned w = 0; w < threads; ++w)
        workers.emplace_back([&] {
            for (std::size_t i = next++; i < copies.size() && ok; i = next++) {
                const auto& c = copies[i];
                const auto* t = reinterpret_cast<const unsigned char*>(target.base_address() +
                    target_manifest.tensors[c.target].absolute_offset);
                const auto* p = reinterpret_cast<const unsigned char*>(pack->mapped->base_address() +
                    manifest.tensors[c.pack].absolute_offset);
                if (!copy_matches(c, t, p)) ok = false;
            }
        });
    for (auto& w : workers) w.join();
    if (!ok) throw std::invalid_argument("dense Q6P32 pack copy does not decode to its target tensor");

    q6p::clear();
    for (const auto& c : copies) {
        q6p::register_tensor(reinterpret_cast<const void*>(target_binding.tensors[c.target].address),
            {reinterpret_cast<const unsigned char*>(binding.tensors[c.pack].address), static_cast<std::uint32_t>(c.rows),
                static_cast<std::uint32_t>(c.cols), c.kind});
        (c.kind == q6p::Kind::q6p32 ? pack->q6p_tensors : pack->bf16_tensors)++;
        pack->q6p_bytes += manifest.tensors[c.pack].byte_size;
        pack->replaced_bytes += target_manifest.tensors[c.target].byte_size;
    }
    return pack;
}

}  // namespace strix
