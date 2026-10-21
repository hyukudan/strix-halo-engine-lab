#include "strix/deepseek41_attention_state.hpp"

#include "strix/deepseek41_attention_host.hpp"
#include "strix/ds41_attention.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace strix::deepseek41 {
namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL, kFnvPrime = 1099511628211ULL;
constexpr char kMagic[8] = {'D', 'S', '4', '1', 'A', 'T', 'S', 'T'};
constexpr std::uint32_t kVersion = 1;

std::uint64_t fnv_bytes(const void* data, std::size_t n, std::uint64_t h = kFnvOffset) {
    const auto* b = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * kFnvPrime;
    return h;
}

// FNV of the live window of a ring: positions max(0, T - window) .. T - 1 in order (as the runner's ring_window_hash).
std::uint64_t ring_window_hash(const std::uint8_t* ring, std::uint32_t positions) {
    namespace attn = ds41::attn;
    std::uint64_t h = kFnvOffset;
    const std::uint32_t first = positions > attn::kWindow ? positions - attn::kWindow : 0U;
    for (std::uint32_t p = first; p < positions; ++p)
        h = fnv_bytes(ring + std::size_t{p % AttentionHost::ring_capacity} * attn::kWindowEntryBytes, attn::kWindowEntryBytes, h);
    return h;
}

struct File {
    std::FILE* f = nullptr;
    std::string path;
    std::uint64_t sum = kFnvOffset;  // FNV of every byte written / read so far (the file's checksum)
    File(const std::filesystem::path& p, const char* mode) : f(std::fopen(p.string().c_str(), mode)), path(p.string()) {
        if (f == nullptr) throw std::runtime_error("deepseek41 attention state: cannot open " + path);
    }
    ~File() {
        if (f != nullptr) std::fclose(f);
    }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    void write(const void* p, std::size_t n) {
        if (n != 0U && std::fwrite(p, 1, n, f) != n) throw std::runtime_error("deepseek41 attention state: write failed: " + path);
        sum = fnv_bytes(p, n, sum);
    }
    void read(void* p, std::size_t n) {
        if (n != 0U && std::fread(p, 1, n, f) != n) throw std::runtime_error("deepseek41 attention state: truncated file: " + path);
        sum = fnv_bytes(p, n, sum);
    }
    template <typename T>
    void put(const T& v) { write(&v, sizeof(T)); }
    template <typename T>
    T get() {
        T v{};
        read(&v, sizeof(T));
        return v;
    }
    template <typename T>
    void put_vec(const std::vector<T>& v) {
        put<std::uint64_t>(v.size());
        write(v.data(), v.size() * sizeof(T));
    }
    template <typename T>
    std::vector<T> get_vec(std::uint64_t limit) {
        const auto n = get<std::uint64_t>();
        if (n > limit) throw std::runtime_error("deepseek41 attention state: corrupt size in " + path);
        std::vector<T> v(n);
        read(v.data(), n * sizeof(T));
        return v;
    }
};

}  // namespace

std::vector<std::pair<std::string, std::uint64_t>> AttentionState::hashes() const {
    std::vector<std::pair<std::string, std::uint64_t>> out;
    for (std::size_t l = 0; l < layers.size(); ++l) {
        const auto& L = layers[l];
        const std::string at = "@" + std::to_string(l);
        out.emplace_back("ring" + at, ring_window_hash(L.ring.data(), position));
        if (L.owns_kv) {
            out.emplace_back("kv" + at, fnv_bytes(L.kv.data(), L.kv.size()));
            out.emplace_back("idx_keys" + at, fnv_bytes(L.idx_keys.data(), L.idx_keys.size()));
            if (L.ratio == 2U) out.emplace_back("carry" + at, L.carry_valid ? fnv_bytes(L.carry.data(), L.carry.size() * 4U) : 0U);
        }
        if (L.owns_index)
            out.emplace_back("topk" + at, L.topk_known ? fnv_bytes(L.topk.data(), L.topk.size() * 4U, L.topk.size()) : 0U);
    }
    return out;
}

std::uint64_t AttentionState::digest(bool without_topk) const {
    std::uint64_t h = kFnvOffset;
    for (const auto& [name, v] : hashes()) {
        if (without_topk && name.rfind("topk", 0) == 0) continue;
        h = fnv_bytes(name.data(), name.size(), h);
        h = fnv_bytes(&v, sizeof(v), h);
    }
    h = fnv_bytes(tokens.data(), tokens.size() * 4U, h);
    h = fnv_bytes(&position, 4U, h);
    return fnv_bytes(&next_token, 4U, h);
}

std::uint64_t AttentionState::bytes() const {
    std::uint64_t n = 0;
    for (const auto& L : layers) n += L.ring.size() + L.kv.size() + L.idx_keys.size() + L.carry.size() * 4U + L.topk.size() * 4U;
    return n;
}

void save_attention_state(const std::filesystem::path& path, const AttentionState& s) {
    const auto tmp = std::filesystem::path(path.string() + ".tmp");
    {
        File f(tmp, "wb");
        f.write(kMagic, sizeof(kMagic));
        f.put(kVersion);
        f.put(s.position);
        f.put(s.next_token);
        f.put<std::uint32_t>(static_cast<std::uint32_t>(s.layers.size()));
        for (const auto& L : s.layers) {
            const std::uint8_t flags = (L.owns_kv ? 1U : 0U) | (L.owns_index ? 2U : 0U) | (L.carry_valid ? 4U : 0U) | (L.topk_known ? 8U : 0U);
            f.put(flags);
            f.put(L.ratio);
            f.put_vec(L.ring);
            f.put_vec(L.kv);
            f.put_vec(L.idx_keys);
            f.put_vec(L.carry);
            f.put_vec(L.topk);
        }
        f.put_vec(s.tokens);
        f.put(s.digest());
        const std::uint64_t checksum = f.sum;
        f.put(checksum);
        if (std::fflush(f.f) != 0) throw std::runtime_error("deepseek41 attention state: flush failed: " + tmp.string());
    }
    std::filesystem::rename(tmp, path);
}

AttentionState load_attention_state(const std::filesystem::path& path) {
    File f(path, "rb");
    char magic[sizeof(kMagic)];
    f.read(magic, sizeof(magic));
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) throw std::runtime_error("deepseek41 attention state: not a state file: " + path.string());
    if (f.get<std::uint32_t>() != kVersion) throw std::runtime_error("deepseek41 attention state: unknown version: " + path.string());
    AttentionState s;
    s.position = f.get<std::uint32_t>();
    s.next_token = f.get<std::uint32_t>();
    const auto layers = f.get<std::uint32_t>();
    if (layers > 64U) throw std::runtime_error("deepseek41 attention state: corrupt layer count");
    constexpr std::uint64_t big = std::uint64_t{1} << 36U;
    s.layers.resize(layers);
    for (auto& L : s.layers) {
        const auto flags = f.get<std::uint8_t>();
        L.owns_kv = (flags & 1U) != 0U;
        L.owns_index = (flags & 2U) != 0U;
        L.carry_valid = (flags & 4U) != 0U;
        L.topk_known = (flags & 8U) != 0U;
        L.ratio = f.get<std::uint32_t>();
        L.ring = f.get_vec<std::uint8_t>(big);
        L.kv = f.get_vec<std::uint8_t>(big);
        L.idx_keys = f.get_vec<std::uint8_t>(big);
        L.carry = f.get_vec<float>(big);
        L.topk = f.get_vec<std::uint32_t>(big);
    }
    s.tokens = f.get_vec<std::uint32_t>(big);
    const auto digest = f.get<std::uint64_t>();
    const std::uint64_t checksum = f.sum;
    if (f.get<std::uint64_t>() != checksum || digest != s.digest())
        throw std::runtime_error("deepseek41 attention state: checksum or digest mismatch (corrupt file): " + path.string());
    if (s.tokens.size() != s.position) throw std::runtime_error("deepseek41 attention state: token history length != position");
    return s;
}

std::uint64_t attention_state_device_bytes(std::uint32_t max_positions, std::uint32_t layers) {
    namespace attn = ds41::attn;
    const auto plan = attn::released_plan();
    std::uint64_t n = 0;
    for (std::uint32_t l = 0; l < layers && l < plan.size(); ++l) {
        const auto& r = plan[l];
        n += std::uint64_t{AttentionHost::ring_capacity} * attn::kWindowEntryBytes;
        if (r.owns_kv) {
            const std::uint64_t entries = max_positions / r.ratio + 1U;
            n += entries * (attn::kKvEntryBytes + attn::kIdxEntryBytes) + 2U * (2U * attn::kHeadDim * 4U + 4U);
        }
        if (r.owns_index) n += std::uint64_t{8} * (attn::kTopK + 1U) * 4U;
    }
    return n;
}

}  // namespace strix::deepseek41
