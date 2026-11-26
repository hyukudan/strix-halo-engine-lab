// MiniMax EXL3 pack reader for the RTX PRO prefill. See mm_rtx_pack.hpp.
#include "mm_rtx_pack.hpp"

#include "strix/serve/json.hpp"
#include "strix/weight/shard_name.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <system_error>

#if defined(_WIN32)
#define MM_FSEEK _fseeki64
#else
#define MM_FSEEK fseeko
#endif

namespace strix::minimax::rtxpf::host {

namespace {

constexpr std::uint64_t kIndexMax = std::uint64_t{64} << 20;   // as strix::weight::kSafetensorsManifestMaxBytes
constexpr std::uint64_t kHeaderMax = std::uint64_t{64} << 20;  // as strix::weight::SafetensorsIndexLimits

std::uint64_t file_bytes(const std::string& path) {
    std::error_code ec;
    const auto n = std::filesystem::file_size(path, ec);
    if (ec) throw std::runtime_error("cannot stat " + path);
    return static_cast<std::uint64_t>(n);
}

std::string slurp(const std::string& path, std::uint64_t max_bytes) {
    if (file_bytes(path) > max_bytes) throw std::runtime_error("file exceeds its size limit: " + path);
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string s;
    char buf[65536];
    std::size_t n;
    while (s.size() <= max_bytes && (n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    std::fclose(f);
    if (s.size() > max_bytes) throw std::runtime_error("file exceeds its size limit: " + path);
    return s;
}

const serve::Json& member(const serve::Json& o, const char* key) {
    const serve::Json* v = o.find(key);
    if (v == nullptr) throw std::runtime_error(std::string("safetensors header: missing ") + key);
    return *v;
}

std::uint64_t extent(const serve::Json& j) {
    const auto v = j.as_int();
    if (v < 0) throw std::runtime_error("safetensors header: negative extent");
    return static_cast<std::uint64_t>(v);
}

// An EXL3 matrix (trellis [input/16][output/16][bits*8] 16-bit words, suh [input] / svh [output] FP16): the shapes the
// slicers index with must agree with the byte extents the header declared.
void check_exl3(const std::string& prefix, const TensorInfo& tr, const TensorInfo& suh, const TensorInfo& svh) {
    const auto vec16 = [](const TensorInfo& t) { return t.shape.size() == 1 && t.shape[0] <= UINT64_MAX / 2 && t.bytes == t.shape[0] * 2; };
    bool ok = tr.shape.size() == 3 && vec16(suh) && vec16(svh);
    if (ok) {
        std::uint64_t count = 2;
        for (const std::uint64_t d : tr.shape) {
            if (d != 0 && count > UINT64_MAX / d) { ok = false; break; }
            count *= d;
        }
        ok = ok && count == tr.bytes;
    }
    if (!ok) throw std::runtime_error("EXL3 tensors of " + prefix + " disagree with their byte extents");
}

}  // namespace

float half_to_float(std::uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp(static_cast<float>(m), -24) : std::ldexp(static_cast<float>(m | 1024), e - 25);
    return s ? -v : v;
}

Pack::Pack(std::string dir, unsigned prefix_layers) : dir_(std::move(dir)) {
    const std::string index_path = dir_ + "/model.safetensors.index.json";
    const serve::Json idx = serve::parse_json_unique(slurp(index_path, kIndexMax));
    const serve::Json* weight_map = idx.find("weight_map");
    if (weight_map == nullptr || !weight_map->is_object()) throw std::runtime_error("no weight_map object in " + index_path);
    std::map<std::string, std::uint32_t> file_id;
    std::vector<std::string> names;
    for (const auto& [name, f] : weight_map->as_object()) {
        if (prefix_layers != 0) {
            const std::string lp = "language_model.model.layers.";
            bool keep = name == "language_model.model.embed_tokens.weight";
            if (name.compare(0, lp.size(), lp) == 0) {
                const auto dot = name.find('.', lp.size());
                keep = dot != std::string::npos && std::stoul(name.substr(lp.size(), dot - lp.size())) < prefix_layers;
            }
            if (!keep) continue;
        }
        // Shard names come from the file: only plain basenames inside dir_ are opened.
        if (!f.is_string() || !weight::shard_basename_ok(f.as_string()))
            throw std::runtime_error("invalid safetensors shard name for " + name + " in " + index_path);
        if (file_id.count(f.as_string()) == 0) {
            file_id[f.as_string()] = static_cast<std::uint32_t>(names.size());
            names.push_back(f.as_string());
        }
    }
    try {
    for (std::uint32_t i = 0; i < names.size(); ++i) {
        const std::string path = dir_ + "/" + names[i];
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open " + path);
        files_.push_back(f);
        const std::uint64_t size = file_bytes(path);
        std::uint64_t hlen = 0;
        if (size < 8 || std::fread(&hlen, 8, 1, f) != 1) throw std::runtime_error("bad header " + path);
        if (hlen < 2 || hlen > kHeaderMax || hlen > size - 8) throw std::runtime_error("bad header length " + path);
        const std::uint64_t data = size - 8 - hlen;  // payload bytes after the header
        std::string h(hlen, '\0');
        if (std::fread(h.data(), 1, hlen, f) != hlen) throw std::runtime_error("bad header " + path);
        const serve::Json hj = serve::parse_json_unique(h);
        for (const auto& [name, t] : hj.as_object()) {
            if (name == "__metadata__") continue;
            TensorInfo ti;
            ti.dtype = member(t, "dtype").as_string();
            for (const auto& d : member(t, "shape").as_array()) ti.shape.push_back(extent(d));
            const auto& off = member(t, "data_offsets").as_array();
            if (off.size() != 2) throw std::runtime_error("safetensors header: data_offsets needs two values: " + name);
            const std::uint64_t first = extent(off[0]), last = extent(off[1]);
            if (last < first || last > data) throw std::runtime_error("safetensors header: data_offsets outside the payload: " + name);
            ti.file = i;
            ti.offset = 8 + hlen + first;
            ti.bytes = last - first;
            tensors_[name] = ti;
        }
    }
    } catch (...) {
        for (auto* f : files_) std::fclose(f);
        throw;
    }
}

Pack::~Pack() {
    for (auto* f : files_) std::fclose(f);
}

const TensorInfo& Pack::info(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("tensor not found: " + name);
    return it->second;
}

void Pack::read(const TensorInfo& t, std::uint64_t offset, std::uint64_t n, void* dst) {
    if (offset > t.bytes || n > t.bytes - offset) throw std::runtime_error("read past tensor end");
    std::FILE* f = files_[t.file];
    if (MM_FSEEK(f, static_cast<long long>(t.offset + offset), SEEK_SET) != 0) throw std::runtime_error("seek");
    if (std::fread(dst, 1, n, f) != n) throw std::runtime_error("short read");
    bytes_read_ += n;
}

std::vector<float> Pack::floats(const std::string& name) {
    const TensorInfo& t = info(name);
    const std::uint64_t width = t.dtype == "F32" ? 4 : 2;
    if (t.bytes % width != 0) throw std::runtime_error("floats: " + name + " is not a whole number of " + t.dtype + " elements");
    std::vector<std::uint8_t> raw(t.bytes);
    read(t, 0, t.bytes, raw.data());
    std::vector<float> out;
    if (t.dtype == "F32") {
        out.resize(t.bytes / 4);
        std::memcpy(out.data(), raw.data(), t.bytes);
    } else if (t.dtype == "BF16" || t.dtype == "F16") {
        out.resize(t.bytes / 2);
        for (std::size_t i = 0; i < out.size(); ++i) {
            std::uint16_t h;
            std::memcpy(&h, raw.data() + 2 * i, 2);
            if (t.dtype == "BF16") {
                const std::uint32_t b = static_cast<std::uint32_t>(h) << 16;
                std::memcpy(&out[i], &b, 4);
            } else {
                out[i] = half_to_float(h);
            }
        }
    } else {
        throw std::runtime_error("floats: unsupported dtype " + t.dtype + " for " + name);
    }
    return out;
}

Exl3Host Pack::slice_out(const std::string& prefix, std::uint64_t o0, std::uint64_t n) {
    const TensorInfo &tr = info(prefix + ".trellis"), &suh = info(prefix + ".suh"), &svh = info(prefix + ".svh");
    check_exl3(prefix, tr, suh, svh);
    if (o0 % 128 != 0 || n % 128 != 0 || o0 > svh.shape[0] || n > svh.shape[0] - o0 || (o0 + n) / 16 > tr.shape[1])
        throw std::invalid_argument("bad output slice " + prefix);
    const std::uint64_t in_tiles = tr.shape[0], out_tiles = tr.shape[1], w = tr.shape[2] * 2;  // bytes per tile
    if (whole_name_ != prefix) {
        whole_.resize(tr.bytes);
        read(tr, 0, tr.bytes, whole_.data());
        whole_name_ = prefix;
    }
    const std::vector<std::uint8_t>& whole = whole_;
    Exl3Host e;
    e.trellis.resize(in_tiles * (n / 16) * w);
    for (std::uint64_t i = 0; i < in_tiles; ++i)
        std::memcpy(e.trellis.data() + i * (n / 16) * w, whole.data() + (i * out_tiles + o0 / 16) * w, (n / 16) * w);
    e.suh.resize(suh.shape[0]);
    read(suh, 0, suh.bytes, e.suh.data());
    e.svh.resize(n);
    read(svh, o0 * 2, n * 2, e.svh.data());
    e.input = static_cast<std::uint32_t>(suh.shape[0]);
    e.output = static_cast<std::uint32_t>(n);
    e.bits = static_cast<std::uint32_t>(tr.shape[2] / 16);
    return e;
}

Exl3Host Pack::slice_in(const std::string& prefix, std::uint64_t i0, std::uint64_t n) {
    const TensorInfo &tr = info(prefix + ".trellis"), &suh = info(prefix + ".suh"), &svh = info(prefix + ".svh");
    check_exl3(prefix, tr, suh, svh);
    if (i0 % 128 != 0 || n % 128 != 0 || i0 > suh.shape[0] || n > suh.shape[0] - i0) throw std::invalid_argument("bad input slice " + prefix);
    const std::uint64_t row_bytes = tr.shape[1] * tr.shape[2] * 2;
    Exl3Host e;
    e.trellis.resize((n / 16) * row_bytes);
    read(tr, (i0 / 16) * row_bytes, e.trellis.size(), e.trellis.data());
    e.suh.resize(n);
    read(suh, i0 * 2, n * 2, e.suh.data());
    e.svh.resize(svh.shape[0]);
    read(svh, 0, svh.bytes, e.svh.data());
    e.input = static_cast<std::uint32_t>(n);
    e.output = static_cast<std::uint32_t>(svh.shape[0]);
    e.bits = static_cast<std::uint32_t>(tr.shape[2] / 16);
    return e;
}

}  // namespace strix::minimax::rtxpf::host
