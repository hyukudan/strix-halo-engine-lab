#include "strix/file_snapshot.hpp"
#include "strix/model_input.hpp"
#include "strix/ds41_native_fp8.hpp"

#include "strix/serve/json.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strix::ds41 {

namespace {

// The device butterfly: v[l] = v[l] + v[l ^ d] for d = 16 .. 1; lane 0.
float butterfly(std::array<float, product_lanes> lanes) noexcept {
    for (unsigned d = 16U; d; d /= 2U) {
        std::array<float, product_lanes> next{};
        for (unsigned l = 0; l < product_lanes; ++l) next[l] = lanes[l] + lanes[l ^ d];
        lanes = next;
    }
    return lanes[0];
}

std::string read_bytes(const std::filesystem::path& file, std::uint64_t offset, std::uint64_t bytes) {
    strix::ModelInput in(file);
    if (!in) throw std::runtime_error("ds41 native fp8: cannot open " + file.string());
    in.seekg(static_cast<std::streamoff>(offset));
    std::string out(static_cast<std::size_t>(bytes), '\0');
    if (!in.read(out.data(), static_cast<std::streamsize>(bytes))) throw std::runtime_error("ds41 native fp8: short read of " + file.string());
    return out;
}

struct SiteName {
    const char* model;     // suffix of blk.L.<model>
    const char* original;  // layers.L.<original>
    const char* site;
};
constexpr std::array<SiteName, 8> site_names{{
    {"attn_q_a.weight", "attn.wq_a", "q_a"},
    {"attn_q_b.weight", "attn.wq_b", "q_b"},
    {"attn_kv.weight", "attn.wkv", "kv"},
    {"attn_output_a.weight", "attn.wo_a", "o_a"},
    {"attn_output_b.weight", "attn.wo_b", "o_b"},
    {"ffn_gate_shexp.weight", "ffn.shared_experts.w1", "shexp"},
    {"ffn_up_shexp.weight", "ffn.shared_experts.w3", "shexp"},
    {"ffn_down_shexp.weight", "ffn.shared_experts.w2", "shexp"},
}};

std::optional<std::pair<std::string, const SiteName*>> split_model_name(const std::string& name) {
    static const std::regex re(R"(blk\.(\d+)\.(.+))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    for (const auto& s : site_names)
        if (m[2] == s.model) return std::make_pair(m[1].str(), &s);
    return std::nullopt;
}

}  // namespace

float fp8_dot_scale(std::uint8_t e) noexcept { return std::bit_cast<float>(static_cast<std::uint32_t>(e - 9U) << 23U); }

std::string fp8_tensor_check(std::span<const std::uint8_t> codes, std::span<const std::uint8_t> scales, std::size_t rows,
                             std::size_t width) {
    if (rows == 0U || width == 0U || rows % fp8_block != 0U || width % fp8_block != 0U) return "rows and width must be multiples of 32";
    if (codes.size() != rows * width || scales.size() != rows / fp8_block * (width / fp8_block)) return "codes or exponents of the wrong size";
    for (const auto c : codes)
        if ((c & 0x7FU) == 0x7FU) return "NaN code";
    for (const auto e : scales)
        if (e < fp8_min_exponent || e > fp8_max_exponent) return "block exponent " + std::to_string(e) + " outside [10, 254]";
    return {};
}

float product_fp8(const std::uint8_t* codes, const std::uint8_t* scales, const std::int8_t* x, const float* deltas, std::size_t width) {
    if (width == 0U || width % fp8_block != 0U) throw std::invalid_argument("ds41 product_fp8: width must be a multiple of 32");
    std::array<float, product_lanes> lanes{};
    for (std::size_t g = 0; g < width / fp8_block; ++g) {
        std::int32_t dot = 0;
        for (std::size_t j = 0; j < fp8_block; ++j) dot += e4m3_integer(codes[fp8_block * g + j]) * x[fp8_block * g + j];
        const float term = static_cast<float>(dot) * (fp8_dot_scale(scales[g]) * deltas[g]);
        lanes[g % product_lanes] = lanes[g % product_lanes] + term;
    }
    return butterfly(lanes);
}

float fp8_weight(const std::uint8_t* tensor, std::size_t rows, std::size_t width, std::size_t r, std::size_t c) {
    const std::uint8_t e = tensor[rows * width + r / fp8_block * (width / fp8_block) + c / fp8_block];
    return e4m3_to_float(tensor[r * width + c]) * e8m0_to_float(e);
}

std::optional<std::string> fp8_site_class(const std::string& model_name) {
    const auto s = split_model_name(model_name);
    if (!s) return std::nullopt;
    return std::string(s->second->site);
}

std::optional<std::string> fp8_original_name(const std::string& model_name) {
    const auto s = split_model_name(model_name);
    if (!s) return std::nullopt;
    return "layers." + s->first + "." + s->second->original;
}

std::vector<std::string> parse_fp8_site_classes(const std::string& list) {
    static const std::array<const char*, 6> all{"q_a", "q_b", "kv", "o_a", "o_b", "shexp"};
    std::vector<std::string> out;
    std::stringstream in(list);
    std::string item;
    while (std::getline(in, item, ',')) {
        if (item.empty()) continue;
        if (item == "all") {
            out.assign(all.begin(), all.end());
            continue;
        }
        if (std::find_if(all.begin(), all.end(), [&](const char* a) { return item == a; }) == all.end())
            throw std::invalid_argument("ds41 native fp8: unknown site class '" + item + "' (q_a, q_b, kv, o_a, o_b, shexp, all)");
        if (std::find(out.begin(), out.end(), item) == out.end()) out.push_back(item);
    }
    return out;
}

SafetensorsResolution resolve_safetensors_checkpoint(const std::filesystem::path& dir) {
    SafetensorsResolution out;
    const auto index = dir / "model.safetensors.index.json";
    out.indexed = std::filesystem::exists(index);
    if (out.indexed) {
        out.files.emplace("model.safetensors.index.json", index);
        const auto root = serve::parse_json(read_bytes(index, 0, std::filesystem::file_size(index)));
        const auto* map = root.find("weight_map");
        if (map == nullptr || !map->is_object()) throw std::runtime_error("ds41 native fp8: no weight_map in " + index.string());
        for (const auto& [name, file] : map->as_object()) {
            out.shard_of.emplace(name, file.as_string());
            out.files.emplace(file.as_string(), dir / file.as_string());
        }
    } else {
        for (const auto& e : std::filesystem::directory_iterator(dir))
            if (e.path().extension() == ".safetensors") {
                out.files.emplace(e.path().filename().string(), e.path());
                out.unindexed_shards.push_back(e.path());
            }
    }
    return out;
}

SafetensorsCheckpoint::SafetensorsCheckpoint(const std::filesystem::path& dir, bool reject_duplicate_names)
    : reject_duplicate_names_(reject_duplicate_names), dir_(dir) {
    auto resolved = resolve_safetensors_checkpoint(dir);
    shard_of_ = std::move(resolved.shard_of);
    if (!resolved.indexed)
        for (const auto& file : resolved.unindexed_shards) load_header(file);
    if (reject_duplicate_names_ && resolved.indexed)
        for (const auto& [name, file] : resolved.files)
            if (name != "model.safetensors.index.json") load_header(file);
    if (shard_of_.empty() && tensors_.empty()) throw std::runtime_error("ds41 native fp8: no tensors in " + dir.string());
}

void SafetensorsCheckpoint::load_header(const std::filesystem::path& file) {
    const auto size = std::filesystem::file_size(file);
    const auto head = read_bytes(file, 0, 8);
    std::uint64_t n = 0;
    for (int k = 7; k >= 0; --k) n = (n << 8U) | static_cast<std::uint8_t>(head[static_cast<std::size_t>(k)]);
    if (n == 0U || n > size - 8U || n > (256ULL << 20U)) throw std::runtime_error("ds41 native fp8: bad safetensors header in " + file.string());
    const auto root = serve::parse_json(read_bytes(file, 8, n));
    if (!root.is_object()) throw std::runtime_error("ds41 native fp8: safetensors header is not an object in " + file.string());
    for (const auto& [name, v] : root.as_object()) {
        if (name == "__metadata__") continue;
        const auto* dtype = v.find("dtype");
        const auto* shape = v.find("shape");
        const auto* offsets = v.find("data_offsets");
        if (dtype == nullptr || shape == nullptr || offsets == nullptr || offsets->size() != 2U)
            throw std::runtime_error("ds41 native fp8: malformed entry " + name + " in " + file.string());
        Tensor t;
        t.file = file;
        t.dtype = dtype->as_string();
        for (const auto& d : shape->as_array()) t.shape.push_back(static_cast<std::uint64_t>(d.as_int()));
        const auto a = static_cast<std::uint64_t>(offsets->as_array()[0].as_int()), b = static_cast<std::uint64_t>(offsets->as_array()[1].as_int());
        if (b < a || 8U + n + b > size) throw std::runtime_error("ds41 native fp8: " + name + " outside " + file.string());
        t.offset = 8U + n + a;
        t.bytes = b - a;
        if (reject_duplicate_names_ && tensors_.contains(name))
            throw std::runtime_error("session cache: duplicate tensor name '" + name + "' across consumed shards");
        tensors_[name] = std::move(t);
        shard_of_.emplace(name, file.filename().string());
    }
    loaded_[file.filename().string()] = true;
}

const SafetensorsCheckpoint::Tensor* SafetensorsCheckpoint::find(const std::string& name) {
    if (const auto it = tensors_.find(name); it != tensors_.end()) return &it->second;
    const auto s = shard_of_.find(name);
    if (s == shard_of_.end()) return nullptr;
    const auto file = dir_ / s->second;
    if (!loaded_[s->second]) {
        if (!std::filesystem::exists(file)) return nullptr;
        load_header(file);
    }
    const auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

void SafetensorsCheckpoint::read(const Tensor& t, std::uint8_t* out) const {
#if defined(_WIN32)
    const auto bytes = read_bytes(t.file, t.offset, t.bytes);
    std::copy(bytes.begin(), bytes.end(), out);
#else
    const int fd = ::open(t.file.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("ds41 native fp8: cannot open " + t.file.string());
    strix::LoaderCapture::record(t.file, fd);
    std::uint64_t done = 0;
    while (done < t.bytes) {
        const auto n = ::pread(fd, out + done, static_cast<std::size_t>(std::min<std::uint64_t>(t.bytes - done, 1ULL << 30U)),
                               static_cast<off_t>(t.offset + done));
        if (n <= 0) {
            ::close(fd);
            throw std::runtime_error("ds41 native fp8: short read of " + t.file.string());
        }
        done += static_cast<std::uint64_t>(n);
    }
    (void)::posix_fadvise(fd, static_cast<off_t>(t.offset), static_cast<off_t>(t.bytes), POSIX_FADV_DONTNEED);
    ::close(fd);
#endif
}

Fp8Tensor read_fp8_tensor(SafetensorsCheckpoint& checkpoint, const std::string& original, std::size_t rows, std::size_t width) {
    const auto* w = checkpoint.find(original + ".weight");
    const auto* s = checkpoint.find(original + ".scale");
    const auto where = " (" + original + " in " + checkpoint.dir().string() + ")";
    if (w == nullptr || s == nullptr) throw std::runtime_error("ds41 native fp8: missing weight or scale" + where);
    if (w->dtype != "F8_E4M3" || s->dtype != "F8_E8M0") throw std::runtime_error("ds41 native fp8: dtypes " + w->dtype + " / " + s->dtype + where);
    if (w->shape != std::vector<std::uint64_t>{rows, width} ||
        s->shape != std::vector<std::uint64_t>{rows / fp8_block, width / fp8_block} || w->bytes != rows * width ||
        s->bytes != rows / fp8_block * (width / fp8_block))
        throw std::runtime_error("ds41 native fp8: shapes differ from the model's tensor (32 x 32 blocks expected)" + where);
    Fp8Tensor out;
    out.rows = rows;
    out.width = width;
    out.bytes.resize(fp8_tensor_bytes(rows, width));
    checkpoint.read(*w, out.bytes.data());
    checkpoint.read(*s, out.bytes.data() + rows * width);
    const auto bad = fp8_tensor_check(std::span<const std::uint8_t>(out.bytes.data(), rows * width),
                                      std::span<const std::uint8_t>(out.bytes.data() + rows * width, s->bytes), rows, width);
    if (!bad.empty()) throw std::runtime_error("ds41 native fp8: " + bad + where);
    return out;
}

}  // namespace strix::ds41
