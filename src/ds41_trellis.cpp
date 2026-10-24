#include "strix/ds41_trellis.hpp"

#include "strix/ds41_native_fp8.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace strix::ds41 {
namespace {

struct TrellisName {
    const char* model;  // suffix of blk.L.<model>
    const char* pack;   // layers.L.<pack>
    const char* site;
};
constexpr std::array<TrellisName, 8> trellis_names{{
    {"attn_q_a.weight", "attn.wq_a", "q_a"},
    {"attn_q_b.weight", "attn.wq_b", "q_b"},
    {"attn_kv.weight", "attn.wkv", "kv"},
    {"attn_output_a.weight", "attn.wo_a", "o_a"},
    {"attn_output_b.weight", "attn.wo_b", "o_b"},
    {"ffn_gate_shexp.weight", "ffn.shared_experts.w1", "shexp"},
    {"ffn_up_shexp.weight", "ffn.shared_experts.w3", "shexp"},
    {"ffn_down_shexp.weight", "ffn.shared_experts.w2", "shexp"},
}};
constexpr std::array<const char*, 7> all_classes{"q_a", "q_b", "kv", "o_a", "o_b", "shexp", "head"};
constexpr std::uint32_t mul1_marker = 0x83DCD12DU;

std::optional<std::pair<std::string, const TrellisName*>> split_name(const std::string& name) {
    static const std::regex re(R"(blk\.(\d+)\.(.+))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    for (const auto& s : trellis_names)
        if (m[2] == s.model) return std::make_pair(m[1].str(), &s);
    return std::nullopt;
}

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("ds41 trellis: " + what); }

}  // namespace

std::optional<std::string> trellis_site_class(const std::string& model_name) {
    if (model_name == "output.weight") return std::string("head");
    const auto s = split_name(model_name);
    if (!s) return std::nullopt;
    return std::string(s->second->site);
}

std::optional<std::string> trellis_pack_name(const std::string& model_name, int group) {
    if (model_name == "output.weight") return std::string("head");
    const auto s = split_name(model_name);
    if (!s) return std::nullopt;
    std::string out = "layers." + s->first + "." + s->second->pack;
    if (group >= 0) out += ".slice." + std::to_string(group);
    return out;
}

std::vector<std::string> parse_trellis_site_classes(const std::string& list) {
    std::vector<std::string> out;
    std::stringstream in(list);
    std::string item;
    while (std::getline(in, item, ',')) {
        if (item.empty()) continue;
        if (item == "all") {
            for (const auto* c : all_classes)
                if (std::find(out.begin(), out.end(), c) == out.end()) out.emplace_back(c);
            continue;
        }
        if (std::find_if(all_classes.begin(), all_classes.end(), [&](const char* a) { return item == a; }) == all_classes.end())
            fail("unknown site class '" + item + "' (q_a, q_b, kv, o_a, o_b, shexp, head, all)");
        if (std::find(out.begin(), out.end(), item) == out.end()) out.push_back(item);
    }
    return out;
}

bool trellis_class_enabled(const std::vector<std::string>& classes, const std::string& model_name) {
    const auto c = trellis_site_class(model_name);
    return c && std::find(classes.begin(), classes.end(), *c) != classes.end();
}

std::string TrellisSlice::describe() const {
    switch (kind) {
    case Kind::none: return "all";
    case Kind::rows: return "rows[" + std::to_string(first) + "+" + std::to_string(count) + "]";
    case Kind::columns: return "cols[" + std::to_string(first) + "+" + std::to_string(count) + "]";
    }
    return "?";
}

std::string trellis_mix_class(const std::string& pack_name) {
    if (pack_name == "head") return "head";
    static const std::regex re(R"(layers\.\d+\.(attn\.(wq_a|wkv|wq_b|wo_a|wo_b)|ffn\.shared_experts\.(w1|w2|w3))(\..*)?)");
    std::smatch m;
    if (!std::regex_match(pack_name, m, re)) fail("no mix class for " + pack_name);
    const std::string a = m[2].str(), f = m[3].str();
    if (a == "wq_a") return "q_a";
    if (a == "wkv") return "kv";
    if (a == "wq_b") return "q_b";
    if (a == "wo_a") return "o_a";
    if (a == "wo_b") return "o_b";
    return f == "w2" ? "shexp_w2" : "shexp_w13";
}

std::string trellis_env_spec() {
    const char* mix = std::getenv("STRIX_DS41_TRELLIS_MIX");
    const char* one = std::getenv("STRIX_DS41_TRELLIS");
    const bool has_one = one != nullptr && *one != 0;
    if (mix == nullptr || *mix == 0) return has_one ? std::string(one) : std::string();
    std::string spec = mix;
    if (spec.find('=') == std::string::npos) {  // a manifest file: one class=dir per line
        std::ifstream in(spec);
        if (!in) fail("STRIX_DS41_TRELLIS_MIX: no manifest " + spec);
        spec.clear();
        for (std::string line; std::getline(in, line);) {
            if (const auto h = line.find('#'); h != std::string::npos) line.resize(h);
            line.erase(0, line.find_first_not_of(" \t\r"));
            line.erase(line.find_last_not_of(" \t\r") + 1U);
            if (line.empty()) continue;
            spec += (spec.empty() ? "" : ",") + line;
        }
    }
    if (has_one && spec.find("default=") == std::string::npos) spec += std::string(",default=") + one;
    return spec;
}

TrellisPack::TrellisPack(const std::filesystem::path& spec_path) : dir_(spec_path) {
    const std::string spec = spec_path.string();
    if (spec.find('=') == std::string::npos) {  // one pack for every class
        if (!std::filesystem::is_directory(spec_path)) fail("pack directory " + spec + " not found");
        checkpoint_ = std::make_shared<SafetensorsCheckpoint>(spec_path);
        description_ = spec;
        default_dir_ = spec_path;
        return;
    }
    static const std::array<const char*, 8> classes{"q_a", "kv", "q_b", "o_a", "o_b", "shexp_w13", "shexp_w2", "head"};
    std::map<std::string, std::shared_ptr<SafetensorsCheckpoint>> by_dir;
    const auto open = [&](const std::string& dir) {
        auto& c = by_dir[dir];
        if (!c) {
            if (!std::filesystem::is_directory(dir)) fail("pack directory " + dir + " not found");
            c = std::make_shared<SafetensorsCheckpoint>(dir);
        }
        return c;
    };
    std::map<std::string, std::string> dirs;
    std::string default_dir;
    std::stringstream in(spec);
    for (std::string item; std::getline(in, item, ',');) {
        if (item.empty()) continue;
        const auto eq = item.find('=');
        if (eq == std::string::npos || eq == 0U || eq + 1U == item.size()) fail("mix item '" + item + "' is not class=dir");
        const std::string cls = item.substr(0, eq), dir = item.substr(eq + 1U);
        if (cls == "default") {
            default_dir = dir;
            checkpoint_ = open(dir);
            continue;
        }
        std::vector<std::string> targets;
        if (cls == "shexp") targets = {"shexp_w13", "shexp_w2"};
        else if (std::find_if(classes.begin(), classes.end(), [&](const char* c) { return cls == c; }) != classes.end()) targets = {cls};
        else fail("unknown mix class '" + cls + "' (q_a, kv, q_b, o_a, o_b, shexp, shexp_w13, shexp_w2, head, default)");
        for (const auto& t : targets) {
            if (dirs.contains(t)) fail("mix class " + t + " given twice");
            dirs[t] = dir;
            dirs_[t] = dir;
            by_class_[t] = open(dir);
        }
    }
    if (by_class_.empty() && !checkpoint_) fail("empty mix spec");
    default_dir_ = default_dir;
    for (const auto& [c, d] : dirs) description_ += (description_.empty() ? "" : ",") + c + "=" + d;
    if (!default_dir.empty()) description_ += (description_.empty() ? "" : ",") + std::string("default=") + default_dir;
}

SafetensorsCheckpoint& TrellisPack::checkpoint(const std::string& pack_name) {
    if (!by_class_.empty()) {
        const auto it = by_class_.find(trellis_mix_class(pack_name));
        if (it != by_class_.end()) return *it->second;
    }
    if (!checkpoint_) fail("the mix " + description_ + " has no pack for " + pack_name + " (class " + trellis_mix_class(pack_name) + ")");
    return *checkpoint_;
}

TrellisPack::~TrellisPack() = default;

std::filesystem::path TrellisPack::class_dir(const std::string& mix_class) const {
    if (const auto it = dirs_.find(mix_class); it != dirs_.end()) return it->second;
    return default_dir_;
}

std::optional<TrellisTensorInfo> TrellisPack::info(const std::string& pack_name) {
    const auto* t = checkpoint(pack_name).find(pack_name + ".trellis");
    if (t == nullptr) return std::nullopt;
    if (t->dtype != "I16" || t->shape.size() != 3U || t->shape[2] % 16U != 0U) fail(pack_name + ".trellis: not I16 [I/16][N/16][16K]");
    TrellisTensorInfo info;
    info.name = pack_name;
    info.columns = static_cast<std::uint32_t>(t->shape[0] * 16U);
    info.rows = static_cast<std::uint32_t>(t->shape[1] * 16U);
    info.bits = static_cast<std::uint32_t>(t->shape[2] / 16U);
    if (info.bits < 1U || info.bits > 8U || info.rows % 128U != 0U || info.columns % 128U != 0U ||
        t->bytes != std::uint64_t{info.rows} * info.columns * info.bits / 8U)
        fail(pack_name + ".trellis: bad geometry");
    return info;
}

TrellisHostBlob TrellisPack::load(const std::string& pack_name, const TrellisSlice& slice) {
    const auto inf = info(pack_name);
    if (!inf) fail("no linear " + pack_name + " in " + dir_.string());
    auto& ck = checkpoint(pack_name);
    const auto* tq = ck.find(pack_name + ".trellis");
    const auto* tu = ck.find(pack_name + ".suh");
    const auto* tv = ck.find(pack_name + ".svh");
    const auto* tm = ck.find(pack_name + ".mul1");
    if (tu == nullptr || tv == nullptr || tm == nullptr) fail(pack_name + ": suh, svh or mul1 missing");
    if (tu->dtype != "F16" || tu->bytes != std::uint64_t{inf->columns} * 2U || tv->dtype != "F16" || tv->bytes != std::uint64_t{inf->rows} * 2U ||
        tm->dtype != "I32" || tm->bytes != 4U)
        fail(pack_name + ": suh / svh / mul1 of the wrong type or size");
    std::uint32_t marker = 0;
    {
        std::array<std::uint8_t, 4> b{};
        ck.read(*tm, b.data());
        marker = static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8U) | (static_cast<std::uint32_t>(b[2]) << 16U) |
                 (static_cast<std::uint32_t>(b[3]) << 24U);
    }
    if (marker != mul1_marker) fail(pack_name + ": codebook marker is not mul1");
    std::uint32_t r0 = 0, nr = inf->rows, c0 = 0, nc = inf->columns;
    if (slice.kind == TrellisSlice::Kind::rows) {
        r0 = slice.first;
        nr = slice.count;
    } else if (slice.kind == TrellisSlice::Kind::columns) {
        c0 = slice.first;
        nc = slice.count;
    }
    if (nr == 0U || nc == 0U || r0 % 128U != 0U || nr % 128U != 0U || c0 % 128U != 0U || nc % 128U != 0U || r0 + nr > inf->rows ||
        c0 + nc > inf->columns)
        fail(pack_name + ": slice " + slice.describe() + " not 128-aligned inside " + std::to_string(inf->rows) + " x " +
             std::to_string(inf->columns));
    std::vector<std::uint8_t> q(tq->bytes), u(tu->bytes), v(tv->bytes);
    ck.read(*tq, q.data());
    ck.read(*tu, u.data());
    ck.read(*tv, v.data());
    TrellisHostBlob out;
    out.name = pack_name;
    out.rows = nr;
    out.columns = nc;
    out.bits = inf->bits;
    out.marker = marker;
    const std::size_t tile = std::size_t{16U} * inf->bits * 2U;  // bytes of one 16 x 16 tile
    const std::size_t tiles_n = inf->rows / 16U;
    out.trellis_offset = 0;
    out.trellis_bytes = std::uint64_t{nr} * nc * inf->bits / 8U;
    out.blob.reserve(out.trellis_bytes + std::size_t{nc} * 2U + std::size_t{nr} * 2U);
    for (std::size_t ti = c0 / 16U; ti < (c0 + nc) / 16U; ++ti) {
        const auto* src = q.data() + (ti * tiles_n + r0 / 16U) * tile;
        out.blob.insert(out.blob.end(), src, src + std::size_t{nr / 16U} * tile);
    }
    if (out.blob.size() != out.trellis_bytes) fail(pack_name + ": slice size mismatch");
    out.suh_offset = out.blob.size();
    out.suh_bytes = std::uint64_t{nc} * 2U;
    out.blob.insert(out.blob.end(), u.data() + std::size_t{c0} * 2U, u.data() + std::size_t{c0 + nc} * 2U);
    out.svh_offset = out.blob.size();
    out.svh_bytes = std::uint64_t{nr} * 2U;
    out.blob.insert(out.blob.end(), v.data() + std::size_t{r0} * 2U, v.data() + std::size_t{r0 + nr} * 2U);
    return out;
}

std::uint64_t trellis_site_bytes(const TrellisTensorInfo& info, const TrellisSlice& slice) {
    std::uint64_t rows = info.rows, columns = info.columns;
    if (slice.kind == TrellisSlice::Kind::rows) rows = slice.count;
    if (slice.kind == TrellisSlice::Kind::columns) columns = slice.count;
    return rows * columns * info.bits / 8U + 2U * (rows + columns);
}

namespace {

// Site classes with an engine hook (q_a and kv keep today's form) and the mix classes each one reads.
constexpr std::array<std::pair<const char*, std::array<const char*, 2>>, 5> hooked_classes{{
    {"q_b", {"q_b", nullptr}},
    {"o_a", {"o_a", nullptr}},
    {"o_b", {"o_b", nullptr}},
    {"shexp", {"shexp_w13", "shexp_w2"}},
    {"head", {"head", nullptr}},
}};

std::string representative_pack_name(const std::string& mix_class, int layer) {
    const std::string l = "layers." + std::to_string(layer) + ".";
    if (mix_class == "head") return "head";
    if (mix_class == "q_a") return l + "attn.wq_a";
    if (mix_class == "kv") return l + "attn.wkv";
    if (mix_class == "q_b") return l + "attn.wq_b";
    if (mix_class == "o_a") return l + "attn.wo_a.slice.0";
    if (mix_class == "o_b") return l + "attn.wo_b";
    if (mix_class == "shexp_w13") return l + "ffn.shared_experts.w1";
    if (mix_class == "shexp_w2") return l + "ffn.shared_experts.w2";
    fail("unknown mix class " + mix_class);
}

// The enabled site classes that have an engine hook, in the fixed order above.
std::vector<std::string> hooked(const std::vector<std::string>& site_classes) {
    std::vector<std::string> out;
    for (const auto& [site, mix] : hooked_classes)
        if (std::find(site_classes.begin(), site_classes.end(), site) != site_classes.end()) out.emplace_back(site);
    return out;
}

std::uint64_t fnv1a(const std::string& s, std::uint64_t h = 14695981039346656037ULL) {
    for (const unsigned char c : s) h = (h ^ c) * 1099511628211ULL;
    return h;
}

std::string hex16(std::uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
}

// A pack directory's fingerprint: its SHA256 manifest (<dir>.sha256 next to it) when present, else its file names and sizes.
std::string dir_fingerprint(const std::filesystem::path& dir) {
    namespace fs = std::filesystem;
    fs::path d = dir;
    if (!d.has_filename()) d = d.parent_path();  // "dir/" -> "dir"
    const fs::path manifest = d.parent_path() / (d.filename().string() + ".sha256");
    if (std::ifstream in(manifest, std::ios::binary); in) {
        std::stringstream ss;
        ss << in.rdbuf();
        return "sha256-manifest:" + hex16(fnv1a(ss.str()));
    }
    std::vector<std::pair<std::string, std::uint64_t>> files;
    for (const auto& e : fs::directory_iterator(d))
        if (e.is_regular_file()) files.emplace_back(e.path().filename().string(), e.file_size());
    std::sort(files.begin(), files.end());
    std::uint64_t h = 14695981039346656037ULL;
    for (const auto& [n, b] : files) h = fnv1a(n + ":" + std::to_string(b) + ";", h);
    return "files:" + hex16(h);
}

}  // namespace

std::string trellis_step_identity(std::uint32_t bits, const std::string& site) {
    return "ds41-trellis-k" + std::to_string(bits) + "-v1:" + site;
}

std::string trellis_shexp_step_identity(std::uint32_t w13_bits, std::uint32_t w2_bits, bool shexp_tp) {
    const std::string site = shexp_tp ? "shexp-tp" : "shexp";
    if (w13_bits == w2_bits) return trellis_step_identity(w13_bits, site);
    return "ds41-trellis-mix-v1{shexp_w13=K" + std::to_string(w13_bits) + ",shexp_w2=K" + std::to_string(w2_bits) + "}:" + site;
}

std::string trellis_config_identity(const std::map<std::string, std::uint32_t>& bits_by_mix_class,
                                    const std::vector<std::string>& site_classes, bool shexp_tp) {
    const auto sites = hooked(site_classes);
    if (sites.empty()) return {};
    std::map<std::string, std::uint32_t> used;  // mix class -> K over the enabled sites
    for (const auto& site : sites)
        for (const auto& [cls, mix] : hooked_classes)
            if (site == cls)
                for (const char* m : mix)
                    if (m != nullptr) {
                        const auto it = bits_by_mix_class.find(m);
                        if (it == bits_by_mix_class.end()) fail(std::string("no K for mix class ") + m);
                        used[m] = it->second;
                    }
    std::string site_list;
    for (const auto& site : sites) site_list += (site_list.empty() ? "" : ",") + site;
    std::set<std::uint32_t> ks;
    for (const auto& [m, k] : used) ks.insert(k);
    std::string out;
    if (ks.size() == 1U) {
        out = "ds41-trellis-k" + std::to_string(*ks.begin()) + "-v1:" + site_list;
    } else {
        std::string per;
        for (const auto& [m, k] : used) per += (per.empty() ? "" : ",") + m + "=K" + std::to_string(k);
        out = "ds41-trellis-mix-v1{" + per + "}:" + site_list;
        // the declared mixes (docs/models/deepseek-v4.1/trellis-k5-v1.md): every used class must carry the K they assign
        const std::map<std::string, std::uint32_t> mix_a{{"q_b", 3}, {"o_a", 3}, {"o_b", 3}, {"shexp_w2", 3}, {"shexp_w13", 4},
                                                         {"q_a", 4}, {"kv", 4}, {"head", 5}};
        const auto matches = [&](const std::map<std::string, std::uint32_t>& decl, std::uint32_t dense) {
            for (const auto& [m, k] : used) {
                const auto it = decl.find(m);
                if ((it != decl.end() ? it->second : dense) != k) return false;
            }
            return true;
        };
        if (matches(mix_a, 0U)) out += " (= ds41-trellis-mixA-v1)";
        else if (matches({{"head", 5}}, 4U)) out += " (= ds41-trellis-mixB-v1)";
    }
    if (shexp_tp && std::find(sites.begin(), sites.end(), "shexp") != sites.end()) out += ":shexp-tp";
    return out;
}

std::map<std::string, std::uint32_t> trellis_class_bits(TrellisPack& pack, const std::vector<std::string>& site_classes, int layer) {
    std::map<std::string, std::uint32_t> out;
    for (const auto& site : hooked(site_classes))
        for (const auto& [cls, mix] : hooked_classes)
            if (site == cls)
                for (const char* m : mix)
                    if (m != nullptr) {
                        const auto name = representative_pack_name(m, layer);
                        const auto info = pack.info(name);
                        if (!info) fail("the pack " + pack.describe() + " has no " + name + " (class " + m + ")");
                        out[m] = info->bits;
                    }
    return out;
}

std::string trellis_weights_namespace() {
    const auto spec = trellis_env_spec();
    if (spec.empty()) return {};
    const char* sites_env = std::getenv("STRIX_DS41_TRELLIS_SITES");
    const auto classes = parse_trellis_site_classes(sites_env != nullptr ? sites_env : "");
    const auto sites = hooked(classes);
    if (sites.empty()) return {};
    const char* tp = std::getenv("STRIX_DS41_SHEXP_TP");
    TrellisPack pack(spec);
    const auto bits = trellis_class_bits(pack, classes);
    std::string out = "trellis " + trellis_config_identity(bits, classes, tp != nullptr && *tp == '1') + "; packs ";
    bool first = true;
    for (const auto& [m, k] : bits) {
        const auto dir = pack.class_dir(m);
        out += std::string(first ? "" : ",") + m + "=" + dir.string() + "#" + dir_fingerprint(dir);
        first = false;
    }
    return out;
}

}  // namespace strix::ds41
