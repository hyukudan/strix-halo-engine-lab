#include "strix/asset_manifest.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/model_manifest.hpp"
#include "strix/mtp_draft_tensor_audit.hpp"
#include "strix/serve/json.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <map>
#include <set>

namespace strix {
#ifdef STRIX_ASSET_TEST_HOOKS
void asset_test_after_hash();
#endif
namespace {
using serve::Json;
constexpr std::size_t cap = 1024 * 1024;
[[noreturn]] void invalid() { throw std::invalid_argument("invalid asset schema"); }
const Json& field(const Json& j, const char* key) {
    const auto* p = j.find(key);
    if (!p) invalid();
    return *p;
}
std::string string(const Json& j, const char* key) { return field(j, key).as_string(); }
bool token(std::string_view s) {
    return !s.empty() && s.size() <= 128 && std::all_of(s.begin(), s.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '+';
    });
}
void hash_check(std::string_view s) {
    if (s.size() != 64 || !std::all_of(s.begin(), s.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    })) invalid();
}
std::uint64_t integer(const Json& j) {
    const auto& s = j.number_source();
    std::uint64_t n = 0;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), n);
    if (r.ec != std::errc{} || r.ptr != s.data() + s.size() || s.empty() || s[0] == '-') invalid();
    return n;
}
Json canonical(Json j) {
    if (j.is_object()) {
        for (auto& [k, v] : j.as_object()) { (void)k; v = canonical(v); }
        std::sort(j.as_object().begin(), j.as_object().end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    } else if (j.is_array()) {
        for (auto& v : j.as_array()) v = canonical(v);
    } else if (j.is_number()) j = Json(integer(j));
    return j;
}
std::string digest(std::string_view s) {
    return model_content_identity_hex(sha256_bytes(std::as_bytes(std::span(s.data(), s.size()))));
}
std::string lower(std::string s) {
    for (auto& c : s) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
    return s;
}
bool public_name(std::string_view s) {
    return s == "target" || s == "target_shard" || s == "draft" || s == "vocabulary" ||
           s == "tokenizer" || s == "profile" || s == "calibration" || s == "distill-g";
}
void member_check(std::string_view s) {
    if (s.empty() || s.size() > 512 || s.front() == '/' || s.find('\\') != s.npos || s.find(':') != s.npos) invalid();
    for (const auto& part : std::filesystem::path(s)) {
        const auto p = part.string();
        if (p == "." || p == ".." || !token(p) || p.back() == '.' || p.back() == ' ') invalid();
        const auto stem = lower(p.substr(0, p.find('.')));
        if (stem == "con" || stem == "prn" || stem == "aux" || stem == "nul" ||
            (stem.size() == 4 && (stem.starts_with("com") || stem.starts_with("lpt")) &&
             stem[3] >= '1' && stem[3] <= '9')) invalid();
    }
    if (s.back() == '/' || s.find("//") != s.npos) invalid();
}
bool role_valid(std::string_view role) {
    return role == "target" || role == "target_shard" || role == "draft" ||
           role == "vocabulary" || role == "tokenizer" || role == "profile" || role == "calibration";
}
void keys(const Json& j, std::initializer_list<std::string_view> allowed) {
    for (const auto& [k, v] : j.as_object()) {
        (void)v;
        if (std::find(allowed.begin(), allowed.end(), k) == allowed.end()) invalid();
    }
}
// Conventional root notice files: LICENSE, NOTICE, COPYING or README, optionally
// with a short suffix such as .txt, .md or -MIT (member_check already ran).
bool root_notice(std::string_view v) {
    for (const std::string_view stem : {"LICENSE", "NOTICE", "COPYING", "README"}) {
        if (!v.starts_with(stem)) continue;
        const auto rest = v.substr(stem.size());
        if (rest.empty()) return true;
        if (rest.size() > 32 || (rest.front() != '.' && rest.front() != '-')) return false;
        return std::all_of(rest.begin(), rest.end(), [](char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '-';
        });
    }
    return false;
}
Json parse(std::string_view text, bool manifest) {
    if (text.size() > cap) invalid();
    auto j = serve::parse_json_unique(text);
    keys(j, {"format", "revision", "engine", "model_key", "profile_pair", "required", "files", "provenance", "licenses"});
    if (string(j, "format") != "strix-assets" || integer(field(j, "revision")) != 1) invalid();
    for (const auto* k : {"engine", "model_key", "profile_pair"}) if (!token(string(j, k))) invalid();
    const auto engine = string(j, "engine");
    if (engine != "qwen" && engine != "deepseek41" && engine != "minimax" && engine != "glm") invalid();
    for (const auto* k : {"provenance", "licenses"}) {
        for (const auto& v : field(j, k).as_array()) {
            keys(v, {"type", "value"});
            const auto type = string(v, "type"), value = string(v, "value");
            if (value.empty() || value.size() > 512) invalid();
            if (type == "member") {
                member_check(value);
                if (value.find('/') == value.npos && !root_notice(value)) invalid();
            } else if (type == "public-id") {
                if (value != "Qwen-Community-License" && value != "Apache-2.0" && value != "MIT" &&
                    value != "BSD-3-Clause" && value != "CC-BY-4.0") invalid();
            } else if (type == "https") {
                // Public-source hosts only; no credentials, ports, query, fragment or escapes.
                if (!value.starts_with("https://")) invalid();
                const auto slash = value.find('/', 8);
                if (slash == value.npos || slash + 1 == value.size()) invalid();
                const auto host = value.substr(8, slash - 8);
                if (host != "huggingface.co" && host != "github.com" && host != "www.apache.org" &&
                    host != "spdx.org" && host != "creativecommons.org") invalid();
                // Public source paths may contain Unicode scalars (already strict UTF-8).
                for (const unsigned char c : value.substr(slash + 1)) {
                    if (c < 0x80U && !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '/')) invalid();
                }
                if (value.find("/../") != value.npos || value.ends_with("/..")) invalid();
            } else invalid();
        }
    }
    std::set<std::pair<std::string, std::string>> required, actual;
    for (auto& [role, list] : j.set("required", field(j, "required")).as_object()) {
        if (!role_valid(role) || list.as_array().empty()) invalid();
        for (const auto& key : list.as_array()) {
            if (!token(key.as_string()) || !required.emplace(role, key.as_string()).second) invalid();
        }
        std::sort(list.as_array().begin(), list.as_array().end(), [](const auto& a, const auto& b) { return a.as_string() < b.as_string(); });
    }
    if (required.empty() || required.size() > 1024) invalid();
    std::set<std::string> members;
    auto& files = j.set("files", field(j, "files")).as_array();
    for (const auto& f : files) {
        keys(f, {"role", "key", "member", "source_revision", "bytes", "sha256", "expected_sha256", "content_key", "content_identity", "audit"});
        const auto role = string(f, "role"), key = string(f, "key"), member = string(f, "member");
        if (!role_valid(role) || !token(key) || !token(string(f, "source_revision"))) invalid();
        member_check(member);
        if (!actual.emplace(role, key).second || !members.insert(lower(member)).second) invalid();
        if (f.find("bytes")) (void)integer(field(f, "bytes"));
        if (manifest && (!f.find("sha256") || !f.find("bytes"))) invalid();
        for (const auto* k : {"sha256", "expected_sha256", "content_identity"}) if (f.find(k)) hash_check(string(f, k));
        if (f.find("content_key") && !token(string(f, "content_key"))) invalid();
        if (f.find("content_identity") && !f.find("content_key")) invalid();
        if (manifest && f.find("content_key") && !f.find("content_identity")) invalid();
        if (f.find("audit") && string(f, "audit") != "gguf" && string(f, "audit") != "mtp") invalid();
    }
    if (required != actual) {
        if (manifest) throw AssetMismatch("asset role closure mismatch");
        invalid();
    }
    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) {
        return std::make_pair(string(a, "role"), string(a, "key")) < std::make_pair(string(b, "role"), string(b, "key"));
    });
    return canonical(j);
}
std::filesystem::path resolve(const std::filesystem::path& root, const Json& f) {
    const auto base = std::filesystem::canonical(root);
    const auto path = std::filesystem::weakly_canonical(base / string(f, "member"));
    const auto relative = path.lexically_relative(base);
    if (relative.empty() || *relative.begin() == ".." || relative.is_absolute()) invalid();
    std::error_code ec;
    const bool regular = std::filesystem::is_regular_file(path, ec);
    if (ec && ec != std::errc::no_such_file_or_directory) throw std::runtime_error("asset status I/O failure");
    if (!regular) throw AssetMismatch("asset missing or not regular");
    return path;
}
void measure(Json& f, const std::filesystem::path& root) {
    const auto path = resolve(root, f);
    const auto size = std::filesystem::file_size(path);
    const auto time = std::filesystem::last_write_time(path);
    if (f.find("bytes") && integer(field(f, "bytes")) != size) throw AssetMismatch("asset size mismatch");
    std::string sha, content;
    try {
        if (f.find("content_key")) {
            const auto result = hash_single_model_content_file({string(f, "content_key"), path}, FileHashPolicy::full);
            sha = model_content_identity_hex(result.file_sha256);
            content = model_content_identity_hex(result.content_identity);
        } else sha = model_content_identity_hex(sha256_file_content(path, FileHashPolicy::full));
    } catch (const ModelContentChanged&) { throw AssetMismatch("asset changed while hashing"); }
#ifdef STRIX_ASSET_TEST_HOOKS
    asset_test_after_hash();
#endif
    if (f.find("audit")) {
        try {
            const auto m = build_model_manifest(gguf::read_file(path));
            if (string(f, "audit") == "mtp" && !audit_mtp_draft_tensor_families(m).structurally_complete())
                throw AssetMismatch("asset structural mismatch");
        } catch (const std::exception&) { throw AssetMismatch("asset structural mismatch"); }
    }
    std::error_code ec;
    const auto after_size = std::filesystem::file_size(path, ec);
    if (ec || size != after_size) throw AssetMismatch("asset changed while hashing");
    const auto after_time = std::filesystem::last_write_time(path, ec);
    if (ec || time != after_time || path != resolve(root, f)) throw AssetMismatch("asset changed while hashing");
    for (const auto* k : {"sha256", "expected_sha256"}) {
        if (f.find(k) && string(f, k) != sha) throw AssetMismatch("asset hash mismatch");
    }
    if (f.find("content_identity") && string(f, "content_identity") != content)
        throw AssetMismatch("asset content identity mismatch");
    f.set("bytes", Json(static_cast<std::uint64_t>(size)));
    f.set("sha256", sha);
    if (!content.empty()) f.set("content_identity", content);
}
void measure_named(Json& f, const std::filesystem::path& root, std::size_t index,
                   std::vector<std::filesystem::path>* measured = nullptr) {
    try {
        if (measured) {
            const auto path = resolve(root, f);
            for (const auto& previous : *measured) if (std::filesystem::equivalent(previous, path)) invalid();
            measured->push_back(path);
        } else measure(f, root);
    } catch (AssetMismatch& e) {
        e.role = string(f, "role");
        const auto key = string(f, "key");
        if (public_name(key)) e.key = key;
        else e.key_digest = digest(key).substr(0, 16);
        e.index = index;
        throw;
    }
}
}  // namespace

std::string read_asset_json(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("asset JSON read failed");
    std::string text(cap + 1, '\0');
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<std::size_t>(in.gcount()));
    if (in.bad() || text.size() > cap) throw std::runtime_error("asset JSON unreadable or oversized");
    return text;
}
std::string create_asset_manifest(std::string_view spec, const std::filesystem::path& root) {
    auto j = parse(spec, false);
    std::vector<std::filesystem::path> measured;
    std::size_t index = 0;
    auto& files = j.set("files", field(j, "files")).as_array();
    for (auto& f : files) measure_named(f, root, index++, &measured);
    index = 0;
    for (auto& f : files) measure_named(f, root, index++);
    const auto out = canonical(j).dump();
    if (out.size() > cap) invalid();
    return out;
}
std::string asset_manifest_digest(std::string_view manifest) { return digest(parse(manifest, true).dump()); }
std::string verify_asset_manifest(std::string_view manifest, const std::filesystem::path& root,
                                  const std::vector<std::string>& roles) {
    auto j = parse(manifest, true);
    std::set<std::string> selected(roles.begin(), roles.end());
    for (const auto& r : selected) if (!field(j, "required").find(r)) invalid();
    const auto identity = digest(j.dump());
    std::map<std::string, Json> groups;
    Json checked = Json::array(), unchecked = Json::array();
    for (const auto& [role, list] : field(j, "required").as_object()) {
        (void)list;
        (selected.empty() || selected.contains(role) ? checked : unchecked).push(Json(role));
    }
    std::vector<std::filesystem::path> measured;
    std::size_t index = 0;
    auto& files = j.set("files", field(j, "files")).as_array();
    for (auto& f : files) {
        const auto entry_index = index++;
        if (selected.empty() || selected.contains(string(f, "role"))) measure_named(f, root, entry_index, &measured);
    }
    index = 0;
    for (auto& f : files) {
        const auto role = string(f, "role");
        const auto entry_index = index++;
        if (!selected.empty() && !selected.contains(role)) continue;
        measure_named(f, root, entry_index);
        if (!groups.contains(role)) groups.emplace(role, Json::array());
        groups.at(role).push(Json(Json::Array{field(f, "key"), field(f, "bytes"), field(f, "sha256")}));
    }
    Json counts = Json::object();
    for (const auto& [role, rows] : groups) {
        counts.set(role, Json(Json::Object{{"count", Json(static_cast<std::uint64_t>(rows.size()))}, {"digest", digest(rows.dump())}}));
    }
    return canonical(Json(Json::Object{{"manifest_digest", identity}, {"roles", counts}, {"hash_mode", "full"},
        {"checked_roles", checked}, {"unchecked_roles", unchecked},
        {"verification", unchecked.as_array().empty() ? "full" : "partial"}})).dump();
}
}  // namespace strix
