// GLM PackReader input validation (host-only, Linux): malformed model directories give a clean error, never an
// out-of-directory open, an out-of-bounds access or an unchecked offset; valid packs read exactly as before.
// Optional: GLM_PACK_DIR=<real pack> also checks every tensor of a real pack against a plain header walk.
#include "glm/engine/glm_pack.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
using strix::glm::engine::detail::PackReader;

namespace {

int checks = 0, failures = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        ++checks;                                                             \
        if (!(c)) {                                                           \
            ++failures;                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
        }                                                                     \
    } while (0)

void write_bytes(const fs::path& p, const std::string& s) {
    std::ofstream o(p, std::ios::binary | std::ios::trunc);
    o.write(s.data(), static_cast<std::streamsize>(s.size()));
}

// One shard: 8-byte little-endian header length, the header JSON, then `payload`.
void write_shard(const fs::path& p, const std::string& header, const std::string& payload, std::uint64_t declared = 0) {
    std::string out(8, '\0');
    const std::uint64_t n = declared ? declared : header.size();
    for (int i = 0; i < 8; ++i) out[static_cast<std::size_t>(i)] = static_cast<char>((n >> (8 * i)) & 0xFFU);
    write_bytes(p, out + header + payload);
}

void write_index(const fs::path& dir, const std::string& weight_map) {
    write_bytes(dir / "model.safetensors.index.json", "{\"metadata\":{},\"weight_map\":{" + weight_map + "}}");
}

std::string f32_bytes(const std::vector<float>& v) {
    std::string s(v.size() * 4, '\0');
    std::memcpy(s.data(), v.data(), s.size());
    return s;
}

// Expect the constructor (or `use`) to throw a std::exception; a crash or a silent success fails the case.
bool refused(const fs::path& dir, const std::function<void(PackReader&)>& use = {}) {
    try {
        PackReader pr(dir.string());
        if (use) use(pr);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  refused: %s\n", e.what());
        return true;
    }
    return false;
}

struct Case {
    const char* name;
    std::function<void(const fs::path& root)> run;
};

fs::path fresh(const fs::path& root, const char* name) {
    const fs::path d = root / name / "model";
    fs::remove_all(root / name);
    fs::create_directories(d);
    return d;
}

const std::string kOk = "{\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";

std::vector<Case> cases() {
    return {
        {"shard_not_string", [](const fs::path& root) {
             const fs::path d = fresh(root, "shard_not_string");
             write_index(d, "\"w\":7");
             CHECK(refused(d));
         }},
        {"index_without_weight_map", [](const fs::path& root) {
             const fs::path d = fresh(root, "index_without_weight_map");
             write_bytes(d / "model.safetensors.index.json", "[1,2]");
             CHECK(refused(d));
         }},
        {"one_data_offset", [](const fs::path& root) {
             const fs::path d = fresh(root, "one_data_offset");
             write_shard(d / "a.safetensors", "{\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0]}}", f32_bytes({1.0F, 2.0F}));
             write_index(d, "\"w\":\"a.safetensors\"");
             CHECK(refused(d));
         }},
        {"empty_data_offsets", [](const fs::path& root) {
             const fs::path d = fresh(root, "empty_data_offsets");
             write_shard(d / "a.safetensors", "{\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[]}}", f32_bytes({1.0F, 2.0F}));
             write_index(d, "\"w\":\"a.safetensors\"");
             CHECK(refused(d));
         }},
        {"reversed_data_offsets", [](const fs::path& root) {
             const fs::path d = fresh(root, "reversed_data_offsets");
             write_shard(d / "a.safetensors", "{\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[8,0]}}", f32_bytes({1.0F, 2.0F}));
             write_index(d, "\"w\":\"a.safetensors\"");
             CHECK(refused(d));
         }},
        {"negative_data_offset", [](const fs::path& root) {
             const fs::path d = fresh(root, "negative_data_offset");
             write_shard(d / "a.safetensors", "{\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[-8,0]}}", f32_bytes({1.0F, 2.0F}));
             write_index(d, "\"w\":\"a.safetensors\"");
             CHECK(refused(d));
         }},
        {"data_past_file_end", [](const fs::path& root) {
             const fs::path d = fresh(root, "data_past_file_end");
             write_shard(d / "a.safetensors", "{\"w\":{\"dtype\":\"U8\",\"shape\":[1048576],\"data_offsets\":[0,1048576]}}", "xyz");
             write_index(d, "\"w\":\"a.safetensors\"");
             CHECK(refused(d));
         }},
        {"f32_partial_element", [](const fs::path& root) {
             // 6 bytes of F32: the old floats() copied 6 bytes into a 4-byte vector.
             const fs::path d = fresh(root, "f32_partial_element");
             write_shard(d / "a.safetensors", "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,6]}}", "abcdef");
             write_index(d, "\"w\":\"a.safetensors\"");
             CHECK(refused(d, [](PackReader& pr) { (void)pr.floats("w"); }));
         }},
        {"header_length_past_file", [](const fs::path& root) {
             const fs::path d = fresh(root, "header_length_past_file");
             write_shard(d / "a.safetensors", kOk, f32_bytes({1.0F, 2.0F}), (std::uint64_t{1} << 30) - 1);
             write_index(d, "\"w\":\"a.safetensors\"");
             CHECK(refused(d));
         }},
        {"header_not_json", [](const fs::path& root) {
             const fs::path d = fresh(root, "header_not_json");
             write_shard(d / "a.safetensors", "{\"w\":{\"dtype\":", "");
             write_index(d, "\"w\":\"a.safetensors\"");
             CHECK(refused(d));
         }},
        {"read_offset_wraps", [](const fs::path& root) {
             const fs::path d = fresh(root, "read_offset_wraps");
             write_shard(d / "a.safetensors", kOk, f32_bytes({1.0F, 2.0F}));
             write_index(d, "\"w\":\"a.safetensors\"");
             CHECK(refused(d, [](PackReader& pr) {
                 char buf[8];
                 pr.read(pr.info("w"), UINT64_MAX - 3, 8, buf);
             }));
         }},
        {"valid_pack", [](const fs::path& root) {
             // Two listed shards + one unlisted shard (the GLM pack ships the MTP layer outside the index).
             const fs::path d = fresh(root, "valid_pack");
             const std::uint16_t bf[3] = {0x3F80, 0xC000, 0x4040};  // 1, -2, 3
             std::string bfs(6, '\0');
             std::memcpy(bfs.data(), bf, 6);
             write_shard(d / "model-00001-of-00002.safetensors",
                         "{\"__metadata__\":{\"format\":\"pt\"},\"a\":{\"dtype\":\"BF16\",\"shape\":[3],\"data_offsets\":[0,6]},"
                         "\"b\":{\"dtype\":\"F32\",\"shape\":[1,2],\"data_offsets\":[6,14]}}",
                         bfs + f32_bytes({0.5F, -0.25F}));
             write_shard(d / "model-00002-of-00002.safetensors", "{\"c\":{\"dtype\":\"U8\",\"shape\":[5],\"data_offsets\":[0,5]}}", "hello");
             write_shard(d / "mtp.safetensors", "{\"m\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", f32_bytes({7.0F}));
             write_index(d, "\"a\":\"model-00001-of-00002.safetensors\",\"b\":\"model-00001-of-00002.safetensors\","
                            "\"c\":\"model-00002-of-00002.safetensors\"");
             try {
                 PackReader pr(d.string());
                 CHECK(pr.files() == 3);
                 const auto& a = pr.info("a");
                 CHECK(a.dtype == "BF16" && a.shape == std::vector<std::uint64_t>{3} && a.bytes == 6);
                 const auto& b = pr.info("b");
                 CHECK(b.dtype == "F32" && b.shape == (std::vector<std::uint64_t>{1, 2}) && b.bytes == 8 && b.offset == a.offset + 6);
                 CHECK(pr.floats("a") == (std::vector<float>{1.0F, -2.0F, 3.0F}));
                 CHECK(pr.floats("b") == (std::vector<float>{0.5F, -0.25F}));
                 CHECK(pr.floats("m") == std::vector<float>{7.0F});
                 char c[5];
                 pr.read(pr.info("c"), 1, 4, c);
                 CHECK(std::memcmp(c, "ello", 4) == 0);
                 bool past = false;
                 try {
                     pr.read(pr.info("c"), 2, 4, c);
                 } catch (const std::runtime_error&) {
                     past = true;
                 }
                 CHECK(past);
             } catch (const std::exception& e) {
                 std::fprintf(stderr, "valid pack refused: %s\n", e.what());
                 CHECK(false);
             }
         }},
    };
}

// Plain header walk of a real pack (the previous reader's interpretation): every tensor must read identically.
void real_pack(const fs::path& dir) {
    PackReader pr(dir.string());
    std::size_t n = 0;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file() || e.path().extension() != ".safetensors") continue;
        std::ifstream f(e.path(), std::ios::binary);
        unsigned char len[8] = {};
        f.read(reinterpret_cast<char*>(len), 8);
        std::uint64_t hb = 0;
        for (int i = 0; i < 8; ++i) hb |= std::uint64_t{len[i]} << (8 * i);
        std::string head(hb, '\0');
        f.read(head.data(), static_cast<std::streamsize>(hb));
        const auto j = strix::glm::engine::detail::parse_json(head.data(), head.data() + head.size());
        for (const auto& [name, v] : j.obj()) {
            if (name == "__metadata__") continue;
            const auto& off = v.obj().at("data_offsets").arr();
            const auto& t = pr.info(name);
            const auto b = static_cast<std::uint64_t>(off[0].num()), x = static_cast<std::uint64_t>(off[1].num());
            std::vector<std::uint64_t> shape;
            for (const auto& s : v.obj().at("shape").arr()) shape.push_back(static_cast<std::uint64_t>(s.num()));
            CHECK(t.dtype == v.obj().at("dtype").str() && t.shape == shape && t.offset == 8 + hb + b && t.bytes == x - b);
            ++n;
        }
    }
    std::printf("real pack %s: %zu tensors compared\n", dir.string().c_str(), n);
}

}  // namespace

int main(int argc, char** argv) {
    ::setenv("STRIX_GLM_LOAD_DIRECT", "0", 1);  // parsing is under test; buffered reads work on every test filesystem
    const char* td = std::getenv("TMPDIR");
    const fs::path root = fs::path(td && *td ? td : "/tmp") / ("glm-pack-safety-" + std::to_string(::getpid()));
    fs::create_directories(root);
    const std::string only = argc > 1 ? argv[1] : "";
    for (const auto& c : cases()) {
        if (!only.empty() && only != c.name) continue;
        std::fprintf(stderr, "case %s\n", c.name);
        const int before = failures;
        c.run(root);
        if (failures != before) std::fprintf(stderr, "case %s FAILED\n", c.name);
    }
    if (const char* real = std::getenv("GLM_PACK_DIR"); real && *real && only.empty()) real_pack(real);
    fs::remove_all(root);
    std::printf("glm pack safety tests: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
