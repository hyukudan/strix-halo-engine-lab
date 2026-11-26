// Host-only regression: a cut pack never opens files that contain only suffix layers.
#include "cuda/models/minimax/mm_rtx_pack.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
void shard(const fs::path& file, const std::string& name, std::uint16_t value) {
    const std::string header = "{\"" + name + "\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
    const std::uint64_t bytes = header.size();
    std::ofstream out(file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&bytes), 8);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(&value), 2);
}
}
int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("need fixture output directory");
    const fs::path dir = argv[1];
    fs::create_directories(dir);
    shard(dir / "embedding.safetensors", "language_model.model.embed_tokens.weight", 0x3f80);
    shard(dir / "prefix.safetensors", "language_model.model.layers.0.input_layernorm.weight", 0x4000);
    {
        std::ofstream out(dir / "model.safetensors.index.json");
        out << R"({"weight_map":{"language_model.model.embed_tokens.weight":"embedding.safetensors","language_model.model.layers.0.input_layernorm.weight":"prefix.safetensors","language_model.model.layers.1.input_layernorm.weight":"absent-suffix.safetensors","language_model.model.norm.weight":"absent-norm.safetensors"}})";
    }
    {
        strix::minimax::rtxpf::host::Pack pack(dir.string(), 1);
        if (pack.floats("language_model.model.embed_tokens.weight")[0] != 1.0F ||
            pack.floats("language_model.model.layers.0.input_layernorm.weight")[0] != 2.0F ||
            pack.has("language_model.model.layers.1.input_layernorm.weight"))
            throw std::runtime_error("prefix pack shape/content");
    }
    bool rejected = false;
    try { strix::minimax::rtxpf::host::Pack pack(dir.string(), 2); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("requested missing layer was accepted");
    std::puts("PASS: prefix-only pack reads existing prefix and excludes missing suffix");

    // Malformed packs: every row is refused with an exception (never an out-of-directory open or out-of-bounds access).
    using Pack = strix::minimax::rtxpf::host::Pack;
    const auto raw_shard = [](const fs::path& file, const std::string& header, const std::string& payload,
                              std::uint64_t declared) {
        std::ofstream out(file, std::ios::binary);
        const std::uint64_t bytes = declared != 0 ? declared : header.size();
        out.write(reinterpret_cast<const char*>(&bytes), 8);
        out.write(header.data(), static_cast<std::streamsize>(header.size()));
        out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    };
    struct Row { const char* name; std::string shard_name, header, payload; std::uint64_t declared; int use; };
    const std::string w = R"({"w":{"dtype":"F32","shape":[2],"data_offsets":)";
    const std::string exl3 = R"({"m.trellis":{"dtype":"I16","shape":[8,8,32],"data_offsets":[0,4]},)"
                             R"("m.suh":{"dtype":"F16","shape":[128],"data_offsets":[4,260]},)"
                             R"("m.svh":{"dtype":"F16","shape":[128],"data_offsets":[260,516]}})";
    const std::vector<Row> rows = {
        {"parent shard", "../outside.safetensors", w + "[0,8]}}", std::string(8, '\0'), 0, 0},
        {"one offset", "a.safetensors", w + "[0]}}", std::string(8, '\0'), 0, 0},
        {"no offsets", "a.safetensors", w + "[]}}", std::string(8, '\0'), 0, 0},
        {"reversed offsets", "a.safetensors", w + "[8,0]}}", std::string(8, '\0'), 0, 0},
        {"negative offset", "a.safetensors", w + "[-8,0]}}", std::string(8, '\0'), 0, 0},
        {"past payload", "a.safetensors", w + "[0,4096]}}", std::string(8, '\0'), 0, 0},
        {"header past file", "a.safetensors", w + "[0,8]}}", std::string(8, '\0'), 1ULL << 40, 0},
        {"nested header", "a.safetensors", std::string(200000, '['), "", 0, 0},
        {"truncated escape", "a.safetensors", "{\"w\\", "", 0, 0},
        {"partial f32", "a.safetensors", R"({"w":{"dtype":"F32","shape":[1],"data_offsets":[0,6]}})", "abcdef", 0, 1},
        {"read wraps", "a.safetensors", w + "[0,8]}}", std::string(8, '\0'), 0, 2},
        {"exl3 extents", "a.safetensors", exl3, std::string(516, '\0'), 0, 3},
    };
    int failed = 0;
    for (const auto& row : rows) {
        const fs::path d = dir / "malformed" / row.name / "model";
        fs::create_directories(d);
        const fs::path file = row.shard_name.rfind("../", 0) == 0 ? d.parent_path() / row.shard_name.substr(3) : d / row.shard_name;
        raw_shard(file, row.header, row.payload, row.declared);
        {
            std::ofstream out(d / "model.safetensors.index.json");
            out << "{\"weight_map\":{\"w\":\"" << row.shard_name << "\"}}";
        }
        bool refused = false;
        try {
            Pack pack(d.string(), 0);
            if (row.use == 1) (void)pack.floats("w");
            if (row.use == 2) {
                char buf[8];
                pack.read(pack.info("w"), UINT64_MAX - 3, 8, buf);
            }
            if (row.use == 3) (void)pack.slice_out("m", 0, 128);
        } catch (const std::exception&) {
            refused = true;
        }
        if (!refused) {
            std::fprintf(stderr, "FAIL: malformed pack accepted: %s\n", row.name);
            ++failed;
        }
    }
    if (failed != 0) throw std::runtime_error("malformed packs accepted");
    std::puts("PASS: malformed packs are refused");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
