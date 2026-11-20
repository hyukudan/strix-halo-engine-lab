// Hy4-flash DFly2 naming map (include/strix/weight/hy4_dfly_map.hpp) on the shared tensor_map checker, through the shared
// safetensors index: a synthetic shard written from the table at a small geometry is claimed exactly once with no error;
// a wrong shape, a missing tensor and a foreign tensor are each reported. With a path argument the real official shard
// header is checked at the full geometry (87 tensors, BF16).
#include "strix/weight/hy4_dfly_map.hpp"
#include "strix/weight/source.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace tmap = strix::weight::tensor_map;
namespace w = strix::weight;

namespace {
#define DM_CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(1); } } while (0)

struct Named { std::string name; std::vector<std::uint64_t> shape; };  // stored order (outer axis first)

std::vector<Named> expand_table(const tmap::Model& m) {
    std::vector<Named> out;
    for (const auto& e : tmap::hy4_dfly2_table()) {
        const bool layered = e.layers != tmap::Layers::global;
        for (std::uint32_t l = 0; l < (layered ? m.blocks : 1U); ++l) {
            Named n{tmap::expand(e.name, layered ? std::int64_t(l) : -1), {}};
            for (const auto& x : e.dims) {
                if (x.a == tmap::Sym::none) break;
                n.shape.insert(n.shape.begin(), m.resolve(x.a, l) * m.resolve(x.b, l));  // GGUF order -> stored order
            }
            out.push_back(std::move(n));
        }
    }
    return out;
}

std::vector<w::Byte> write_shard(const std::vector<Named>& ts) {
    std::string h = "{\"__metadata__\":{\"format\":\"pt\"}";
    std::uint64_t off = 0;
    for (const auto& t : ts) {
        std::uint64_t n = 2;
        std::string shape;
        for (const auto d : t.shape) { n *= d; shape += (shape.empty() ? "" : ",") + std::to_string(d); }
        h += ",\"" + t.name + "\":{\"dtype\":\"BF16\",\"shape\":[" + shape + "],\"data_offsets\":[" + std::to_string(off) + "," +
             std::to_string(off + n) + "]}";
        off += n;
    }
    h += "}";
    std::vector<w::Byte> file(8 + h.size() + off, w::Byte{0});
    std::uint64_t len = h.size();
    for (int i = 0; i < 8; ++i) file[std::size_t(i)] = w::Byte(std::uint8_t(len >> (8 * i)));
    std::memcpy(file.data() + 8, h.data(), h.size());
    return file;
}

tmap::Result check_bytes(const std::vector<w::Byte>& file, const tmap::Model& m, std::size_t* count = nullptr) {
    w::TensorIndex index;
    const auto st = w::index_safetensors(w::memory_source(file), index);
    DM_CHECK(bool(st), "index_safetensors: %s", st.detail.c_str());
    std::vector<strix::gguf::TensorInfo> infos;
    std::string error;
    DM_CHECK(tmap::safetensors_infos(index, infos, error), "%s", error.c_str());
    if (count) *count = infos.size();
    return tmap::check(infos, tmap::hy4_dfly2_table(), m);
}

tmap::Model small_model() {
    tmap::Model m = tmap::hy4_dfly2_model();
    const auto set = [&m](tmap::Sym s, std::uint64_t v) { m.value[static_cast<std::size_t>(s)] = v; };
    set(tmap::Sym::hidden, 8); set(tmap::Sym::heads, 2); set(tmap::Sym::q_lora, 4); set(tmap::Sym::kv_lora, 4);
    set(tmap::Sym::kv_a, 6); set(tmap::Sym::qk_nope, 3); set(tmap::Sym::qk_head, 5); set(tmap::Sym::v_head, 4);
    set(tmap::Sym::kv_b_head, 7); set(tmap::Sym::dense_ff, 12);
    return m;
}
}  // namespace

int main(int argc, char** argv) {
    const auto m = small_model();
    auto ts = expand_table(m);
    DM_CHECK(ts.size() == 87, "table expands to %zu tensors (the shard has 87)", ts.size());
    std::size_t n = 0;
    auto r = check_bytes(write_shard(ts), m, &n);
    DM_CHECK(r.ok() && r.claims.size() == 87 && n == 87, "clean shard: %s", r.ok() ? "claims" : r.errors[0].c_str());

    auto bad = ts;
    for (auto& t : bad)
        if (t.name == "model.dfly_layers.3.self_attn.kv_b_proj.weight") t.shape[0] += 1;
    DM_CHECK(!check_bytes(write_shard(bad), m).ok(), "a wrong kv_b shape must be reported");
    bad = ts;
    bad.pop_back();
    DM_CHECK(!check_bytes(write_shard(bad), m).ok(), "a missing tensor must be reported");
    bad = ts;
    bad.push_back({"model.dfly_layers.5.mlp.up_proj.weight", {12, 8}});
    DM_CHECK(!check_bytes(write_shard(bad), m).ok(), "a sixth draft layer must be reported");

    if (argc > 1) {  // the official shard: header only is read (index_safetensors), at the full geometry
        std::shared_ptr<const w::Source> src;
        const auto st = w::open_source(argv[1], src);
        DM_CHECK(bool(st), "open %s: %s", argv[1], st.detail.c_str());
        w::TensorIndex index;
        const auto ix = w::index_safetensors(src, index);
        DM_CHECK(bool(ix), "index: %s", ix.detail.c_str());
        std::vector<strix::gguf::TensorInfo> infos;
        std::string error;
        DM_CHECK(tmap::safetensors_infos(index, infos, error), "%s", error.c_str());
        const auto real = tmap::check(infos, tmap::hy4_dfly2_table(), tmap::hy4_dfly2_model());
        for (std::size_t i = 0; i < real.errors.size() && i < 8; ++i) std::fprintf(stderr, "  %s\n", real.errors[i].c_str());
        DM_CHECK(real.ok() && infos.size() == 87, "official shard: %zu tensors, %zu errors", infos.size(), real.errors.size());
        std::printf("official DFly2 shard: 87 tensors claimed, shapes and BF16 match the map\n");
    }
    std::printf("hy4 dfly2 map tests PASS\n");
    return 0;
}
