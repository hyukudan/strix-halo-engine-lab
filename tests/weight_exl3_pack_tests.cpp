// EXL3 (trellis) source of the Strix pack builder: a "<base>.trellis" + ".suh" + ".svh" tensor set read through
// PackMatrixSource, converted to SP4 rows, and compared with packing the same dense FP32 matrix directly.
//  1. bind + decode equal exl3_dense::dense_rows bit for bit (2, 3, 4, 5, 6, 8 bits);
//  2. the SP4 rows of the decoded source are byte-identical to the rows of the dense matrix packed directly, and the
//     shared SP4 row decoder gives the same values as the coder's dequantised output;
//  3. refusals: width not 16*K, K outside 1..8, payload and shape disagree, shape not in whole 128-blocks, missing or
//     mis-sized suh / svh, wrong dtype.
#include "strix/weight/formats/exl3_dense.hpp"
#include "strix/weight/host_lowbit.hpp"
#include "strix/weight/host_lowbit_gptq.hpp"
#include "strix/weight/pack_source.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

namespace wei = strix::weight;
namespace ed = wei::exl3_dense;
namespace fs = std::filesystem;

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what.c_str());
        ++failures;
    }
}

struct Tensor {
    std::string name, dtype;
    std::vector<std::uint64_t> shape;
    std::vector<std::uint8_t> bytes;
};

void write_safetensors(const fs::path& path, const std::vector<Tensor>& tensors) {
    std::string header = "{";
    std::uint64_t offset = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        std::string shape;
        for (std::size_t d = 0; d < t.shape.size(); ++d) shape += (d ? "," : "") + std::to_string(t.shape[d]);
        header += (i ? "," : "") + std::string("\"") + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[" + shape +
                  "],\"data_offsets\":[" + std::to_string(offset) + "," + std::to_string(offset + t.bytes.size()) + "]}";
        offset += t.bytes.size();
    }
    header += "}";
    while (header.size() % 8U) header += ' ';
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    std::uint8_t prefix[8];
    for (unsigned i = 0; i < 8U; ++i) prefix[i] = static_cast<std::uint8_t>((std::uint64_t{header.size()} >> (8U * i)) & 255U);
    out.write(reinterpret_cast<const char*>(prefix), 8);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto& t : tensors) out.write(reinterpret_cast<const char*>(t.bytes.data()), static_cast<std::streamsize>(t.bytes.size()));
}

std::uint16_t half_of(float x) {
    std::uint16_t h = 0;
    (void)wei::ieee_half::encode_finite(x, h);
    return h;
}

std::vector<std::uint8_t> half_bytes(const std::vector<std::uint16_t>& values) {
    std::vector<std::uint8_t> bytes;
    for (const auto v : values) {
        bytes.push_back(static_cast<std::uint8_t>(v & 255U));
        bytes.push_back(static_cast<std::uint8_t>(v >> 8U));
    }
    return bytes;
}

struct Set {
    unsigned inputs, outputs, bits;
    std::vector<std::uint8_t> rings;
    std::vector<std::uint16_t> suh, svh;
    Tensor trellis, suh_t, svh_t;
};

Set synthetic(unsigned inputs, unsigned outputs, unsigned bits, std::mt19937& rng) {
    Set s{inputs, outputs, bits, {}, {}, {}, {}, {}, {}};
    s.rings.resize(std::size_t{inputs / 16U} * (outputs / 16U) * 32U * bits);
    for (auto& b : s.rings) b = static_cast<std::uint8_t>(rng() & 255U);
    std::uniform_real_distribution<float> scale(-0.25F, 0.25F);
    for (unsigned i = 0; i < inputs; ++i) s.suh.push_back(half_of(scale(rng)));
    for (unsigned o = 0; o < outputs; ++o) s.svh.push_back(half_of(scale(rng)));
    s.trellis = {"layer.q_proj.trellis", "U16", {inputs / 16U, outputs / 16U, 16U * bits}, s.rings};
    s.suh_t = {"layer.q_proj.suh", "F16", {inputs}, half_bytes(s.suh)};
    s.svh_t = {"layer.q_proj.svh", "F16", {outputs}, half_bytes(s.svh)};
    return s;
}

wei::Status bind(const fs::path& dir, const std::vector<Tensor>& tensors, wei::PackMatrixSource& out,
                 const std::string& name = "layer.q_proj.trellis") {
    const auto path = dir / "model.safetensors";
    write_safetensors(path, tensors);
    wei::PackCheckpoint checkpoint;
    if (const auto status = wei::PackCheckpoint::open(path.string(), checkpoint); !status) return status;
    const auto* tensor = checkpoint.find(name);
    if (!tensor) return {wei::Error::malformed, "no such tensor"};
    const auto status = wei::PackMatrixSource::bind(checkpoint, *tensor, out);
    // The source slices keep the file open through their own ownership; nothing else to release here.
    return status;
}

template <class S>
void pack_equal(const std::vector<float>& a, const std::vector<float>& b, std::size_t n, std::size_t k, const std::string& tag) {
    const auto table = wei::sp::default_table<S::bits>();
    const auto ha = wei::sp::make_header<S>(a.data(), n, k, table);
    const auto hb = wei::sp::make_header<S>(b.data(), n, k, table);
    std::vector<std::uint8_t> ba(n * S::row_bytes(k)), bb(ba.size());
    std::vector<float> da(n * k), db(n * k);
    check(bool(wei::sp::encode_rows<S>(a.data(), n, k, ha, ba.data(), da.data())) &&
              bool(wei::sp::encode_rows<S>(b.data(), n, k, hb, bb.data(), db.data())),
          tag + ": encode");
    check(ba == bb && std::memcmp(&ha, &hb, sizeof ha) == 0, tag + ": SP4 bytes and header are identical");
    bool same = true;
    const std::size_t blocks = k / S::elements;
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t blk = 0; blk < blocks; ++blk)
            for (unsigned j = 0; j < S::elements; ++j) {
                const float v = wei::sp::decode<S>(ba.data() + (r * blocks + blk) * S::bytes, j, ha);
                same = same && std::memcmp(&v, &da[r * k + blk * S::elements + j], sizeof v) == 0;
            }
    check(same, tag + ": the shared SP4 row decoder returns the coder's values");
}

void conformance(const fs::path& dir, std::mt19937& rng) {
    for (const unsigned bits : {2U, 3U, 4U, 5U, 6U, 8U}) {
        const auto s = synthetic(256, 128, bits, rng);
        wei::PackMatrixSource source;
        const auto tag = "K=" + std::to_string(bits);
        const auto status = bind(dir, {s.trellis, s.suh_t, s.svh_t}, source);
        check(bool(status), tag + ": bind " + status.detail);
        if (!status) continue;
        check(source.encoding == wei::PackMatrixSource::Encoding::exl3 && source.rows == 128 && source.columns == 256,
              tag + ": geometry (rows = outputs, columns = inputs)");
        std::vector<float> decoded;
        wei::Digest256 digest{}, again{};
        check(bool(source.decode(decoded, &digest)) && bool(source.digest(again)) && digest.bytes == again.bytes,
              tag + ": decode and digest agree");
        const auto dense = ed::dense_rows({s.rings, s.suh, s.svh, s.inputs, s.outputs, bits});
        check(decoded.size() == dense.size() &&
                  std::memcmp(decoded.data(), dense.data(), dense.size() * sizeof(float)) == 0,
              tag + ": decode equals dense_rows bit for bit");
        pack_equal<wei::sp::SP4Storage>(decoded, dense, source.rows, source.columns, tag + " SP4");
        auto flipped = s;
        flipped.svh_t.bytes[0] ^= 1U;
        wei::PackMatrixSource other;
        wei::Digest256 changed{};
        check(bool(bind(dir, {flipped.trellis, flipped.suh_t, flipped.svh_t}, other)) && bool(other.digest(changed)) &&
                  changed.bytes != digest.bytes,
              tag + ": the digest covers the output scales");
    }
}

void refusals(const fs::path& dir, std::mt19937& rng) {
    const auto s = synthetic(128, 128, 4, rng);
    wei::PackMatrixSource out;
    const auto refused = [&](const std::string& what, const std::vector<Tensor>& tensors) {
        check(!bind(dir, tensors, out), "refused: " + what);
    };
    refused("missing svh", {s.trellis, s.suh_t});
    refused("missing suh", {s.trellis, s.svh_t});
    auto t = s;
    t.suh_t.shape = {127};
    t.suh_t.bytes.resize(254);
    refused("suh length differs from the input width", {t.trellis, t.suh_t, t.svh_t});
    t = s;
    t.svh_t.shape = {256};
    t.svh_t.bytes.resize(512);
    refused("svh length differs from the output width", {t.trellis, t.suh_t, t.svh_t});
    t = s;
    t.suh_t.dtype = "F32";
    refused("suh dtype", {t.trellis, t.suh_t, t.svh_t});
    t = s;
    t.trellis.dtype = "BF16";
    refused("trellis dtype", {t.trellis, t.suh_t, t.svh_t});
    for (const std::uint64_t width : {0ULL, 8ULL, 24ULL, 144ULL, 160ULL}) {
        t = s;
        t.trellis.shape[2] = width;
        t.trellis.bytes.assign(std::size_t{8} * 8U * width * 2U, 0);
        refused("trellis width " + std::to_string(width) + " (not 16*K, K in 1..8)", {t.trellis, t.suh_t, t.svh_t});
    }
    t = s;
    t.trellis.bytes.pop_back();
    refused("payload shorter than its shape", {t.trellis, t.suh_t, t.svh_t});
    t = s;
    t.trellis.shape = {8, 8};
    refused("rank 2 trellis", {t.trellis, t.suh_t, t.svh_t});
    const auto small = synthetic(64, 128, 4, rng);
    refused("input width not in whole 128-blocks", {small.trellis, small.suh_t, small.svh_t});
    // A source built by hand with a ring payload that fits no K is refused by decode as well.
    wei::PackMatrixSource ok;
    check(bool(bind(dir, {s.trellis, s.suh_t, s.svh_t}, ok)), "baseline binds");
    auto cut = ok;
    --cut.codes.range.bytes;
    std::vector<float> values;
    check(!cut.decode(values), "refused: decode of a payload that fits no K");
    cut = ok;
    --cut.scales.range.bytes;
    check(!cut.decode(values), "refused: decode with short suh");
}

}  // namespace

int main() {
    std::mt19937 rng(20261128U);
    const auto dir = fs::temp_directory_path() / ("strix-exl3-pack-" + std::to_string(rng()));
    fs::create_directories(dir);
    conformance(dir, rng);
    refusals(dir, rng);
    std::error_code ignored;
    fs::remove_all(dir, ignored);
    if (failures) return 1;
    std::printf("exl3 pack source: all checks passed\n");
    return 0;
}
