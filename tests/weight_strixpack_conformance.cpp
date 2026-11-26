#include "core/lowbit_f16_staging.hpp"
#include "core/sha256.hpp"
#include "moe_core/grouped_strixpack.hpp"
#include "strix/serve/json.hpp"
#include "strix/weight/host_lowbit_gptq.hpp"

#include <bit>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
namespace w = strix::weight;
namespace sp = w::sp;
namespace grouped = strix::moe_core::grouped_strixpack;
namespace stage = strix::core::lowbit_f16;
namespace fs = std::filesystem;
using Bytes = std::vector<std::uint8_t>;

void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}

Bytes read(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path.string());
    return Bytes(std::istreambuf_iterator<char>(f), {});
}

std::uint32_t le(std::span<const std::uint8_t> data) {
    std::uint32_t v = 0;
    for (std::size_t i = 0; i < data.size(); ++i) v |= std::uint32_t{data[i]} << (8U * i);
    return v;
}

std::vector<float> floats(const Bytes& data) {
    require(data.size() % 4U == 0, "f32le shape");
    std::vector<float> values(data.size() / 4U);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = std::bit_cast<float>(le(std::span(data).subspan(4U * i, 4U)));
    return values;
}

bool same(float a, float b) { return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b); }
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return std::ranges::equal(a, b, [](float x, float y) { return same(x, y); });
}

w::Digest256 digest(std::span<const std::uint8_t> bytes) {
    strix::core::Sha256 h;
    h.update(bytes.data(), bytes.size());
    return {h.finish()};
}

std::string hex(w::Digest256 d) {
    constexpr std::string_view alphabet = "0123456789abcdef";
    std::string s;
    for (auto b : d.bytes) {
        s += alphabet[b / 16U];
        s += alphabet[b % 16U];
    }
    return s;
}

// Wire-only decoder. It does not use Storage, production scale expansion,
// the production table, sign handling, or block unpacking.
float ref(const std::uint8_t* block, unsigned j, unsigned bits, int base, const std::uint8_t* table) {
    const unsigned field = bits == 4U ? 4U : 2U;
    const unsigned bit = field * j;
    const unsigned lo = (block[bit / 8U] >> (bit % 8U)) % (1U << field);
    const unsigned c = bits == 3U ? lo + 4U * ((block[32U + j / 8U] >> (j % 8U)) % 2U) : lo;
    const unsigned levels = 1U << (bits - 1U);
    const unsigned s = block[16U * bits + j / 32U];
    const int m = static_cast<int>(table[c % levels] * (8U + s % 8U));
    return std::ldexp(static_cast<float>(c < levels ? m : -m), base + static_cast<int>(s / 8U) - 3);
}

constexpr sp::CpuIsa isas[]{sp::CpuIsa::scalar, sp::CpuIsa::avx2, sp::CpuIsa::avx512, sp::CpuIsa::automatic};

template <class S>
void differential(const sp::Header& h, const Bytes& payload, std::size_t n, std::size_t k,
                  const std::vector<float>& x, const std::vector<float>& published = {},
                  const std::vector<float>& projected = {}) {
    std::vector<float> expect(n * k), got(n * k);
    Bytes repacked(payload.size());
    for (std::size_t b = 0; b < n * k / 128U; ++b) {
        const auto* block = payload.data() + b * S::bytes;
        for (unsigned j = 0; j < 128; ++j) {
            auto& v = expect[b * 128U + j];
            v = ref(block, j, S::bits, h.base, h.table.data());
            require(same(v, sp::decode<S>(block, j, h)), "single-element decoder bits");
        }
        require(sp::repack_block<S>(expect.data() + b * 128U, block + S::scale_offset, h,
                                    repacked.data() + b * S::bytes), "lossless repack status");
    }
    require(repacked == payload, "pack -> decode -> repack bytes");
    require(published.empty() || same(expect, published), "published decoded bits");

    std::vector<float> y(x.size() / k * n), actual(y.size());
    for (std::size_t a = 0; a < x.size() / k; ++a)
        for (std::size_t r = 0; r < n; ++r) {
            float acc = 0;
            for (std::size_t j = 0; j < k; ++j) acc = std::fma(x[a * k + j], expect[r * k + j], acc);
            y[a * n + r] = acc;
        }
    require(projected.empty() || same(y, projected), "published projection bits");

    for (auto isa : isas) {
        if (!sp::cpu_supports(isa)) {
            require(sp::cpu_decode(S::bits, isa) == sp::cpu_decode(S::bits, sp::CpuIsa::scalar) &&
                    sp::cpu_project(S::bits, isa) == sp::cpu_project(S::bits, sp::CpuIsa::scalar), "unsupported ISA fallback");
            continue;
        }
        sp::cpu_decode(S::bits, isa)(h, payload.data(), n, k, got.data());
        require(same(expect, got), "CPU bulk decode bits");
        for (std::size_t a = 0; a < x.size() / k; ++a)
            sp::cpu_project(S::bits, isa)(h, payload.data(), n, k, x.data() + a * k, actual.data() + a * n);
        require(same(y, actual), "CPU fused projection bits");
    }
}

struct HostMma {
    using Fragment = std::array<float, 16>;
    std::array<float, 8> operator()(const Fragment& a, const Fragment& w, std::array<float, 8> sum) const {
        for (unsigned j = 0; j < 16; ++j) sum[0] = std::fma(a[j], w[j], sum[0]);
        return sum;
    }
};

void fixed_head(const sp::Header& h, const Bytes& payload, std::size_t n, std::size_t k,
                const std::vector<float>& x, const std::vector<float>& expected) {
    constexpr auto table = stage::sign_index_table(sp::default_table<4>());
    const std::span bytes(payload);
    std::vector<float> out(expected.size());
    for (std::size_t a = 0; a < x.size() / k; ++a)
        for (std::size_t r = 0; r < n; ++r) {
            std::array<float, 8> y{};
            for (std::size_t b = 0; b < k / 128U; ++b) {
                const auto block = bytes.subspan(68U * (r * (k / 128U) + b), 68U);
                for (unsigned g = 0; g < 4; ++g) {
                    stage::GroupAccumulator<std::array<float, 8>, HostMma> group;
                    for (unsigned u = 0; u < 2; ++u) {
                        HostMma::Fragment af{}, wf{};
                        const auto j0 = 32U * g + 16U * u;
                        for (unsigned i = 0; i < 2; ++i) {
                            std::uint32_t pairs[4];
                            stage::sign_index_pairs<table>(le(block.subspan(j0 / 2U + 4U * i, 4U)), pairs);
                            for (unsigned j = 0; j < 8; ++j)
                                wf[8U * i + j] = w::lowbit::f16_bits_to_f32(
                                    static_cast<std::uint16_t>(pairs[j / 2U] >> (16U * (j % 2U))));
                        }
                        for (unsigned j = 0; j < 16; ++j) {
                            const float value = x[a * k + b * 128U + j0 + j];
                            af[j] = w::lowbit::f16_bits_to_f32(w::lowbit::f32_to_f16_bits(value));
                            const auto want = ref(block.data(), j0 + j, 4, 0, h.table.data());
                            require(same(wf[j] * sp::scale_value(block[64U + g], 0), want), "fixed head FP16 operand");
                        }
                        group.add(af, wf);
                    }
                    group.apply(sp::scale_value(block[64U + g], 0), y);
                }
            }
            out[a * n + r] = y[0] * std::ldexp(1.0F, h.base);
        }
    require(same(out, expected), "fixed-head staging and group FMA chain");
}

void container_refusals(const Bytes& original, const w::Digest256& checkpoint) {
    const fs::path path = "strixpack-conformance-invalid.lp";
    struct Cleanup {
        const fs::path& path;
        ~Cleanup() { std::error_code ec; fs::remove(path, ec); }
    } cleanup{path};
    const auto probe = [&](const Bytes& data) {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        f.close();
        require(bool(f), "writing the invalid container");
        w::LowbitPack pack;
        require(!w::LowbitPack::open_host(path.string(), checkpoint, pack), "invalid container accepted");
    };
    const auto field = [&](std::string_view text, char replacement) {
        auto data = original;
        const auto i = std::search(data.begin(), data.end(), text.begin(), text.end());
        require(i != data.end(), "fixture mutation anchor");
        *i = static_cast<std::uint8_t>(replacement);
        probe(data);
    };
    field("sp3 strix.sp3", '\t');
    field("8192", '0');  // offset not aligned
    field("010305", '2');  // tensor version
    field("128 4096", '0');  // width 28: a partial block
    auto data = original;
    const std::string_view second = "8192 936";
    auto p = std::search(data.begin(), data.end(), second.begin(), second.end());
    require(p != data.end(), "fixture overlap anchor");
    std::copy_n("4096", 4, p);
    probe(data);
    data = original;
    data[4095] = 0;
    probe(data);
    data = original;
    data.pop_back();
    probe(data);
}

void bundle(const fs::path& dir) {
    const auto text = read(dir / "manifest.json");
    const auto manifest = strix::serve::parse_json_unique(std::string(text.begin(), text.end()));
    for (const auto& [name, sha] : manifest.find("sha256")->as_object())
        require(hex(digest(read(dir / name))) == sha.as_string(), "fixture file SHA-256");
    const std::string id = "Strix Pack conformance checkpoint string";
    const auto checkpoint = digest(Bytes(id.begin(), id.end()));
    const auto path = (dir / "vectors.lp").string();
    w::LowbitPack pack;
    require(bool(w::LowbitPack::open_host(path, checkpoint, pack)), "fixture container open");
    require(hex(pack.identity()) == manifest.find("pack_identity")->as_string() && bool(pack.verify_identity()),
            "fixture container identity");
    const auto bytes = read(path);
    for (const auto& e : pack.entries()) {
        const auto& h = *e.sp_header;
        const auto k = static_cast<std::size_t>(e.columns), n = static_cast<std::size_t>(e.rows);
        const auto payload_slice = std::span(bytes).subspan(static_cast<std::size_t>(e.offset), static_cast<std::size_t>(e.bytes));
        const Bytes payload(payload_slice.begin(), payload_slice.end());
        const auto input = floats(read(dir / (e.tensor + ".activations.f32le")));
        const auto expected = floats(read(dir / (e.tensor + ".decoded.f32le")));
        const auto projections = floats(read(dir / (e.tensor + ".project.f32le")));
        if (h.bits == 2) differential<sp::SP2Storage>(h, payload, n, k, input, expected, projections);
        if (h.bits == 3) differential<sp::SP3Storage>(h, payload, n, k, input, expected, projections);
        if (h.bits == 4) differential<sp::SP4Storage>(h, payload, n, k, input, expected, projections);
        if (e.storage == w::kSP4TWeightStorage) {
            const auto expected_head = floats(read(dir / "sp4t.head.f32le"));
            require(!same(expected_head, projections), "head and ordered FP32 discriminating fixture");
            fixed_head(h, payload, n, k, input, expected_head);
        }
        const grouped::MatrixView view{e.storage, n, k, h, payload};
        const std::array<std::uint32_t, 2> ids{0, 0};
        const std::array<float, 2> gates{1, 1};
        const auto plan = strix::moe_core::build_expert_major_plan({2, 1, 1, 0, ids}, gates, 1);
        std::vector<float> y(projections.size());
        for (auto mode : {grouped::InputRows::token, grouped::InputRows::assignment}) {
            require(bool(grouped::project(plan, std::span(&view, 1), mode, k, n, input, y)), "grouped fixture status");
            require(same(y, projections), "grouped fixture projection");
        }
    }
    container_refusals(bytes, checkpoint);
}

w::Slice source_slice(const Bytes& values) {
    return {w::memory_source(std::as_bytes(std::span(values))), {0, values.size()}};
}

void source_decoders(const fs::path& dir) {
    using Encoding = w::PackMatrixSource::Encoding;
    std::vector<float> decoded;
    // All BF16 encodings, including signed zero, infinity and NaN payloads.
    auto codes = read(dir / "bf16.codes.bin");
    auto source = w::PackMatrixSource{Encoding::bf16, 256, 256, source_slice(codes), {}};
    require(bool(source.decode(decoded)), "BF16 source decode");
    for (std::uint32_t i = 0; i < 65536U; ++i)
        require(std::bit_cast<std::uint32_t>(decoded[i]) == i * 65536U, "BF16 source widening");
    require(same(decoded, floats(read(dir / "bf16.decoded.f32le"))), "published BF16 decode");

    // All 16 codes at every finite E8M0 exponent whose largest code is finite.
    codes = read(dir / "mxfp4.codes.bin");
    auto scales = read(dir / "mxfp4.scales.bin");
    source = {Encoding::mxfp4, 253, 32, source_slice(codes), source_slice(scales)};
    require(bool(source.decode(decoded)), "MXFP4 source decode");
    constexpr float magnitudes[]{0, 0.5F, 1, 1.5F, 2, 3, 4, 6};
    for (unsigned r = 0; r < 253U; ++r)
        for (unsigned j = 0; j < 32; ++j) {
            const float m = magnitudes[j % 8U];
            const auto ref_value = std::ldexp(j % 16U < 8U ? m : -m, static_cast<int>(r) - 127);
            require(same(decoded[r * 32U + j], ref_value), "MXFP4 E2M1/E8M0 arithmetic");
        }
    require(same(decoded, floats(read(dir / "mxfp4.decoded.f32le"))), "published MXFP4 decode");
    for (std::uint8_t bad : {std::uint8_t{254}, std::uint8_t{255}}) {
        scales.back() = bad;
        source.scales = source_slice(scales);
        require(!source.decode(decoded) && decoded.size() == 253U * 32U, "MXFP4 source refusal");
    }

    // E4M3FN covers all byte codes and both edges of a final 1x1 tile.
    constexpr std::size_t n = 129, k = 129;
    codes = read(dir / "fp8.codes.bin");
    scales = read(dir / "fp8.scales.bin");
    constexpr float multipliers[]{0.5F, -2.0F, 4.0F, 0.25F};
    source = {Encoding::fp8_block, n, k, source_slice(codes), source_slice(scales)};
    require(bool(source.decode(decoded)), "E4M3 source decode");
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t j = 0; j < k; ++j) {
            const auto c = codes[r * k + j];
            const auto fraction = c % 8U, exp = (c / 8U) % 16U;
            const float mag = exp == 0 ? std::ldexp(static_cast<float>(fraction), -9) :
                std::ldexp(static_cast<float>(8U + fraction), static_cast<int>(exp) - 10);
            const auto ref_value = (c < 128 ? mag : -mag) * multipliers[(r / 128U) * 2U + j / 128U];
            const float actual = decoded[r * k + j];
            require(c % 128U == 127U ? std::isnan(actual) : same(actual, ref_value), "E4M3FN source arithmetic");
        }
    for (auto& v : decoded)
        if (std::isnan(v)) v = std::bit_cast<float>(0x7fc00000U);
    require(same(decoded, floats(read(dir / "fp8.decoded.f32le"))), "published FP8 decode");
    --source.codes.range.bytes;
    require(!source.decode(decoded), "short source payload accepted");
}

template <class S>
void coder_and_geometry(w::FormatId format, const fs::path& dir) {
    using w::lowbit::EncodeStatus;
    auto zero = floats(read(dir / "zero.f32le"));
    std::vector<float> decoded(128);
    const auto table = sp::default_table<S::bits>();
    const auto h = sp::make_header<S>(zero.data(), 1, 128, table);
    Bytes bytes(S::bytes);
    require(bool(sp::encode_rows<S>(zero.data(), 1, 128, h, bytes.data(), decoded.data())), "zero encode");
    require(h.base == 0 && bytes == read(dir / ("zero.sp" + std::to_string(S::bits))) &&
            decoded == std::vector<float>(128, static_cast<float>(table[0])), "hand-computed zero group");
    require(sp::nearest_scale_byte(1.0625, 0) == 0U && sp::nearest_scale_byte(1.9375, 0) == 7U &&
            sp::nearest_scale_byte(1e100, 0) == 255U, "scale midpoint, carry tie and saturation");
    const float midpoint = static_cast<float>(table[0] + table[1]) / 2;
    require(sp::quantise<S>(midpoint, 1, table.data()) == 0 &&
            sp::quantise<S>(-midpoint, 1, table.data()) == S::levels &&
            sp::quantise<S>(std::numeric_limits<float>::max(), 1, table.data()) == S::levels - 1,
            "code midpoint and saturation");
    zero.front() = std::numeric_limits<float>::denorm_min();
    require(sp::make_header<S>(zero.data(), 1, 128, table).base == -120, "denormal source base");
    const auto nonfinite = floats(read(dir / "nonfinite.f32le"));
    require(nonfinite.size() == 3U * 128U, "bad encoder fixture geometry");
    for (unsigned i = 0; i < 3U; ++i) {
        const auto* bad = nonfinite.data() + i * 128U;
        bytes.assign(bytes.size(), 0xa5);
        decoded.assign(decoded.size(), -7);
        require(sp::encode_rows<S>(bad, 1, 128, h, bytes.data(), decoded.data()).code == EncodeStatus::nonfinite,
                "RTN refuses nonfinite inputs");
        w::lowbit::gptq::Options opt;
        opt.dequant = decoded.data();
        require(w::lowbit::gptq::encode_rows_sp<S>(h, bad, 1, 128, bytes.data(), opt).code == EncodeStatus::nonfinite,
                "calibrated coder refuses nonfinite inputs");
        require(bytes == Bytes(S::bytes, 0xa5) && decoded == std::vector<float>(128, -7), "refusal mutated buffers");
    }
    try {
        (void)sp::encode_rows<S>(zero.data(), 1, 127, h, bytes.data());
        throw std::runtime_error("RTN accepted odd K");
    } catch (const std::invalid_argument&) {}

    w::Shape shape{};
    shape.rank = 2;
    auto size = [&](std::uint64_t n, std::uint64_t k, std::uint64_t expected) {
        shape.dims = {n, k, 0, 0};
        w::MatrixLoadGeometry g{};
        const auto s = w::strixpack_matrix_geometry(format, shape, g);
        require(expected ? s && g.bytes == expected : !s, "shape and address-range boundary");
    };
    constexpr auto max = std::numeric_limits<std::uint64_t>::max();
    size(0, 128, 0);
    size(1, 0, 0);
    size(1, 129, 0);
    size(1, 128, S::bytes);
    size(max / S::bytes, 128, (max / S::bytes) * S::bytes);
    size(max / S::bytes + 1U, 128, 0);
    size(1, max / 128U * 128U, (max / 128U) * S::bytes);
    size(200058, 2304, 200058U * 18U * S::bytes);
}

template <class S>
void coded_round_trip() {
    constexpr std::size_t k = 256, n = 5;
    std::vector<float> weights(n * k), decoded(n * k), factor(k * k);
    std::mt19937 rng(0x7125 + S::bits);
    for (auto& value : weights) value = (static_cast<float>(rng() % 101U) - 50.0F) / 128.0F;
    for (std::size_t j = 0; j < k; ++j) {
        factor[j * k + j] = 1.0F;
        if (j + 1U < k) factor[j * k + j + 1U] = 0.25F;
    }
    const auto h = sp::make_header<S>(weights.data(), n, k, sp::fit_table<S>(weights.data(), n, k, 1));
    Bytes bytes(n * S::row_bytes(k));
    for (const auto* f : {static_cast<const float*>(nullptr), static_cast<const float*>(factor.data())}) {
        w::lowbit::gptq::Options opts;
        opts.factor = f;
        opts.dequant = decoded.data();
        opts.threads = 1;
        require(bool(w::lowbit::gptq::encode_rows_sp<S>(h, weights.data(), n, k, bytes.data(), opts)), "coder round trip");
        differential<S>(h, bytes, n, k, std::vector<float>(k, 0.25F), decoded);
    }
}

template <class S>
void random_matrix() {
    constexpr std::size_t n = 2049, k = 2048;
    auto h = sp::header_for<S>(1.0F, sp::default_table<S::bits>());
    std::mt19937 rng(500U + S::bits);
    Bytes bytes(n * S::row_bytes(k));
    for (auto& b : bytes) b = static_cast<std::uint8_t>(rng());
    std::vector<float> x(k);
    for (auto& v : x) v = (static_cast<float>(rng() % 10001U) - 5000.0F) / 5001.0F;
    differential<S>(h, bytes, n, k, x);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected fixture directory");
        bundle(argv[1]);
        source_decoders(argv[1]);
        coder_and_geometry<sp::SP2Storage>(w::kSP2WeightStorage, argv[1]);
        coder_and_geometry<sp::SP3Storage>(w::kSP3WeightStorage, argv[1]);
        coder_and_geometry<sp::SP4Storage>(w::kSP4WeightStorage, argv[1]);
        coded_round_trip<sp::SP2Storage>();
        coded_round_trip<sp::SP3Storage>();
        coded_round_trip<sp::SP4Storage>();
        random_matrix<sp::SP2Storage>();
        random_matrix<sp::SP3Storage>();
        random_matrix<sp::SP4Storage>();
        std::cout << "Strix Pack v1 conformance PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
