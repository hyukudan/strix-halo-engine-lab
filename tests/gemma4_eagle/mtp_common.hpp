#pragma once
// MTP drafter test helpers: procedural target embedding (bit-identical to mtp_fixture.py), RoPE tables in the target
// RopeTables layout, synthetic target KV (full arrays and the engine's tagged 512-slot sliding ring), fixture reader.
#include "common.hpp"
#include "gemma4/eagle/mtp/mtp.hpp"
#include "gemma4/model/model.hpp"

#include <cmath>
#include <fstream>

namespace mtp_test {
using namespace eagle_test;
namespace mt = strix::gemma4::eagle::mtp;

inline std::uint64_t splitmix(std::uint64_t x) {
    std::uint64_t z = x + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}
// [262144][2560] BF16, row t col i = bf16(((splitmix(t*2560+i)>>40 & 0xffff)/65536 - 0.5) * 0.125)   (~1.34 GB)
struct Table {
    std::vector<std::uint8_t> bytes;
    Table() : bytes(std::size_t(strix::gemma4::vocab) * strix::gemma4::hidden * 2) {
        for (std::uint64_t x = 0; x < std::uint64_t(strix::gemma4::vocab) * strix::gemma4::hidden; ++x) {
            const double v = (double((splitmix(x) >> 40U) & 0xffffU) / 65536.0 - 0.5) * 0.125;
            const auto b = bf16_bits(float(v));
            bytes[2 * x] = std::uint8_t(b & 0xffU);
            bytes[2 * x + 1] = std::uint8_t(b >> 8U);
        }
    }
    LinearView view() const { return {bytes.data(), strix::gemma4::vocab, strix::gemma4::hidden, 2ULL * strix::gemma4::hidden, WeightFormat::bf16}; }
};

struct Rope {
    std::vector<float> sliding, global;
    std::uint32_t positions;
    // g4-v1 table recipe (rope_table_row): FP32 inverse frequency and FP32 angle = float(p) * f, as the target's
    // hashed tables and HF's rotary embedding do; cos/sin of that FP32 angle.
    explicit Rope(std::uint32_t n) : sliding(std::size_t(n) * 256), global(std::size_t(n) * 512), positions(n) {
        for (std::uint32_t p = 0; p < n; ++p) {
            for (std::uint32_t i = 0; i < 128; ++i) {
                const float f = 1.0F / float(std::pow(1e4, double(2 * i) / 256.0)), a = float(p) * f;
                sliding[std::size_t(p) * 256 + 2 * i] = float(std::cos(double(a)));
                sliding[std::size_t(p) * 256 + 2 * i + 1] = float(std::sin(double(a)));
            }
            for (std::uint32_t i = 0; i < 256; ++i) {
                float c = 1.0F, s = 0.0F;
                if (i < 64) {
                    const float f = 1.0F / float(std::pow(1e6, double(2 * i) / 512.0)), a = float(p) * f;
                    c = float(std::cos(double(a))); s = float(std::sin(double(a)));
                }
                global[std::size_t(p) * 512 + 2 * i] = c;
                global[std::size_t(p) * 512 + 2 * i + 1] = s;
            }
        }
    }
    mt::Tables view() const { return {sliding, global, positions}; }
};

// Target KV of the last sliding (2 x 256) and full (2 x 512) layers, position-major.
struct TargetKv {
    std::uint32_t T = 0;
    std::vector<float> ks, vs, kf, vf;
    std::vector<float> ring_k, ring_v;
    std::vector<std::uint64_t> tags;
    TargetKv() = default;
    TargetKv(std::uint32_t t, std::uint64_t seed) : T(t) {
        Rng r{seed};
        auto unit = [&](std::vector<float>& a, std::uint32_t d) {
            a.resize(std::size_t(T) * 2 * d);
            for (std::size_t row = 0; row < a.size() / d; ++row) {
                double ss = 0;
                for (std::uint32_t i = 0; i < d; ++i) { a[row * d + i] = 2 * r.uniform(); ss += double(a[row * d + i]) * a[row * d + i]; }
                const float inv = float(1.0 / std::sqrt(ss / d));
                for (std::uint32_t i = 0; i < d; ++i) a[row * d + i] *= inv;
            }
        };
        unit(ks, 256); unit(vs, 256); unit(kf, 512); unit(vf, 512);
    }
    mt::KvLayer full_sliding() const { return {ks.data(), vs.data(), nullptr, 512, T, true}; }
    mt::KvLayer full() const { return {kf.data(), vf.data(), nullptr, 1024, T, true}; }
    // Engine-style sliding ring holding positions max(0,p-512)..p-1 (the target's committed layer-22 state at p).
    mt::KvLayer ring(std::uint64_t p) {
        ring_k.assign(512ULL * 512, 0.0F); ring_v = ring_k; tags.assign(512, UINT64_MAX);
        for (std::uint64_t j = p > 512 ? p - 512 : 0; j < p; ++j) {
            std::copy_n(ks.data() + j * 512, 512, ring_k.data() + (j % 512) * 512);
            std::copy_n(vs.data() + j * 512, 512, ring_v.data() + (j % 512) * 512);
            tags[j % 512] = j;
        }
        return {ring_k.data(), ring_v.data(), tags.data(), 512, 512, false};
    }
};

struct Case {
    std::uint64_t p = 0;
    std::uint32_t anchor = 0;
    std::vector<std::uint32_t> tokens, inputs; // HF argmax per depth, HF input token per depth
    std::vector<double> margins;
};
struct Fixture {
    TargetKv kv;
    std::vector<float> hidden, step_hidden; // step_hidden [case][depth][2560]: HF input h of every depth
    std::vector<Case> cases;
    std::size_t K = 0;
    std::uint32_t sliding_keys = 513;
    static bool load(const std::filesystem::path& dir, Fixture& f) {
        if (!std::filesystem::exists(dir / "meta.json")) return false;
        const auto meta = read_json(dir / "meta.json");
        f.kv.T = std::uint32_t(meta.find("T")->as_int());
        f.sliding_keys = std::uint32_t(meta.find("sliding_keys")->as_int());
        for (const auto& c : meta.find("cases")->as_array()) {
            Case x;
            x.p = std::uint64_t(c.find("p")->as_int());
            x.anchor = std::uint32_t(c.find("anchor")->as_int());
            for (const auto& t : c.find("tokens")->as_array()) x.tokens.push_back(std::uint32_t(t.as_int()));
            for (const auto& m : c.find("margins")->as_array()) x.margins.push_back(m.as_double());
            for (const auto& t : c.find("inputs")->as_array()) x.inputs.push_back(std::uint32_t(t.as_int()));
            f.cases.push_back(x);
        }
        const auto T = f.kv.T;
        std::ifstream in(dir / "data.bin", std::ios::binary);
        auto read = [&](std::vector<float>& a, std::size_t n) {
            a.resize(n);
            in.read(reinterpret_cast<char*>(a.data()), std::streamsize(n * 4));
        };
        read(f.kv.ks, std::size_t(T) * 512); read(f.kv.vs, std::size_t(T) * 512);
        read(f.kv.kf, std::size_t(T) * 1024); read(f.kv.vf, std::size_t(T) * 1024);
        read(f.hidden, std::size_t(T) * 2560);
        f.K = std::size_t(meta.find("K")->as_int());
        read(f.step_hidden, f.cases.size() * f.K * 2560);
        return bool(in);
    }
};

} // namespace mtp_test
