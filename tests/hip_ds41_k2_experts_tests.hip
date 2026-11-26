// GPU test of the deepseek41 K2 routed experts (hip_ds41_k2_experts.hpp) on the shared trellis MoE family, DS41 shape
// (H 5120, I 2304), 2 experts, 1 row, top 2, random K2 tiles and +-1 suh / svh, read through the K2 pack loader:
//   (1) two-rank form == single-node form, bitwise (rank partials placed in one array, same down finish);
//   (2) the same launch twice gives the same bits;
//   (3) single-node y vs a float64 CPU reference of the EXL3 linear (normalized 128-block Hadamards, mul1 codebook via
//       core::trellis::reference_bits) with the DS41 clamped SwiGLU: relative L2 error < 1e-2 (FP16 activations).
// Usage: strix-hip-ds41-k2-experts-tests [scratch dir]   (writes a 17.7 MB synthetic pack there, removes it at the end)
#include "strix/hip_ds41_k2_experts.hpp"

#include "core/trellis_codec.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace k2 = strix::ds41::k2;

namespace {
#define HIP_OK(x) do { if ((x) != hipSuccess) { std::fprintf(stderr, "HIP error %s\n", #x); std::exit(2); } } while (0)

void write_pack(const std::filesystem::path& file, const std::vector<std::pair<std::string, std::vector<std::uint8_t>>>& data,
                const std::vector<std::string>& meta) {
    std::string header = "{";
    std::uint64_t off = 0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        header += (i ? "," : "") + std::string("\"") + data[i].first + "\":{" + meta[i] + ",\"data_offsets\":[" + std::to_string(off) + "," +
                  std::to_string(off + data[i].second.size()) + "]}";
        off += data[i].second.size();
    }
    header += "}";
    while (header.size() % 8U != 0U) header += ' ';
    std::ofstream f(file, std::ios::binary);
    const std::uint64_t n = header.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(header.data(), static_cast<std::streamsize>(n));
    for (const auto& d : data) f.write(reinterpret_cast<const char*>(d.second.data()), static_cast<std::streamsize>(d.second.size()));
}

void fwht128(std::vector<double>& v) {  // normalized, per 128-block
    for (std::size_t b = 0; b < v.size(); b += 128)
        for (std::size_t h = 1; h < 128; h <<= 1)
            for (std::size_t i = b; i < b + 128; i += 2 * h)
                for (std::size_t j = i; j < i + h; ++j) {
                    const double a = v[j], c = v[j + h];
                    v[j] = a + c;
                    v[j + h] = a - c;
                }
    for (auto& x : v) x /= std::sqrt(128.0);
}

double half_of(const std::vector<std::uint8_t>& b, std::size_t i) {
    std::uint16_t h;
    std::memcpy(&h, b.data() + 2 * i, 2);
    return static_cast<double>(static_cast<float>(__builtin_bit_cast(_Float16, h)));
}

// y = svh * H_O(Q^T H_I(suh * x)) in double.
std::vector<double> linear_ref(const std::vector<std::uint8_t>& t, const std::vector<std::uint8_t>& suh, const std::vector<std::uint8_t>& svh,
                               const std::vector<double>& x, unsigned in, unsigned out) {
    std::vector<double> u(in), acc(out, 0.0);
    for (unsigned k = 0; k < in; ++k) u[k] = x[k] * half_of(suh, k);
    fwht128(u);
    const auto* words = reinterpret_cast<const std::uint32_t*>(t.data());
    for (unsigned k = 0; k < in; ++k)
        for (unsigned n = 0; n < out; ++n)
            acc[n] += u[k] * static_cast<double>(static_cast<float>(__builtin_bit_cast(_Float16, strix::core::trellis::reference_bits(words, 2, out, k, n))));
    fwht128(acc);
    for (unsigned n = 0; n < out; ++n) acc[n] *= half_of(svh, n);
    return acc;
}
}  // namespace

int main(int argc, char** argv) {
    const k2::Geometry g{5120, 2304, 2, 2};
    const auto dir = std::filesystem::path(argc > 1 ? argv[1] : std::filesystem::temp_directory_path().string()) / "strix-k2-gpu-test";
    std::filesystem::create_directories(dir);
    std::mt19937 rng(77);
    std::vector<std::pair<std::string, std::vector<std::uint8_t>>> data;
    std::vector<std::string> meta;
    std::vector<std::uint8_t> T[2][3], SU[2][3], SV[2][3];
    auto signs = [&](unsigned n) {
        std::vector<std::uint8_t> v(2U * n);
        for (unsigned i = 0; i < n; ++i) { const std::uint16_t h = (rng() & 1U) ? 0x3C00 : 0xBC00; std::memcpy(v.data() + 2 * i, &h, 2); }
        return v;
    };
    for (unsigned e = 0; e < 2; ++e)
        for (unsigned w = 0; w < 3; ++w) {
            const unsigned in = w < 2 ? g.hidden : g.inter, out = w < 2 ? g.inter : g.hidden;
            T[e][w].resize(std::size_t{in} * out / 4U);
            for (auto& b : T[e][w]) b = static_cast<std::uint8_t>(rng() & 255U);
            SU[e][w] = signs(in);
            SV[e][w] = signs(out);
            const std::string name = k2::pack_name(0, e, static_cast<k2::Mat>(w));
            data.push_back({name + ".trellis", T[e][w]});
            meta.push_back("\"dtype\":\"I16\",\"shape\":[" + std::to_string(in / 16) + "," + std::to_string(out / 16) + ",32]");
            data.push_back({name + ".suh", SU[e][w]});
            meta.push_back("\"dtype\":\"F16\",\"shape\":[" + std::to_string(in) + "]");
            data.push_back({name + ".svh", SV[e][w]});
            meta.push_back("\"dtype\":\"F16\",\"shape\":[" + std::to_string(out) + "]");
            data.push_back({name + ".mul1", {0x2D, 0xD1, 0xDC, 0x83}});
            meta.push_back("\"dtype\":\"I32\",\"shape\":[]");
        }
    write_pack(dir / "x.L0.c00.safetensors", data, meta);
    std::vector<float> x(g.hidden);
    std::uniform_real_distribution<float> dist(-0.05F, 0.05F);
    for (auto& v : x) v = dist(rng);
    const std::int32_t ids[2] = {1, 0};
    const float wts[2] = {0.625F, 0.375F};
    k2::Pack pack(dir, g);
    float *dx, *dw, *gu, *dn, *dy;
    std::int32_t* di;
    _Float16* dh;
    HIP_OK(hipMalloc(&dx, 4 * g.hidden));
    HIP_OK(hipMalloc(&dw, 8));
    HIP_OK(hipMalloc(&di, 8));
    HIP_OK(hipMemcpy(dx, x.data(), 4 * g.hidden, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dw, wts, 8, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(di, ids, 8, hipMemcpyHostToDevice));
    HIP_OK(hipMalloc(&gu, 4 * 2 * 2 * 16 * g.inter));
    HIP_OK(hipMalloc(&dh, 2 * 2 * g.inter));
    HIP_OK(hipMalloc(&dn, 4 * 2 * 4 * g.hidden));
    HIP_OK(hipMalloc(&dy, 4 * g.hidden));
    auto run = [&](std::uint32_t ranks) {
        HIP_OK(hipMemset(dn, 0xFF, 4 * 2 * 4 * g.hidden));
        k2::DeviceLayer last{};
        for (std::uint32_t r = 0; r < ranks; ++r) {
            auto L = k2::upload_layer(pack.load_layer(0, k2::rank_slice(g, r, ranks)));
            k2::DecodeView v{&L, 1, 2, 10.0F, dx, di, dw, gu, dh, dn, dy};
            k2::enqueue_k2_experts(v, nullptr);
            HIP_OK(hipDeviceSynchronize());
            if (r + 1 < ranks) k2::free_layer(L); else last = L;
        }
        k2::DecodeView v{&last, 1, 2, 10.0F, dx, di, dw, gu, dh, dn, dy};
        k2::enqueue_k2_down_finish(v, nullptr);  // any rank's layer: down svh is whole H on both
        HIP_OK(hipDeviceSynchronize());
        k2::free_layer(last);
        std::vector<float> y(g.hidden);
        HIP_OK(hipMemcpy(y.data(), dy, 4 * g.hidden, hipMemcpyDeviceToHost));
        return y;
    };
    const auto y1 = run(1), y1b = run(1), y2 = run(2);
    const bool same_tp = std::memcmp(y1.data(), y2.data(), 4 * g.hidden) == 0;
    const bool same_rerun = std::memcmp(y1.data(), y1b.data(), 4 * g.hidden) == 0;
    std::vector<double> ref(g.hidden, 0.0), xd(x.begin(), x.end());
    for (unsigned s = 0; s < 2; ++s) {
        const unsigned e = static_cast<unsigned>(ids[s]);
        const auto gt = linear_ref(T[e][0], SU[e][0], SV[e][0], xd, g.hidden, g.inter);
        const auto ut = linear_ref(T[e][1], SU[e][1], SV[e][1], xd, g.hidden, g.inter);
        std::vector<double> hh(g.inter);
        for (unsigned i = 0; i < g.inter; ++i) {
            const double gg = std::min(gt[i], 10.0), uu = std::max(-10.0, std::min(ut[i], 10.0));
            hh[i] = gg / (1.0 + std::exp(-gg)) * uu;
        }
        const auto yd = linear_ref(T[e][2], SU[e][2], SV[e][2], hh, g.inter, g.hidden);
        for (unsigned n = 0; n < g.hidden; ++n) ref[n] += wts[s] * yd[n];
    }
    double num = 0, den = 0;
    for (unsigned n = 0; n < g.hidden; ++n) { num += (y1[n] - ref[n]) * (y1[n] - ref[n]); den += ref[n] * ref[n]; }
    const double rel = std::sqrt(num / den);
    std::printf("K2GPU tp2_equals_single=%d rerun_equal=%d rel_l2_vs_f64=%.3e\n", same_tp ? 1 : 0, same_rerun ? 1 : 0, rel);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);  // the test's own synthetic pack
    const bool ok = same_tp && same_rerun && rel < 1e-2;
    std::puts(ok ? "K2GPU PASS" : "K2GPU FAIL");
    return ok ? 0 : 1;
}
