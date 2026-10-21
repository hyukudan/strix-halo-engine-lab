#include "strix/ds41_formats.hpp"
#include "strix/ds41_native_fp8.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Native FP8 dense weights: the integer form of every E4M3 code and the dot scales are exact, the declared product
// (lane order of the device kernel) agrees with a double-precision dot of the decoded weights and activations, the
// K-split parts and the extreme codes stay exact in int32, the model -> checkpoint names, and the safetensors reader
// (index and directory forms, rejection of NaN codes, exponents and shapes). With STRIX_DS41_NATIVE_FP8=DIR of the
// original checkpoint: every dense tensor of layer 0 is read and checked.
namespace {

using namespace strix::ds41;
int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

// rows x width weights in the device layout: E4M3 codes of N(0, 1) values scaled into the format's range, block
// exponents in [114, 119] (the 1017 checkpoint's range).
std::vector<std::uint8_t> random_fp8(std::mt19937& rng, std::size_t rows, std::size_t width) {
    std::vector<std::uint8_t> t(fp8_tensor_bytes(rows, width));
    std::normal_distribution<float> d(0.0F, 60.0F);
    for (std::size_t i = 0; i < rows * width; ++i) t[i] = float_to_e4m3(d(rng));
    std::uniform_int_distribution<int> e(114, 119);
    for (std::size_t i = rows * width; i < t.size(); ++i) t[i] = static_cast<std::uint8_t>(e(rng));
    return t;
}
std::vector<float> random_values(std::mt19937& rng, std::size_t n, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
// sum_c w(r, k0 + c) * x(c) in double with the A8 values x = code * delta, and sum |w x| (the rounding bound's scale).
std::pair<double, double> double_dot(const std::vector<std::uint8_t>& t, std::size_t rows, std::size_t width, std::size_t r, std::size_t k0,
                                     const A8Activation& x, std::size_t n) {
    double s = 0.0, a = 0.0;
    for (std::size_t c = 0; c < n; ++c) {
        const double p = static_cast<double>(fp8_weight(t.data(), rows, width, r, k0 + c)) * x.codes[c] * static_cast<double>(x.deltas[c / 32U]);
        s += p;
        a += std::fabs(p);
    }
    return {s, a};
}

void write_file(const std::filesystem::path& p, const std::string& bytes) {
    std::ofstream out(p, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}
// A safetensors file of the named (dtype, shape, bytes) entries.
struct Entry {
    std::string name, dtype;
    std::vector<std::size_t> shape;
    std::string bytes;
};
std::string safetensors(const std::vector<Entry>& entries) {
    std::string header = "{\"__metadata__\":{\"format\":\"pt\"}";
    std::size_t at = 0;
    std::string data;
    for (const auto& e : entries) {
        header += ",\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
        for (std::size_t i = 0; i < e.shape.size(); ++i) header += (i ? "," : "") + std::to_string(e.shape[i]);
        header += "],\"data_offsets\":[" + std::to_string(at) + "," + std::to_string(at + e.bytes.size()) + "]}";
        at += e.bytes.size();
        data += e.bytes;
    }
    header += "}";
    while (header.size() % 8U) header += ' ';
    std::string out(8, '\0');
    std::uint64_t n = header.size();
    for (int k = 0; k < 8; ++k, n >>= 8U) out[static_cast<std::size_t>(k)] = static_cast<char>(n & 0xffU);
    return out + header + data;
}

}  // namespace

int main() try {
    // Integer forms and dot scales.
    {
        unsigned bad = 0;
        for (unsigned c = 0; c < 256U; ++c) {
            if ((c & 0x7FU) == 0x7FU) continue;
            const auto code = static_cast<std::uint8_t>(c);
            const float v = std::ldexp(static_cast<float>(e4m3_integer(code)), -9);
            bad += std::bit_cast<std::uint32_t>(v) != std::bit_cast<std::uint32_t>(e4m3_to_float(code)) && !(v == 0.0F && e4m3_to_float(code) == 0.0F);
        }
        expect(bad == 0U, "e4m3_integer * 2^-9 is the E4M3 value for every finite code");
        expect(e4m3_integer(0x7E) == 229376 && e4m3_integer(0xFE) == -229376 && e4m3_integer(0x01) == 1 && e4m3_integer(0x08) == 8,
               "integer forms of 448, -448, the least subnormal and the least normal");
        bool scales = true;
        for (unsigned e = fp8_min_exponent; e <= fp8_max_exponent; ++e)
            scales = scales && fp8_dot_scale(static_cast<std::uint8_t>(e)) == std::ldexp(1.0F, static_cast<int>(e) - 136);
        expect(scales, "fp8_dot_scale(e) = 2^(e - 136) on [10, 254]");
        expect(fp8_dot_scale(10) == std::ldexp(1.0F, -126) && std::isnormal(fp8_dot_scale(10)), "least dot scale is a normal float");
    }

    std::mt19937 rng(41017);
    // The declared product against double precision, plain and K-split rows, the widths of the model's sites.
    for (const auto& [rows, width] : {std::pair<std::size_t, std::size_t>{64, 5120}, {32, 1280}, {64, 8192}, {32, 4096}, {32, 2304}}) {
        const auto t = random_fp8(rng, rows, width);
        expect(fp8_tensor_check({t.data(), rows * width}, {t.data() + rows * width, t.size() - rows * width}, rows, width).empty(),
               "random tensor passes the check");
        const auto x = quantize_a8(random_values(rng, width, 1.5F));
        double worst = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            const std::uint8_t* codes = t.data() + r * width;
            const std::uint8_t* scales = t.data() + rows * width + r / 32U * (width / 32U);
            const float y = product_fp8(codes, scales, x.codes.data(), x.deltas.data(), width);
            const auto [ref, mag] = double_dot(t, rows, width, r, 0, x, width);
            // groups: one rounding of the int32 dot and of the term, then width / 32 / 32 lane additions and 5 butterfly levels
            const double bound = mag * 0x1p-24 * (2.0 + static_cast<double>(width) / 1024.0 + 5.0) + 1e-30;
            worst = std::max(worst, std::fabs(static_cast<double>(y) - ref) / bound);
            if (width % 64U == 0U) {  // two K-halves: h0, then h1 accumulated (the K-split declaration)
                const std::size_t half = width / 2U;
                A8Activation h1;
                h1.codes.assign(x.codes.begin() + static_cast<std::ptrdiff_t>(half), x.codes.end());
                h1.deltas.assign(x.deltas.begin() + static_cast<std::ptrdiff_t>(half / 32U), x.deltas.end());
                const float y0 = product_fp8(codes, scales, x.codes.data(), x.deltas.data(), half);
                const float y1 = product_fp8(codes + half, scales + half / 32U, h1.codes.data(), h1.deltas.data(), half);
                const auto [r0, m0] = double_dot(t, rows, width, r, 0, x, half);
                const auto [r1, m1] = double_dot(t, rows, width, r, half, h1, half);
                const double b = (m0 + m1) * 0x1p-24 * (3.0 + static_cast<double>(half) / 1024.0 + 5.0) + 1e-30;
                worst = std::max(worst, std::fabs(static_cast<double>(y0 + y1) - (r0 + r1)) / b);
            }
        }
        expect(worst <= 1.0, "product_fp8 within the rounding bound of the double dot (width " + std::to_string(width) + ", worst " +
                                 std::to_string(worst) + " of the bound)");
    }
    {
        // Extreme codes: 448 and -448 against x = +-127 everywhere, the largest |D| (32 * 229376 * 127 < 2^30).
        const std::size_t rows = 32, width = 1024;
        std::vector<std::uint8_t> t(fp8_tensor_bytes(rows, width), 0x7E);
        for (std::size_t i = rows * width; i < t.size(); ++i) t[i] = 127;
        for (std::size_t i = 0; i < width; i += 2) t[i] = 0xFE;  // row 0 alternates signs, the other rows are all +448
        A8Activation x;
        x.codes.assign(width, 127);
        x.deltas.assign(width / 32U, 1.0F);
        const float y = product_fp8(t.data() + width, t.data() + rows * width, x.codes.data(), x.deltas.data(), width);
        expect(y == 448.0F * 127.0F * static_cast<float>(width), "all +448 times +127: exact");
        const float z = product_fp8(t.data(), t.data() + rows * width, x.codes.data(), x.deltas.data(), width);
        expect(z == 0.0F, "alternating +-448 times +127: exactly zero");
    }

    // Names and classes.
    expect(fp8_original_name("blk.0.attn_q_a.weight") == std::optional<std::string>("layers.0.attn.wq_a"), "q_a name");
    expect(fp8_original_name("blk.39.attn_output_b.weight") == std::optional<std::string>("layers.39.attn.wo_b"), "o_b name");
    expect(fp8_original_name("blk.7.ffn_up_shexp.weight") == std::optional<std::string>("layers.7.ffn.shared_experts.w3"), "shared up name");
    expect(fp8_original_name("blk.7.ffn_gate_shexp.weight") == std::optional<std::string>("layers.7.ffn.shared_experts.w1"), "shared gate name");
    expect(fp8_original_name("blk.7.ffn_down_shexp.weight") == std::optional<std::string>("layers.7.ffn.shared_experts.w2"), "shared down name");
    expect(!fp8_original_name("output.weight") && !fp8_original_name("blk.1.engram_kv.weight") &&
               !fp8_original_name("blk.2.indexer.attn_q_b.weight"),
           "no native FP8 form for the head, Engram or the indexer");
    expect(fp8_site_class("blk.3.ffn_down_shexp.weight") == std::optional<std::string>("shexp") &&
               fp8_site_class("blk.3.attn_kv.weight") == std::optional<std::string>("kv"),
           "site classes");
    expect(parse_fp8_site_classes("all").size() == 6U && parse_fp8_site_classes("q_b,o_a,q_b").size() == 2U, "class lists");
    bool threw = false;
    try { (void)parse_fp8_site_classes("q_a,head"); } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "unknown class rejected");

    // Safetensors reader on a synthetic checkpoint: index form and directory form.
    {
        const auto dir = std::filesystem::temp_directory_path() / ("strix-ds41-fp8-test-" + std::to_string(std::random_device{}()));
        std::filesystem::create_directories(dir);
        const std::size_t rows = 64, width = 96;
        const auto t = random_fp8(rng, rows, width);
        const std::string codes(reinterpret_cast<const char*>(t.data()), rows * width);
        const std::string scales(reinterpret_cast<const char*>(t.data() + rows * width), t.size() - rows * width);
        std::string nan_codes = codes;
        nan_codes[17] = static_cast<char>(0x7F);
        std::string low_scales = scales;
        low_scales[3] = static_cast<char>(9);
        write_file(dir / "a.safetensors", safetensors({{"layers.0.attn.wq_a.weight", "F8_E4M3", {rows, width}, codes},
                                                       {"layers.0.attn.wq_a.scale", "F8_E8M0", {rows / 32U, width / 32U}, scales},
                                                       {"layers.0.attn.wkv.weight", "F8_E4M3", {rows, width}, nan_codes},
                                                       {"layers.0.attn.wkv.scale", "F8_E8M0", {rows / 32U, width / 32U}, scales}}));
        write_file(dir / "b.safetensors", safetensors({{"layers.0.attn.wo_a.weight", "F8_E4M3", {rows, width}, codes},
                                                       {"layers.0.attn.wo_a.scale", "F8_E8M0", {rows / 32U, width / 32U}, low_scales},
                                                       {"layers.0.attn.wq_b.weight", "BF16", {rows, width / 2U}, codes},
                                                       {"layers.0.attn.wq_b.scale", "F8_E8M0", {rows / 32U, width / 32U}, scales}}));
        const auto rejects = [&](SafetensorsCheckpoint& ck, const std::string& name, std::size_t r, std::size_t w) {
            try { (void)read_fp8_tensor(ck, name, r, w); } catch (const std::runtime_error&) { return true; }
            return false;
        };
        for (const bool indexed : {false, true}) {
            if (indexed)
                write_file(dir / "model.safetensors.index.json",
                           "{\"metadata\":{},\"weight_map\":{\"layers.0.attn.wq_a.weight\":\"a.safetensors\",\"layers.0.attn.wq_a.scale\":"
                           "\"a.safetensors\",\"layers.0.attn.wkv.weight\":\"a.safetensors\",\"layers.0.attn.wkv.scale\":\"a.safetensors\","
                           "\"layers.0.attn.wo_a.weight\":\"b.safetensors\",\"layers.0.attn.wo_a.scale\":\"b.safetensors\","
                           "\"layers.0.attn.wq_b.weight\":\"b.safetensors\",\"layers.0.attn.wq_b.scale\":\"b.safetensors\","
                           "\"layers.1.attn.wq_a.weight\":\"missing.safetensors\"}}");
            SafetensorsCheckpoint ck(dir);
            const auto got = read_fp8_tensor(ck, "layers.0.attn.wq_a", rows, width);
            const std::string which = indexed ? " (index)" : " (directory)";
            expect(got.bytes == t, "tensor read back in the device layout" + which);
            expect(rejects(ck, "layers.0.attn.wkv", rows, width), "NaN code rejected" + which);
            expect(rejects(ck, "layers.0.attn.wo_a", rows, width), "exponent below 10 rejected" + which);
            expect(rejects(ck, "layers.0.attn.wq_b", rows, width), "non-FP8 dtype rejected" + which);
            expect(rejects(ck, "layers.0.attn.wq_a", rows, width + 32U), "shape mismatch rejected" + which);
            expect(rejects(ck, "layers.1.attn.wq_a", rows, width), "missing shard rejected" + which);
        }
        std::filesystem::remove_all(dir);
    }

    // The original checkpoint, when given: every dense tensor of layer 0 reads and passes the check.
    if (const char* dir = std::getenv("STRIX_DS41_NATIVE_FP8"); dir != nullptr && *dir != 0) {
        SafetensorsCheckpoint ck(dir);
        struct Site { const char* model; std::size_t rows, width; };
        for (const Site s : {Site{"blk.0.attn_q_a.weight", 1280, 5120}, Site{"blk.0.attn_q_b.weight", 32768, 1280}, Site{"blk.0.attn_kv.weight", 512, 5120},
                             Site{"blk.0.attn_output_a.weight", 8192, 4096}, Site{"blk.0.attn_output_b.weight", 5120, 8192},
                             Site{"blk.0.ffn_gate_shexp.weight", 2304, 5120}, Site{"blk.0.ffn_up_shexp.weight", 2304, 5120},
                             Site{"blk.0.ffn_down_shexp.weight", 5120, 2304}}) {
            const auto name = *fp8_original_name(s.model);
            const auto t = read_fp8_tensor(ck, name, s.rows, s.width);
            unsigned lo = 255, hi = 0;
            for (std::size_t i = s.rows * s.width; i < t.bytes.size(); ++i) { lo = std::min<unsigned>(lo, t.bytes[i]); hi = std::max<unsigned>(hi, t.bytes[i]); }
            std::cout << "checkpoint " << name << " [" << s.rows << ", " << s.width << "] exponents " << lo << ".." << hi << '\n';
        }
    }

    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ds41 native fp8 tests passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
