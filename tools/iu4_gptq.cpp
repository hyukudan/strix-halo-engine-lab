// Error-compensated IU4 g64 copies of GGUF Q8_0 Linears from calibration input rows (any model; core/iu4_gptq.hpp).
// usage: strix-iu4-gptq PART.gguf [PART2.gguf ...] --tensor NAME [--tensor NAME2 ...] --calib X.f32 [--eval Y.f32]
//        [--eval-rows R] [--block B] [--damp 0.01] [--threads T] [--out FILE]   (every --tensor shares the calibration input)
// X / Y: FP32 input rows of that Linear (K floats per row), e.g. from a device tap. Prints, on the eval rows (else the
// calibration rows), the relative output error of RTN (the load-time converter's rule) and of the compensated copy, and
// checks that the RTN path equals onfly::convert_q8_0_host byte for byte. --out appends the compensated copy as one
// record (u32 name length, name, u64 bytes, packed then FP16 scales) for a diagnostic loader. Calibration rows and
// Hessians stay in memory only.
#include "core/iu4_gptq.hpp"
#include "strix/gguf_reader.hpp"

#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::vector<float> read_rows(const std::string& path, std::size_t k) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + path);
    const auto bytes = static_cast<std::size_t>(in.tellg());
    if (bytes == 0 || bytes % (k * 4U)) throw std::runtime_error(path + ": not rows of K floats");
    std::vector<float> v(bytes / 4U);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes));
    return v;
}

int run(int argc, char** argv) {
    std::vector<std::filesystem::path> parts;
    std::vector<std::string> tensors;
    std::string calib, eval, out;
    std::size_t eval_rows = 128, block = 0;
    double damp = 0.01;
    unsigned threads = std::max(1U, std::thread::hardware_concurrency());
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--tensor" && more) tensors.emplace_back(argv[++i]);
        else if (a == "--eval-rows" && more) eval_rows = std::stoul(argv[++i]);
        else if (a == "--block" && more) block = std::stoul(argv[++i]);
        else if (a == "--calib" && more) calib = argv[++i];
        else if (a == "--eval" && more) eval = argv[++i];
        else if (a == "--out" && more) out = argv[++i];
        else if (a == "--damp" && more) damp = std::stod(argv[++i]);
        else if (a == "--threads" && more) threads = static_cast<unsigned>(std::stoul(argv[++i]));
        else if (!a.empty() && a[0] != '-') parts.emplace_back(a);
        else throw std::invalid_argument("unknown argument " + a);
    }
    if (parts.empty() || tensors.empty() || calib.empty())
        throw std::invalid_argument("usage: strix-iu4-gptq PART.gguf... --tensor NAME --calib X.f32 [--eval Y.f32] [--damp D] [--threads T] [--out F]");
    namespace co = strix::core;
    std::vector<strix::gguf::FileInfo> infos;
    for (const auto& p : parts) infos.push_back(strix::gguf::read_file(p));
    std::vector<float> xc, xe;
    std::vector<float> u;
    std::vector<std::uint8_t> dead;
    std::size_t kc = 0, rc = 0, re = 0, ndead = 0;
    bool all_same = true;
    for (const auto& tensor : tensors) {
        std::vector<std::uint8_t> q8;
        std::size_t k = 0, n = 0;
        for (std::size_t pi = 0; pi < parts.size(); ++pi)
            for (const auto& t : infos[pi].tensors) {
                if (t.name != tensor) continue;
                if (t.ggml_type != 8U || t.dimensions.size() != 2U) throw std::runtime_error(tensor + ": not a 2-D Q8_0 tensor");
                k = static_cast<std::size_t>(t.dimensions[0]);
                n = static_cast<std::size_t>(t.dimensions[1]);
                q8.resize(static_cast<std::size_t>(t.byte_size));
                std::ifstream in(parts[pi], std::ios::binary);
                in.seekg(static_cast<std::streamoff>(t.absolute_offset));
                in.read(reinterpret_cast<char*>(q8.data()), static_cast<std::streamsize>(q8.size()));
                if (!in) throw std::runtime_error(tensor + ": read failed");
            }
        if (q8.empty()) throw std::runtime_error(tensor + ": not found");
        if (k % 64U || q8.size() != n * co::onfly::q8_row_bytes(k)) throw std::runtime_error(tensor + ": K % 64 / size");
        const auto t0 = std::chrono::steady_clock::now();
        if (kc != k) {  // one factor per input width (every --tensor shares the calibration rows)
            if (kc != 0) throw std::runtime_error("--tensor widths differ: one calibration input per call");
            xc = read_rows(calib, k);
            xe = eval.empty() ? xc : read_rows(eval, k);
            rc = xc.size() / k;
            re = std::min(xe.size() / k, eval_rows);
            if (block) {
                u = co::gptq::block_inverse_factor(xc.data(), rc, k, block, damp, dead, threads);
            } else {
                std::vector<double> h;
                co::gptq::accumulate_hessian(xc.data(), rc, k, h);
                u = co::gptq::inverse_factor(h, k, damp, dead, threads);
            }
            ndead = 0;
            for (const auto d : dead) ndead += d;
            kc = k;
        }
        const auto t1 = std::chrono::steady_clock::now();
        std::vector<float> w(n * k);
        for (std::size_t c = 0; c < n; ++c)
            for (std::size_t g = 0; g < k / 64U; ++g)
                co::onfly::dequant_q8_group(q8.data() + c * co::onfly::q8_row_bytes(k), static_cast<unsigned>(g), w.data() + c * k + g * 64U);
        std::vector<float> wr, wg;
        const auto rule = co::iu4_scale_rule_mse;
        const auto rtn = co::gptq::quantise(w.data(), n, k, {}, {}, rule, &wr, threads);
        const auto cmp = co::onfly::convert_q8_0_host(q8.data(), n, k, rule);
        const bool same = rtn.packed == cmp.packed && rtn.scales == cmp.scales;
        all_same = all_same && same;
        const auto gq = co::gptq::quantise(w.data(), n, k, u, dead, rule, &wg, threads, block);
        const auto t2 = std::chrono::steady_clock::now();
        const double er = co::gptq::relative_output_error(w.data(), wr.data(), n, k, xe.data(), re, threads);
        const double eg = co::gptq::relative_output_error(w.data(), wg.data(), n, k, xe.data(), re, threads);
        double wre = 0, wge = 0, wn = 0;
        for (std::size_t i = 0; i < w.size(); ++i) {
            wre += double(wr[i] - w[i]) * (wr[i] - w[i]);
            wge += double(wg[i] - w[i]) * (wg[i] - w[i]);
            wn += double(w[i]) * w[i];
        }
        const auto sec = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
        std::printf("gptq %s N=%zu K=%zu calib_rows=%zu eval_rows=%zu%s dead=%zu damp=%.3g factor_s=%.1f quant_s=%.1f | rel_out_err rtn %.6f gptq %.6f "
                    "ratio %.3f | rel_w_err rtn %.5f gptq %.5f | rtn==convert_q8_0_host %s\n",
                    tensor.c_str(), n, k, rc, re, eval.empty() ? "(=calib)" : "", ndead, damp, sec(t0, t1), sec(t1, t2), er, eg,
                    eg > 0 ? er / eg : 0.0, std::sqrt(wre / wn), std::sqrt(wge / wn), same ? "yes" : "NO");
        std::fflush(stdout);
        if (!out.empty()) {
            std::ofstream o(out, std::ios::binary | std::ios::app);
            const auto len = static_cast<std::uint32_t>(tensor.size());
            const std::uint64_t bytes = gq.packed.size() + gq.scales.size() * 2U;
            o.write(reinterpret_cast<const char*>(&len), 4);
            o.write(tensor.data(), static_cast<std::streamsize>(len));
            o.write(reinterpret_cast<const char*>(&bytes), 8);
            o.write(reinterpret_cast<const char*>(gq.packed.data()), static_cast<std::streamsize>(gq.packed.size()));
            o.write(reinterpret_cast<const char*>(gq.scales.data()), static_cast<std::streamsize>(gq.scales.size() * 2U));
            if (!o) throw std::runtime_error("cannot write " + out);
        }
    }
    return all_same ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "strix-iu4-gptq: " << e.what() << "\n";
        return 1;
    }
}
