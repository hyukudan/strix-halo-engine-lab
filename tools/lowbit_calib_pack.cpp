// Calibrated low-bit pack tool (host, any model): BF16 checkpoint matrices + calibration input statistics
// (strix/weight/calib_moments.hpp) -> a low-bit pack sidecar (strix/weight/lowbit_pack.hpp) whose q4_g64 / q6_g64 rows
// are coded with error feedback and / or importance-weighted clipping (strix/weight/host_lowbit_gptq.hpp), and a table
// of the relative output error of every tensor for every technique, measured on a second (evaluation) statistics
// file when one is given.
//
//   strix-lowbit-calib-pack --model MODEL.safetensors --moments CAL.moments [--eval EVAL.moments]
//       --storage SUBSTRING=q4|q6 [...] [--technique rtn|importance|feedback|both] [--damp D] [--threads N]
//       [--compare] [--proxy-stride S] [--out PACK] [--report FILE.tsv]
// --storage rules select the matrices (first rule whose substring occurs in the tensor name); every selected matrix
// must be a consumer of one statistics input. --compare also measures the other technique / format arms per tensor.
// --proxy-stride S measures the output error on every S-th output row only (the coded bytes are always complete).
#include "strix/weight/admission.hpp"
#include "strix/weight/calib_moments.hpp"
#include "strix/weight/host_lowbit.hpp"
#include "strix/weight/host_lowbit_gptq.hpp"
#include "strix/weight/lowbit_pack.hpp"
#include "strix/weight/registry.hpp"
#include "strix/weight/source.hpp"
#include "core/iu4_onfly.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

namespace wei = strix::weight;
namespace gq = strix::weight::lowbit::gptq;

enum class Technique { rtn, importance, feedback, both };

const char* technique_name(Technique t) {
    switch (t) {
        case Technique::rtn: return "rtn";
        case Technique::importance: return "importance";
        case Technique::feedback: return "feedback";
        case Technique::both: return "both";
    }
    return "?";
}

Technique parse_technique(std::string_view s) {
    if (s == "rtn") return Technique::rtn;
    if (s == "importance") return Technique::importance;
    if (s == "feedback") return Technique::feedback;
    if (s == "both") return Technique::both;
    throw std::invalid_argument("technique must be rtn|importance|feedback|both");
}

struct Rule {
    std::string substring;
    unsigned bits;
};

struct Args {
    std::string model, moments, eval, out, report;
    std::vector<Rule> rules;
    Technique technique = Technique::both;
    double damp = 0.01;
    unsigned threads = 0, proxy_stride = 1;
    bool compare = false;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + std::string(arg));
            return argv[++i];
        };
        if (arg == "--model") a.model = value();
        else if (arg == "--moments") a.moments = value();
        else if (arg == "--eval") a.eval = value();
        else if (arg == "--out") a.out = value();
        else if (arg == "--report") a.report = value();
        else if (arg == "--technique") a.technique = parse_technique(value());
        else if (arg == "--damp") a.damp = std::stod(value());
        else if (arg == "--threads") a.threads = static_cast<unsigned>(std::stoul(value()));
        else if (arg == "--compare") a.compare = true;
        else if (arg == "--proxy-stride") a.proxy_stride = std::max(1U, static_cast<unsigned>(std::stoul(value())));
        else if (arg == "--storage") {
            const auto v = value();
            const auto eq = v.rfind('=');
            if (eq == std::string::npos || eq == 0U) throw std::invalid_argument("--storage SUBSTRING=q4|q6");
            const auto fmt = v.substr(eq + 1U);
            if (fmt != "q4" && fmt != "q6") throw std::invalid_argument("--storage format must be q4 or q6");
            a.rules.push_back({v.substr(0, eq), fmt == "q4" ? 4U : 6U});
        } else throw std::invalid_argument("unknown argument " + std::string(arg));
    }
    if (a.model.empty() || a.moments.empty() || a.rules.empty())
        throw std::invalid_argument("need --model, --moments and at least one --storage rule");
    if (a.threads == 0U) a.threads = std::max(1U, std::thread::hardware_concurrency());
    return a;
}

std::vector<float> read_bf16(const wei::RawTensor& t) {
    if (t.wire_dtype != "BF16" || t.shape.rank != 2U) throw std::runtime_error(t.name + ": need a rank-2 BF16 matrix");
    std::vector<wei::Byte> raw(static_cast<std::size_t>(t.payload.range.bytes));
    if (const auto status = wei::read_slice(t.payload, raw); !status) throw std::runtime_error(t.name + ": " + status.detail);
    std::vector<float> w(raw.size() / 2U);
    for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = strix::core::onfly::bf16_bits_to_f32(static_cast<std::uint16_t>(
            std::to_integer<unsigned>(raw[2U * i]) | (std::to_integer<unsigned>(raw[2U * i + 1U]) << 8U)));
    return w;
}

struct Coded {
    std::vector<std::uint8_t> bytes;
    std::vector<float> dequant;
};

template <class Storage>
Coded code(const std::vector<float>& w, std::size_t n, std::size_t k, Technique t, const std::vector<float>& importance,
           const std::vector<float>& factor, const std::vector<std::uint8_t>& dead, unsigned threads) {
    Coded c;
    c.bytes.resize(n * Storage::row_bytes(k));
    c.dequant.resize(n * k);
    gq::Options o;
    o.threads = threads;
    o.dequant = c.dequant.data();
    if (t == Technique::importance || t == Technique::both) o.importance = importance.data();
    if (t == Technique::feedback || t == Technique::both) {
        o.factor = factor.data();
        o.dead = dead.data();
    }
    const auto status = gq::encode_rows<Storage>(w.data(), n, k, c.bytes.data(), o);
    if (!status) throw std::runtime_error(status.code == wei::lowbit::EncodeStatus::nonfinite ? "nonfinite weight"
                                                                                              : "group scale beyond FP16");
    return c;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse(argc, argv);
        const auto t0 = std::chrono::steady_clock::now();
        std::shared_ptr<const wei::Source> source;
        if (const auto status = wei::open_source(args.model, source); !status)
            throw std::runtime_error("open " + args.model + ": " + status.detail);
        wei::TensorIndex index;
        if (const auto status = wei::index_safetensors(source, index); !status)
            throw std::runtime_error("index " + args.model + ": " + status.detail);
        const auto cal = wei::calib::MomentSet::read(args.moments);
        std::optional<wei::calib::MomentSet> eval;
        if (!args.eval.empty()) eval.emplace(wei::calib::MomentSet::read(args.eval));
        std::cerr << "loaded index + moments in " << seconds_since(t0) << " s\n";

        std::ofstream report;
        if (!args.report.empty()) {
            report.open(args.report, std::ios::trunc);
            report << "tensor\tinput\tcal_rows\teval_rows\tformat\ttechnique\trel_error\tselected\n";
        }
        wei::LowbitPackWriter writer;
        std::map<std::string, std::vector<const wei::RawTensor*>> by_input;
        std::size_t selected = 0;
        for (const auto& t : index.tensors) {
            const Rule* rule = nullptr;
            for (const auto& r : args.rules)
                if (t.name.find(r.substring) != std::string::npos) {
                    rule = &r;
                    break;
                }
            if (!rule) continue;
            const auto* input = cal.find_consumer(t.name);
            if (!input) throw std::runtime_error(t.name + " is selected but no statistics input lists it");
            by_input[input->name].push_back(&t);
            ++selected;
        }
        std::cerr << selected << " matrices over " << by_input.size() << " inputs\n";
        double total_rtn = 0.0, total_sel = 0.0;
        for (const auto& [input_name, tensors] : by_input) {
            const auto* input = cal.find(input_name);
            const auto* held = eval ? eval->find(input_name) : input;
            if (!held) throw std::runtime_error("evaluation statistics lack " + input_name);
            const std::size_t k = input->width;
            const auto t1 = std::chrono::steady_clock::now();
            std::vector<std::uint8_t> dead;
            std::vector<float> factor = gq::inverse_factor(input->h, k, args.damp, dead, args.threads);
            std::vector<float> importance(k);
            for (std::size_t i = 0; i < k; ++i) importance[i] = static_cast<float>(input->h[i * k + i]);
            for (const auto* t : tensors) {
                const std::size_t n = t->shape.dims[0];
                if (t->shape.dims[1] != k) throw std::runtime_error(t->name + ": width differs from its input");
                const auto w = read_bf16(*t);
                unsigned bits = 0;
                for (const auto& r : args.rules)
                    if (t->name.find(r.substring) != std::string::npos) {
                        bits = r.bits;
                        break;
                    }
                const auto arm = [&](unsigned b, Technique tech) {
                    return b == 4U ? code<wei::Q4G64Storage>(w, n, k, tech, importance, factor, dead, args.threads)
                                   : code<wei::Q6G64Storage>(w, n, k, tech, importance, factor, dead, args.threads);
                };
                std::vector<std::pair<unsigned, Technique>> arms{{bits, Technique::rtn}};
                if (args.technique != Technique::rtn) arms.push_back({bits, args.technique});
                if (args.compare) {
                    for (const auto tech : {Technique::importance, Technique::feedback, Technique::both})
                        if (tech != args.technique) arms.push_back({bits, tech});
                    const unsigned other = bits == 4U ? 6U : 4U;
                    arms.push_back({other, Technique::rtn});
                    if (args.technique != Technique::rtn) arms.push_back({other, args.technique});
                }
                for (const auto& [b, tech] : arms) {
                    auto coded = arm(b, tech);
                    gq::OutputError err;
                    if (args.proxy_stride == 1U) {
                        err = gq::output_error(w.data(), coded.dequant.data(), n, k, held->h, args.threads);
                    } else {  // every proxy_stride-th output row
                        const std::size_t sampled = (n + args.proxy_stride - 1U) / args.proxy_stride;
                        std::vector<float> ws(sampled * k), qs(sampled * k);
                        for (std::size_t r = 0; r < sampled; ++r) {
                            const auto src = static_cast<std::ptrdiff_t>(r * args.proxy_stride * k);
                            std::copy(w.begin() + src, w.begin() + src + static_cast<std::ptrdiff_t>(k),
                                      ws.begin() + static_cast<std::ptrdiff_t>(r * k));
                            std::copy(coded.dequant.begin() + src, coded.dequant.begin() + src + static_cast<std::ptrdiff_t>(k),
                                      qs.begin() + static_cast<std::ptrdiff_t>(r * k));
                        }
                        err = gq::output_error(ws.data(), qs.data(), sampled, k, held->h, args.threads);
                    }
                    const bool chosen = b == bits && tech == args.technique;
                    if (b == bits && tech == Technique::rtn) total_rtn += err.error;
                    if (chosen) total_sel += err.error;
                    if (report)
                        report << t->name << '\t' << input_name << '\t' << input->rows << '\t' << held->rows << "\tq" << b
                               << '\t' << technique_name(tech) << '\t' << err.relative() << '\t' << (chosen ? 1 : 0) << '\n';
                    if (chosen && !args.out.empty()) {
                        wei::Digest256 fingerprint{};
                        if (const auto status = wei::lowbit_source_fingerprint(t->payload, fingerprint); !status)
                            throw std::runtime_error(t->name + ": " + status.detail);
                        writer.add(t->name, b == 4U ? wei::kQ4G64WeightStorage : wei::kQ6G64WeightStorage, n, k, fingerprint,
                                   std::move(coded.bytes));
                    }
                }
            }
            std::cerr << input_name << " (" << tensors.size() << " matrices, K " << k << ") " << seconds_since(t1) << " s\n";
        }
        if (!args.out.empty())
            if (const auto status = writer.write(args.out); !status) throw std::runtime_error("write pack: " + status.detail);
        std::cout << "strix-lowbit-calib-pack/v1 matrices " << selected << " technique " << technique_name(args.technique)
                  << " damp " << args.damp << " output-error rtn " << total_rtn << " selected " << total_sel << " ratio "
                  << (total_rtn > 0.0 ? total_sel / total_rtn : 0.0) << " seconds " << seconds_since(t0) << '\n';
    } catch (const std::exception& e) {
        std::cerr << "strix-lowbit-calib-pack: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
