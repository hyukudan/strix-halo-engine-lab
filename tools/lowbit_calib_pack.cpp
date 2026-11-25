// Calibrated low-bit pack tool (host, any model): BF16 checkpoint matrices + calibration input statistics
// (strix/weight/calib_moments.hpp) -> a low-bit pack sidecar (strix/weight/lowbit_pack.hpp) whose q4_g64 / q6_g64 rows
// are coded with error feedback and / or importance-weighted clipping (strix/weight/host_lowbit_gptq.hpp), and a table
// of the relative output error of every tensor for every technique, measured on a second (evaluation) statistics
// file when one is given.
//
//   strix-lowbit-calib-pack --model MODEL.safetensors --moments CAL.moments --heldout GATE_MANIFEST [...]
//       [--eval EVAL.moments] --storage SUBSTRING=q4|q6|sp4|sp3|sp2 [...] [--technique rtn|importance|feedback|both]
//       [--damp D] [--table fixed|auto] [--threads N] [--compare] [--proxy-stride S] [--out PACK] [--report FILE.tsv]
//       [--dequant-out FILE.safetensors]
// Held-out separation is enforced, not documented: the statistics must carry their window provenance
// (strix/weight/calib_provenance.hpp), and the tool refuses them when a calibration window or any run of 32 ids is
// shared with a gate manifest named by --heldout (every manifest the qualification uses); the --eval statistics must carry
// provenance too and share nothing with the calibration windows or the held-out manifests.
// The pack is bound to the full SHA-256 of the checkpoint file and of every packed tensor's BF16 payload.
// --storage rules select the matrices (first rule whose substring occurs in the tensor name); every selected matrix
// must be a consumer of one statistics input. --compare also measures the other technique / format arms per tensor.
// --proxy-stride S measures the output error on every S-th output row only (the coded bytes are always complete).
// Strix pack rules (sp4 / sp3 / sp2, strix/weight/host_lowbit.hpp sp::) code the matrix on the SP grid with the default
// table (--table fixed) or a per-tensor fitted table (--table auto: sp::fit_table on a row sample, importance-weighted
// when the technique uses importance); every coded superblock is decoded again and must equal the coder's own decode.
// SP has no device consumer yet, so --out refuses SP rules; --dequant-out writes the decoded selected matrices as a
// BF16 safetensors file (SP weights are exact in BF16; the tool refuses a value that is not) for an emulation through
// the existing BF16 routes. The report gives the relative output error and the weight-space SQNR of every arm.
#include "strix/weight/admission.hpp"
#include "strix/weight/calib_moments.hpp"
#include "strix/weight/calib_provenance.hpp"
#include "strix/weight/host_lowbit.hpp"
#include "strix/weight/host_lowbit_gptq.hpp"
#include "strix/weight/lowbit_pack.hpp"
#include "strix/weight/registry.hpp"
#include "strix/weight/source.hpp"
#include "core/iu4_onfly.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
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

enum class Format { q4, q6, sp4, sp3, sp2 };

const char* format_name(Format f) {
    switch (f) {
        case Format::q4: return "q4";
        case Format::q6: return "q6";
        case Format::sp4: return "sp4";
        case Format::sp3: return "sp3";
        case Format::sp2: return "sp2";
    }
    return "?";
}

bool is_sp(Format f) { return f == Format::sp4 || f == Format::sp3 || f == Format::sp2; }

struct Rule {
    std::string substring;
    Format format;
};

struct Args {
    std::string model, moments, eval, out, report, dequant_out;
    std::vector<std::string> heldout;
    std::vector<Rule> rules;
    Technique technique = Technique::both;
    double damp = 0.01;
    unsigned threads = 0, proxy_stride = 1;
    bool compare = false, table_auto = false;
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
        else if (arg == "--heldout") a.heldout.push_back(value());
        else if (arg == "--out") a.out = value();
        else if (arg == "--report") a.report = value();
        else if (arg == "--technique") a.technique = parse_technique(value());
        else if (arg == "--damp") a.damp = std::stod(value());
        else if (arg == "--threads") a.threads = static_cast<unsigned>(std::stoul(value()));
        else if (arg == "--compare") a.compare = true;
        else if (arg == "--proxy-stride") a.proxy_stride = std::max(1U, static_cast<unsigned>(std::stoul(value())));
        else if (arg == "--dequant-out") a.dequant_out = value();
        else if (arg == "--table") {
            const std::string_view t = value();
            if (t != "fixed" && t != "auto") throw std::invalid_argument("--table must be fixed or auto");
            a.table_auto = t == "auto";
        } else if (arg == "--storage") {
            const auto v = value();
            const auto eq = v.rfind('=');
            if (eq == std::string::npos || eq == 0U) throw std::invalid_argument("--storage SUBSTRING=q4|q6|sp4|sp3|sp2");
            const auto fmt = v.substr(eq + 1U);
            Format f{};
            if (fmt == "q4") f = Format::q4;
            else if (fmt == "q6") f = Format::q6;
            else if (fmt == "sp4") f = Format::sp4;
            else if (fmt == "sp3") f = Format::sp3;
            else if (fmt == "sp2") f = Format::sp2;
            else throw std::invalid_argument("--storage format must be q4, q6, sp4, sp3 or sp2");
            a.rules.push_back({v.substr(0, eq), f});
        } else throw std::invalid_argument("unknown argument " + std::string(arg));
    }
    if (a.model.empty() || a.moments.empty() || a.rules.empty() || a.heldout.empty())
        throw std::invalid_argument("need --model, --moments, at least one --heldout gate manifest and one --storage rule");
    if (!a.out.empty())
        for (const auto& r : a.rules)
            if (is_sp(r.format))
                throw std::invalid_argument("--out: a Strix pack rule has no device consumer yet (use --dequant-out)");
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
    std::string table = "-";  // SP: "fixed|auto:t0,t1,...;base=B;flags=F"
};

double weight_sqnr_db(const std::vector<float>& w, const std::vector<float>& q) {
    double sig = 0.0, err = 0.0;
    for (std::size_t i = 0; i < w.size(); ++i) {
        const double a = static_cast<double>(w[i]), d = a - static_cast<double>(q[i]);
        sig += a * a;
        err += d * d;
    }
    return err > 0.0 ? 10.0 * std::log10(sig / err) : 999.0;
}

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

template <class S>
Coded code_sp(const std::vector<float>& w, std::size_t n, std::size_t k, Technique t, bool fit,
              const std::vector<float>& importance, const std::vector<float>& factor, const std::vector<std::uint8_t>& dead,
              unsigned threads) {
    const bool weighted = t == Technique::importance || t == Technique::both;
    auto table = wei::sp::default_table<S::bits>();
    if (fit) {  // a sample of about 2^21 weights (every stride-th row)
        const std::size_t stride = std::max<std::size_t>(1U, n * k / (std::size_t{1} << 21U));
        table = wei::sp::fit_table<S>(w.data(), n, k, stride, weighted ? importance.data() : nullptr);
    }
    auto h = wei::sp::make_header<S>(w.data(), n, k, table);
    Coded c;
    c.bytes.resize(n * S::row_bytes(k));
    c.dequant.resize(n * k);
    gq::Options o;
    o.threads = threads;
    o.dequant = c.dequant.data();
    if (weighted) o.importance = importance.data();
    if (t == Technique::feedback || t == Technique::both) {
        o.factor = factor.data();
        o.dead = dead.data();
    }
    if (const auto status = gq::encode_rows_sp<S>(h, w.data(), n, k, c.bytes.data(), o); !status)
        throw std::runtime_error("nonfinite weight");
    wei::sp::mark_foldable<S>(h, c.bytes.data(), n, k);
    if (const char* why = h.invalid(); *why != '\0') throw std::runtime_error(std::string("Strix pack header: ") + why);
    // Round trip: the stored bytes decode to the coder's values, element for element.
    const std::size_t blocks = k / S::elements;
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t b = 0; b < blocks; ++b)
            for (unsigned j = 0; j < S::elements; ++j) {
                const float v = wei::sp::decode<S>(c.bytes.data() + (r * blocks + b) * S::bytes, j, h);
                if (std::memcmp(&v, &c.dequant[r * k + b * S::elements + j], sizeof v) != 0)
                    throw std::runtime_error("Strix pack bytes do not decode to the coded values");
            }
    std::string text = fit ? "auto:" : "fixed:";
    for (unsigned i = 0; i < S::levels; ++i) text += (i ? "," : "") + std::to_string(h.table[i]);
    text += ";base=" + std::to_string(int{h.base}) + ";flags=" + std::to_string(unsigned{h.flags});
    c.table = std::move(text);
    return c;
}

// BF16 bits of a value that BF16 holds exactly; throws otherwise.
std::uint16_t exact_bf16(float v) {
    std::uint32_t u = 0;
    std::memcpy(&u, &v, sizeof u);
    if ((u & 0xffffU) != 0U) throw std::runtime_error("--dequant-out: a decoded weight is not exact in BF16");
    return static_cast<std::uint16_t>(u >> 16U);
}

using DequantTensors = std::map<std::string, std::pair<std::array<std::size_t, 2>, std::vector<std::uint16_t>>>;

// Selected matrices' decoded weights as one BF16 safetensors file (sorted names, header padded to 8 bytes).
void write_dequant(const std::string& path, const DequantTensors& tensors) {
    std::string header = "{";
    std::size_t offset = 0;
    for (const auto& [name, v] : tensors) {
        const std::size_t bytes = v.second.size() * 2U;
        header += "\"" + name + "\":{\"dtype\":\"BF16\",\"shape\":[" + std::to_string(v.first[0]) + "," +
                  std::to_string(v.first[1]) + "],\"data_offsets\":[" + std::to_string(offset) + "," +
                  std::to_string(offset + bytes) + "]},";
        offset += bytes;
    }
    header += "\"__metadata__\":{\"format\":\"pt\",\"producer\":\"strix-lowbit-calib-pack dequant\"}}";
    while (header.size() % 8U != 0U) header += ' ';
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot open " + path);
    const std::uint64_t len = header.size();
    char prefix[8];
    for (unsigned i = 0; i < 8U; ++i) prefix[i] = static_cast<char>((len >> (8U * i)) & 0xffU);
    f.write(prefix, 8);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto& [name, v] : tensors) {
        std::vector<char> raw(v.second.size() * 2U);
        for (std::size_t i = 0; i < v.second.size(); ++i) {
            raw[2U * i] = static_cast<char>(v.second[i] & 0xffU);
            raw[2U * i + 1U] = static_cast<char>(v.second[i] >> 8U);
        }
        f.write(raw.data(), static_cast<std::streamsize>(raw.size()));
    }
    f.close();
    if (!f) throw std::runtime_error("cannot finish " + path);
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
        // Separation: calibration windows vs every held-out gate case, and vs the proxy evaluation windows.
        wei::calib::Provenance heldout;
        for (const auto& manifest : args.heldout) heldout.add_manifest(manifest);
        wei::calib::require_separation(cal.provenance(), eval ? &eval->provenance() : nullptr, heldout);
        wei::Digest256 checkpoint{};
        if (const auto status = wei::digest_slice({source, {0, source->size()}}, checkpoint); !status)
            throw std::runtime_error("digest " + args.model + ": " + status.detail);
        std::cerr << "loaded index + moments (" << cal.provenance().windows().size() << " calibration windows, "
                  << heldout.windows().size() << " held-out windows, no overlap) + checkpoint digest in "
                  << seconds_since(t0) << " s\n";

        std::ofstream report;
        if (!args.report.empty()) {
            report.open(args.report, std::ios::trunc);
            if (!report) throw std::runtime_error("cannot open report " + args.report);
            report << "tensor\tinput\tcal_rows\teval_rows\tformat\ttechnique\trel_error\tselected\ttable\tweight_sqnr_db\n";
        }
        wei::LowbitPackWriter writer(checkpoint);
        DequantTensors dequant_out;
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
            wei::calib::require_rows(*input, "calibration");
            wei::calib::require_rows(*held, eval ? "evaluation" : "calibration");
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
                Format format = Format::q4;
                for (const auto& r : args.rules)
                    if (t->name.find(r.substring) != std::string::npos) {
                        format = r.format;
                        break;
                    }
                struct Arm {
                    Format format;
                    bool fit;
                    Technique tech;
                };
                const auto arm = [&](const Arm& a) {
                    switch (a.format) {
                        case Format::q4: return code<wei::Q4G64Storage>(w, n, k, a.tech, importance, factor, dead, args.threads);
                        case Format::q6: return code<wei::Q6G64Storage>(w, n, k, a.tech, importance, factor, dead, args.threads);
                        case Format::sp4:
                            return code_sp<wei::sp::SP4Storage>(w, n, k, a.tech, a.fit, importance, factor, dead, args.threads);
                        case Format::sp3:
                            return code_sp<wei::sp::SP3Storage>(w, n, k, a.tech, a.fit, importance, factor, dead, args.threads);
                        case Format::sp2:
                            return code_sp<wei::sp::SP2Storage>(w, n, k, a.tech, a.fit, importance, factor, dead, args.threads);
                    }
                    throw std::logic_error("format");
                };
                const bool fit = is_sp(format) && args.table_auto;
                std::vector<Arm> arms{{format, fit, Technique::rtn}};
                if (args.technique != Technique::rtn) arms.push_back({format, fit, args.technique});
                if (args.compare && !is_sp(format)) {
                    for (const auto tech : {Technique::importance, Technique::feedback, Technique::both})
                        if (tech != args.technique) arms.push_back({format, false, tech});
                    const Format other = format == Format::q4 ? Format::q6 : Format::q4;
                    arms.push_back({other, false, Technique::rtn});
                    if (args.technique != Technique::rtn) arms.push_back({other, false, args.technique});
                } else if (args.compare) {  // SP: the other table mode and the incumbent q4_g64, rtn and the technique
                    arms.push_back({format, !fit, Technique::rtn});
                    if (args.technique != Technique::rtn) arms.push_back({format, !fit, args.technique});
                    arms.push_back({Format::q4, false, Technique::rtn});
                    if (args.technique != Technique::rtn) arms.push_back({Format::q4, false, args.technique});
                }
                for (const auto& a : arms) {
                    const auto tech = a.tech;
                    auto coded = arm(a);
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
                    if (!err.available())
                        throw std::runtime_error(t->name + ": proxy signal on input " + input_name +
                                                 " is zero or not finite, relative error unavailable");
                    const bool chosen = a.format == format && a.fit == fit && tech == args.technique;
                    if (a.format == format && a.fit == fit && tech == Technique::rtn) total_rtn += err.error;
                    if (chosen) total_sel += err.error;
                    if (!args.report.empty())
                        report << t->name << '\t' << input_name << '\t' << input->rows << '\t' << held->rows << '\t'
                               << format_name(a.format) << '\t' << technique_name(tech) << '\t' << err.relative() << '\t'
                               << (chosen ? 1 : 0) << '\t' << coded.table << '\t' << weight_sqnr_db(w, coded.dequant) << '\n';
                    if (chosen && !args.dequant_out.empty()) {
                        std::vector<std::uint16_t> bf(coded.dequant.size());
                        for (std::size_t i = 0; i < bf.size(); ++i) bf[i] = exact_bf16(coded.dequant[i]);
                        dequant_out[t->name] = {{n, k}, std::move(bf)};
                    }
                    if (chosen && !args.out.empty()) {
                        wei::Digest256 digest{};
                        if (const auto status = wei::digest_slice(t->payload, digest); !status)
                            throw std::runtime_error(t->name + ": " + status.detail);
                        writer.add(t->name, a.format == Format::q4 ? wei::kQ4G64WeightStorage : wei::kQ6G64WeightStorage, n, k, digest,
                                   std::move(coded.bytes));
                    }
                }
            }
            std::cerr << input_name << " (" << tensors.size() << " matrices, K " << k << ") " << seconds_since(t1) << " s\n";
        }
        if (!args.report.empty()) {
            report.close();
            if (!report) throw std::runtime_error("cannot finish report " + args.report);
        }
        if (!args.out.empty())
            if (const auto status = writer.write(args.out); !status) throw std::runtime_error("write pack: " + status.detail);
        if (!args.dequant_out.empty()) write_dequant(args.dequant_out, dequant_out);
        std::cout << "strix-lowbit-calib-pack/v1 matrices " << selected << " technique " << technique_name(args.technique)
                  << " damp " << args.damp << " output-error rtn " << total_rtn << " selected " << total_sel << " ratio "
                  << (total_rtn > 0.0 ? total_sel / total_rtn : 0.0) << " seconds " << seconds_since(t0) << '\n';
    } catch (const std::exception& e) {
        std::cerr << "strix-lowbit-calib-pack: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
