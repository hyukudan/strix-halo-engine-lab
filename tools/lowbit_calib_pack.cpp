// Calibrated low-bit packs and per-matrix numerical diagnostics. See docs/engine/lowbit-calibration.md.
#include "strix/serve/json.hpp"
#include "strix/weight/admission.hpp"
#include <functional>

#if defined(STRIX_PACK_INKLING2_MINI)
#include "inkling2/inkling2_mini_pack_source.hpp"
#endif

#include "strix/weight/calib_moments.hpp"
#include "strix/weight/calib_provenance.hpp"
#include "strix/weight/host_lowbit.hpp"
#include "strix/weight/host_lowbit_gptq.hpp"
#include "strix/weight/lowbit_pack.hpp"
#include "strix/weight/pack_plan.hpp"
#include "strix/weight/pack_source.hpp"
#include "strix/weight/registry.hpp"
#include "strix/weight/source.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
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

Format parse_format(std::string_view fmt) {
    if (fmt == "q4") return Format::q4;
    if (fmt == "q6") return Format::q6;
    if (fmt == "sp4") return Format::sp4;
    if (fmt == "sp3") return Format::sp3;
    if (fmt == "sp2") return Format::sp2;
    throw std::invalid_argument("storage format must be q4, q6, sp4, sp3 or sp2");
}

bool is_sp(Format f) { return f == Format::sp4 || f == Format::sp3 || f == Format::sp2; }

bool feedback(Technique t) { return t == Technique::feedback || t == Technique::both; }

wei::FormatId storage_id(Format format, bool fitted) {
    if (format == Format::q4) return wei::kQ4G64WeightStorage;
    if (format == Format::q6) return wei::kQ6G64WeightStorage;
    if (format == Format::sp4 && !fitted) return wei::kSP4TWeightStorage;
    return wei::strixpack_format(format_name(format));
}

struct Rule {
    std::string substring;
    Format format;
};

struct Args {
    std::string model, out, report, dequant_out, plan, probe;
    std::vector<std::string> moments, eval, heldout;
    std::size_t moments_bytes = std::numeric_limits<std::size_t>::max();
    std::vector<Rule> rules;
    Technique technique = Technique::both;
    double damp = 0.01;
    unsigned threads = 0, proxy_stride = 1;
    bool compare = false, table_auto = false;
    bool inkling2_mini = false;
};

std::uint64_t integer(std::string_view text) {
    std::uint64_t value = 0;
    const auto end = text.data() + text.size();
    const auto result = std::from_chars(text.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end)
        throw std::invalid_argument("expected unsigned integer: " + std::string(text));
    return value;
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + std::string(arg));
            return argv[++i];
        };
        if (arg == "--model") a.model = value();
        else if (arg == "--inkling2-mini") a.inkling2_mini = true;
        else if (arg == "--moments") a.moments.push_back(value());
        else if (arg == "--eval") a.eval.push_back(value());
        else if (arg == "--plan") a.plan = value();
        else if (arg == "--probe-source") a.probe = value();
        else if (arg == "--moments-bytes") {
            const auto bytes = integer(value());
            if (bytes > std::numeric_limits<std::size_t>::max()) throw std::invalid_argument("moment byte limit overflows");
            a.moments_bytes = static_cast<std::size_t>(bytes);
        }
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
            const auto t = value();
            if (t != "fixed" && t != "auto") throw std::invalid_argument("--table must be fixed or auto");
            a.table_auto = t == "auto";
        } else if (arg == "--storage") {
            const auto v = value();
            const auto eq = v.rfind('=');
            if (eq == std::string::npos || eq == 0U) throw std::invalid_argument("--storage SUBSTRING=q4|q6|sp4|sp3|sp2");
            a.rules.push_back({v.substr(0, eq), parse_format(v.substr(eq + 1U))});
        } else throw std::invalid_argument("unknown argument " + std::string(arg));
    }
    if (!std::isfinite(a.damp) || a.damp < 0.0) throw std::invalid_argument("--damp must be finite and nonnegative");
    if (a.model.empty()) throw std::invalid_argument("need --model");
    if (!a.probe.empty()) {
        if (!a.plan.empty() || !a.rules.empty() || !a.out.empty() || !a.dequant_out.empty() || !a.report.empty() ||
            !a.moments.empty() || !a.eval.empty() || !a.heldout.empty() || a.compare)
            throw std::invalid_argument("--probe-source is a read-only standalone operation");
        return a;
    }
    if (a.rules.empty() == a.plan.empty())
        throw std::invalid_argument("need --storage rules or --plan");
    if (!a.eval.empty() && a.eval.size() != a.moments.size())
        throw std::invalid_argument("need one --eval file for each --moments file");
    if (a.moments.empty() && a.compare) throw std::invalid_argument("--compare requires --moments");
    if (!a.moments.empty() && a.heldout.empty())
        throw std::invalid_argument("moments require at least one --heldout gate manifest");
    if (a.threads == 0U) a.threads = std::max(1U, std::thread::hardware_concurrency());
    return a;
}

using ResolveSource = std::function<wei::PackMatrixSource(std::string_view)>;

void probe(const wei::PackCheckpoint& checkpoint, const std::string& name, const ResolveSource& resolve) {
    const auto* tensor = checkpoint.find(name);
    const auto source = resolve(name);
    std::vector<float> decoded;
    if (const auto status = source.decode(decoded); !status) throw std::runtime_error(status.detail);
    const auto digest = wei::digest_bytes(std::as_bytes(std::span(decoded)));
    std::cout << name << " dtype=" << (tensor ? tensor->wire_dtype : "microscaled") << " rows=" << source.rows << " columns=" << source.columns
              << " fp32_sha256=";
    constexpr char hex[] = "0123456789abcdef";
    for (const auto byte : digest.bytes) std::cout << hex[byte >> 4U] << hex[byte & 15U];
    std::cout << '\n';
}

struct Selection {
    std::string name;
    Format format;
    Technique technique;
    bool processed = false;
};

std::map<std::string, Selection, std::less<>> select(const Args& args, const wei::PackCheckpoint& checkpoint,
                                                     const ResolveSource& resolve) {
    std::map<std::string, Selection, std::less<>> selected;
    if (!args.plan.empty()) {
        std::ifstream in(args.plan);
        std::string line;
        if (!std::getline(in, line)) throw std::invalid_argument("cannot read " + args.plan);
        if (line.ends_with('\r')) line.pop_back();
        const bool coders = line == "tensor\tformat\tbytes\tcoder";
        if (!coders && line != "tensor\tformat\tbytes") throw std::invalid_argument("invalid pack plan header");
        std::set<std::string> names;
        while (std::getline(in, line)) {
            if (line.ends_with('\r')) line.pop_back();
            Technique technique = args.technique;
            if (coders) {
                const auto tab = line.rfind('\t');
                if (tab == std::string::npos) throw std::invalid_argument("missing pack plan coder field");
                if (tab + 1U != line.size()) technique = parse_technique(std::string_view(line).substr(tab + 1U));
                line.resize(tab);
            }
            const auto first = line.find('\t');
            const auto last = line.rfind('\t');
            if (first == 0 || first == last) throw std::invalid_argument("invalid pack plan row");
            const auto name = line.substr(0, first);
            const auto name_format = line.substr(first + 1U, last - first - 1U);
            if (!names.insert(name).second) throw std::invalid_argument("duplicate pack plan row for " + name);
            const auto bytes = integer(std::string_view(line).substr(last + 1U));
            if (name_format == "keep") continue;
            const auto source = resolve(name);
            if (bytes != wei::strixpack_matrix_bytes(wei::strixpack_format(name_format), source.rows, source.columns))
                throw std::invalid_argument("pack plan bytes disagree with geometry of " + name);
            selected.emplace(name, Selection{name, parse_format(name_format), technique});
        }
        if (!in.eof()) throw std::runtime_error("cannot finish reading " + args.plan);
    } else {
        for (const auto& tensor : checkpoint.tensors()) {
            constexpr std::string_view scale_suffix = "_scale_inv";
            if (tensor.name.ends_with(scale_suffix) &&
                checkpoint.find(std::string_view(tensor.name).substr(0, tensor.name.size() - scale_suffix.size())))
                continue;
            const auto rule = std::ranges::find_if(args.rules, [&](const Rule& r) {
                return tensor.name.find(r.substring) != std::string::npos;
            });
            if (rule != args.rules.end()) selected.emplace(tensor.name, Selection{tensor.name, rule->format, args.technique});
        }
    }
    if (selected.empty()) throw std::invalid_argument("no matrices selected");
    if (args.moments.empty() && std::ranges::any_of(selected, [](const auto& item) {
            return item.second.technique != Technique::rtn;
        })) throw std::invalid_argument("calibrated matrices require --moments");
    return selected;
}

struct Coded {
    std::vector<std::uint8_t> bytes;
    std::vector<float> dequant;
    std::string table = "-";  // SP: "fixed|auto:t0,t1,...;base=B;flags=F"
    std::optional<wei::sp::Header> header{};  // SP: the tensor header of the bytes
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
    if (feedback(t)) {
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
    if (feedback(t)) {
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
    c.header = h;
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
    using strix::serve::Json;
    auto descriptors = Json::object();
    std::uint64_t offset = 0;
    for (const auto& [name, v] : tensors) {
        const std::uint64_t bytes = v.second.size() * sizeof(std::uint16_t);
        descriptors.set(name, Json::Object{
            {"dtype", "BF16"},
            {"shape", Json::Array{std::uint64_t{v.first[0]}, std::uint64_t{v.first[1]}}},
            {"data_offsets", Json::Array{offset, offset + bytes}},
        });
        offset += bytes;
    }
    descriptors.set("__metadata__", Json::Object{{"format", "pt"}, {"producer", "strix-lowbit-calib-pack dequant"}});
    auto header = descriptors.dump();
    while (header.size() % 8U != 0U) header += ' ';
    if (std::filesystem::exists(std::filesystem::symlink_status(path)))
        throw std::invalid_argument("dequant output already exists: " + path);
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
        wei::PackCheckpoint checkpoint;
        if (const auto status = wei::PackCheckpoint::open(args.model, checkpoint); !status)
            throw std::runtime_error("index " + args.model + ": " + status.detail);
#if defined(STRIX_PACK_INKLING2_MINI)
        strix::inkling2::MiniTextWeights mini;
        if (args.inkling2_mini)
            if (const auto status = mini.open(args.model); !status) throw std::runtime_error(status.detail);
#else
        if (args.inkling2_mini) throw std::runtime_error("Inkling 2 Mini source adapter unavailable");
#endif
        const ResolveSource resolve = [&](std::string_view name) {
            wei::PackMatrixSource source;
#if defined(STRIX_PACK_INKLING2_MINI)
            if (args.inkling2_mini) {
                if (const auto status = strix::inkling2::mini_pack_source(checkpoint, mini, name, source); !status)
                    throw std::runtime_error(status.detail);
                return source;
            }
#endif
            const auto* tensor = checkpoint.find(name);
            if (!tensor) throw std::runtime_error("checkpoint lacks " + std::string(name));
            if (const auto status = wei::PackMatrixSource::bind(checkpoint, *tensor, source); !status)
                throw std::runtime_error(status.detail);
            return source;
        };
        if (!args.probe.empty()) {
            probe(checkpoint, args.probe, resolve);
            return 0;
        }
        auto selected = select(args, checkpoint, resolve);
        const bool report_sp = std::ranges::any_of(selected, [](const auto& item) { return is_sp(item.second.format); });
        // Diagnostics and the pack take new paths, including when their names
        // resolve through equivalent parent paths.
        std::set<std::filesystem::path> outputs;
        for (const auto* path : {&args.out, &args.report, &args.dequant_out})
            if (!path->empty() && (std::filesystem::exists(std::filesystem::symlink_status(*path)) ||
                                   !outputs.insert(std::filesystem::weakly_canonical(std::filesystem::absolute(*path))).second))
                throw std::invalid_argument("output already exists or aliases another output: " + *path);
        std::optional<wei::LowbitPackWriter> writer;
        if (!args.out.empty()) {
            std::vector<std::string_view> names;
            names.reserve(selected.size());
            for (const auto& item : selected) names.push_back(item.first);
            wei::Digest256 digest{};
            if (const auto status = checkpoint.digest(digest); !status)
                throw std::runtime_error("checkpoint digest: " + status.detail);
            writer.emplace(digest, args.out, names);
        }
        std::ofstream report;
        if (!args.report.empty()) {
            report.open(args.report, std::ios::trunc);
            if (!report) throw std::runtime_error("cannot open report " + args.report);
            report << "tensor\tinput\tcal_rows\teval_rows\tformat\ttechnique\trel_error\tselected";
            if (report_sp) report << "\ttable\tweight_sqnr_db";
            report << '\n';
        }
        DequantTensors dequant_out;
        double total_rtn = 0.0, total_sel = 0.0;
        const auto process = [&](Selection& selection, const wei::calib::InputMoments* input,
                                 const wei::calib::InputMoments* held, const std::vector<float>& importance,
                                 const std::vector<float>& factor, const std::vector<std::uint8_t>& dead) {
            const auto& name = selection.name;
            if (selection.processed) throw std::runtime_error("multiple moment inputs for " + name);
            const auto source = resolve(name);
            const auto n = source.rows, k = source.columns;
            if (input && (input->width != k || held->width != k))
                throw std::runtime_error(name + ": width differs from its input");
            const auto format = selection.format;
            const auto input_name = input ? inpuname : "-";
            const auto block = is_sp(format) ? wei::sp::SP2Storage::elements : wei::Q4G64Storage::elements;
            if (k % block != 0U) throw std::runtime_error(name + ": width is not a multiple of pack block");
            wei::Digest256 digest{};
            std::vector<float> w;
            if (const auto status = source.decode(w, writer ? &digest : nullptr); !status)
                throw std::runtime_error(name + ": " + status.detail);
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
            if (selection.technique != Technique::rtn) {
                if (args.report.empty() && !args.compare) arms.clear();
                arms.push_back({format, fit, selection.technique});
            }
            if (args.compare && !is_sp(format)) {
                for (const auto tech : {Technique::importance, Technique::feedback, Technique::both})
                    if (tech != selection.technique) arms.push_back({format, false, tech});
                const Format other = format == Format::q4 ? Format::q6 : Format::q4;
                arms.push_back({other, false, Technique::rtn});
                if (selection.technique != Technique::rtn) arms.push_back({other, false, selection.technique});
            } else if (args.compare) {  // SP: the other table mode and the incumbent q4_g64, rtn and the technique
                arms.push_back({format, !fit, Technique::rtn});
                if (selection.technique != Technique::rtn) arms.push_back({format, !fit, selection.technique});
                arms.push_back({Format::q4, false, Technique::rtn});
                if (selection.technique != Technique::rtn) arms.push_back({Format::q4, false, selection.technique});
            }
            for (const auto& a : arms) {
                const auto tech = a.tech;
                auto coded = arm(a);
                gq::OutputError err;
                if (held) {
                    err = gq::output_error(w.data(), coded.dequant.data(), n, k, held->h, args.threads,
                                           held->layout == wei::calib::MomentLayout::diagonal, args.proxy_stride);
                    if (!err.available())
                        throw std::runtime_error(name + ": proxy signal on input " + input_name +
                                                 " is zero or not finite, relative error unavailable");
                }
                const bool chosen = a.format == format && a.fit == fit && tech == selection.technique;
                if (a.format == format && a.fit == fit && tech == Technique::rtn) total_rtn += err.error;
                if (chosen) total_sel += err.error;
                if (!args.report.empty()) {
                    report << name << '\t' << input_name << '\t' << (input ? input->rows : 0U) << '\t' << (held ? held->rows : 0U) << '\t'
                           << format_name(a.format) << '\t' << technique_name(tech) << '\t' << err.relative() << '\t'
                           << (chosen ? 1 : 0);
                    if (report_sp) report << '\t' << coded.table << '\t' << weight_sqnr_db(w, coded.dequant);
                    report << '\n';
                }
                if (chosen && !args.dequant_out.empty()) {
                    std::vector<std::uint16_t> bf(coded.dequant.size());
                    for (std::size_t i = 0; i < bf.size(); ++i) bf[i] = exact_bf16(coded.dequant[i]);
                    dequant_out[name] = {{n, k}, std::move(bf)};
                }
                if (chosen && writer)
                    writer->add(name, storage_id(a.format, a.fit), n, k, digest, std::move(coded.bytes), coded.header);
            }
            selection.processed = true;
        };
        if (args.moments.empty()) {
            for (auto& item : selected) process(item.second, nullptr, nullptr, {}, {}, {});
        } else {
            wei::calib::Provenance heldout, all_cal, all_eval;
            for (const auto& manifest : args.heldout) heldout.add_manifest(manifest);
            for (std::size_t layer = 0; layer < args.moments.size(); ++layer) {
                const auto cal = wei::calib::MomentSet::read(args.moments[layer], args.moments_bytes);
                std::optional<wei::calib::MomentSet> eval;
                if (!args.eval.empty())
                    eval.emplace(wei::calib::MomentSet::read(args.eval[layer], args.moments_bytes - cal.moment_bytes()));
                wei::calib::require_separation(cal.provenance(), eval ? &eval->provenance() : nullptr, heldout);
                if (eval && args.moments.size() > 1U) {
                    all_cal.merge(cal.provenance());
                    all_eval.merge(eval->provenance());
                    wei::calib::require_separation(all_cal, &all_eval, heldout);
                }
                const auto& proxy = eval ? *eval : cal;
                std::cerr << "moments layer " << layer << " bytes " << cal.moment_bytes() + (eval ? eval->moment_bytes() : 0U)
                          << " proxy " << (proxy.layout() == wei::calib::MomentLayout::diagonal ? "diagonal" : "full") << '\n';
                for (const auto& input : cal.inputs()) {
                    std::vector<Selection*> inputs;
                    for (const auto& name : input.consumers) {
                        const auto it = selected.find(name);
                        if (it != selected.end()) inputs.push_back(&it->second);
                    }
                    if (inputs.empty()) continue;
                    const auto* held = proxy.find(input.name);
                    if (!held) throw std::runtime_error("evaluation statistics lack " + input.name);
                    wei::calib::require_rows(input, "calibration");
                    wei::calib::require_rows(*held, eval ? "evaluation" : "calibration");
                    const bool needs_factor = std::ranges::any_of(inputs, [&](const Selection* item) {
                        return feedback(item->technique) || (args.compare && !is_sp(item->format));
                    });
                    std::vector<float> factor;
                    std::vector<std::uint8_t> dead;
                    if (needs_factor) {
                        if (input.layout == wei::calib::MomentLayout::diagonal)
                            throw std::runtime_error(input.name + ": feedback requires full second moments");
                        factor = gq::inverse_factor(input.h, input.width, args.damp, dead, args.threads);
                    }
                    std::vector<float> importance(input.width);
                    for (std::size_t i = 0; i < input.width; ++i) {
                        const double energy = input.diagonal(i);
                        if (energy > std::numeric_limits<float>::max())
                            throw std::runtime_error("calibration importance does not fit FP32 for " + input.name);
                        importance[i] = static_cast<float>(energy);
                    }
                    for (auto* item : inputs) process(*item, &input, held, importance, factor, dead);
                }
            }
        }
        // Explicit RTN selections need no rows, even when other matrices use calibration.
        for (auto& [name, selection] : selected)
            if (!selection.processed && selection.technique == Technique::rtn && !args.compare)
                process(selection, nullptr, nullptr, {}, {}, {});
        for (const auto& [name, selection] : selected)
            if (!selection.processed)
                throw std::runtime_error(std::string(name) + " is selected but no statistics input lists it");
        if (!args.report.empty()) {
            report.close();
            if (!report) throw std::runtime_error("cannot finish report " + args.report);
        }
        if (writer)
            if (const auto status = writer->finish(); !status) throw std::runtime_error("write pack: " + status.detail);
        if (!args.dequant_out.empty()) write_dequant(args.dequant_out, dequant_out);
        std::cout << "strix-lowbit-calib-pack/v1 matrices " << selected.size() << " technique " << technique_name(args.technique)
                  << " damp " << args.damp << " output-error ";
        if (args.moments.empty()) std::cout << "unavailable";
        else std::cout << "rtn " << total_rtn << " selected " << total_sel << " ratio "
                       << (total_rtn > 0.0 ? total_sel / total_rtn : 0.0);
        std::cout << " seconds " << seconds_since(t0) << '\n';
    } catch (const std::exception& e) {
        std::cerr << "strix-lowbit-calib-pack: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
