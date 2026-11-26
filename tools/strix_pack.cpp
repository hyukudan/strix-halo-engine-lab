#include "strix/weight/lowbit_pack.hpp"
#include "strix/weight/pack_source.hpp"
#include "strix/serve/json.hpp"
#include "strix/weight/host_lowbit_gptq.hpp"
#include "strix/weight/pack_plan.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace {

namespace wei = strix::weight;
namespace sp = wei::sp;
namespace ms = wei::microscaled;
using strix::serve::Json;
using Clock = std::chrono::steady_clock;

void require(wei::Status status, std::string_view context) {
    if (!status) throw std::runtime_error(std::string(context) + ": " + status.detail);
}

const Json& field(const Json& object, std::string_view key) {
    if (const auto* value = object.find(key)) return *value;
    throw std::invalid_argument("description needs " + std::string(key));
}

void allow_fields(const Json& object, std::initializer_list<std::string_view> keys) {
    for (const auto& [key, value] : object.as_object()) {
        (void)value;
        if (std::ranges::find(keys, key) == keys.end())
            throw std::invalid_argument("unknown description field " + key);
    }
}

struct Rule {
    std::string role, prefix, suffix, reason;
    wei::FormatId format{};
    bool bank = false;
    ms::RowOrder order = ms::RowOrder::contiguous;
    bool matched = false;
};

std::vector<Rule> description(const std::string& path) {
    constexpr std::size_t limit = 1U << 20U;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open description " + path);
    std::string text(limit + 1U, '\0');
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!in.eof() || in.bad()) throw std::runtime_error("description exceeds 1 MiB or cannot be read");
    text.resize(static_cast<std::size_t>(in.gcount()));
    const auto root = strix::serve::parse_json_unique(text);
    allow_fields(root, {"schema", "roles", "rules"});
    if (field(root, "schema").as_string() != "strix-pack-convert/v1")
        throw std::invalid_argument("unsupported conversion description version (expected strix-pack-convert/v1)");
    std::map<std::string, wei::FormatId, std::less<>> formats;
    for (const auto& [role, format] : field(root, "roles").as_object()) {
        if (role.empty() || role.size() > 64U || !std::ranges::all_of(role, [](char c) {
                return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
            })) throw std::invalid_argument("invalid role " + role);
        const auto name = format.as_string();
        const auto storage = name == "keep" ? wei::FormatId{} : wei::strixpack_format(name);
        formats.emplace(role, storage);
    }
    std::vector<Rule> rules;
    for (const auto& item : field(root, "rules").as_array()) {
        allow_fields(item, {"role", "prefix", "suffix", "layout", "reason"});
        Rule rule;
        rule.role = field(item, "role").as_string();
        const auto it = formats.find(rule.role);
        if (it == formats.end()) throw std::invalid_argument("undefined format for role " + rule.role);
        rule.format = it->second;
        if (const auto* prefix = item.find("prefix")) rule.prefix = prefix->as_string();
        if (const auto* suffix = item.find("suffix")) rule.suffix = suffix->as_string();
        if (rule.prefix.empty() && rule.suffix.empty()) throw std::invalid_argument("rule needs a prefix or suffix");
        if (const auto* reason = item.find("reason")) rule.reason = reason->as_string();
        if (const auto* layout = item.find("layout")) {
            rule.bank = true;
            if (layout->as_string() == "expert-interleaved") rule.order = ms::RowOrder::interleaved_halves;
            else if (layout->as_string() != "expert-contiguous") throw std::invalid_argument("unknown source layout");
        }
        rules.push_back(std::move(rule));
    }
    return rules;
}

struct Job {
    std::string name, role;
    wei::PackMatrixSource source;
    wei::FormatId format;
    std::uint64_t bytes;
};

struct RoleTotals { std::uint64_t source{}, compact{}, retained{}; };

void add(std::uint64_t& total, std::uint64_t bytes) {
    if (bytes > UINT64_MAX - total) throw std::invalid_argument("pack plan byte count overflows");
    total += bytes;
}

struct Conversion {
    std::vector<Job> jobs;
    std::map<std::string, RoleTotals, std::less<>> roles;
    std::map<std::string_view, std::pair<std::string, std::string>> retained;
    std::uint64_t compact{}, kept{}, scratch{};
};

Conversion plan(const wei::PackCheckpoint& checkpoint, std::vector<Rule> rules) {
    Conversion result;
    std::set<std::string_view> replaced;
    std::set<std::string, std::less<>> names;
    std::map<std::string_view, const Rule*> keep;
    const auto replace = [&](const wei::RawTensor& tensor, const Rule& rule, std::string_view sidecar = {}) {
        const auto* raw = sidecar.empty() ? &tensor : checkpoint.find(tensor.name + std::string(sidecar));
        if (raw && !replaced.insert(raw->name).second)
            throw std::invalid_argument("tensor or scale plane selected twice: " + raw->name);
        if (raw) add(result.roles[rule.role].source, raw->payload.range.bytes);
    };
    for (const auto& tensor : checkpoint.tensors()) {
        Rule* match = nullptr;
        for (auto& rule : rules) {
            if (!tensor.name.starts_with(rule.prefix) || !tensor.name.ends_with(rule.suffix)) continue;
            if (match) throw std::invalid_argument("ambiguous role for " + tensor.name);
            match = &rule;
        }
        if (!match) continue;
        match->matched = true;
        if (match->format.name.empty()) {
            keep.emplace(tensor.name, match);
            continue;
        }
        const auto emit = [&](std::string name, wei::PackMatrixSource source) {
            std::uint64_t bytes = 0;
            try { bytes = wei::strixpack_matrix_bytes(match->format, source.rows, source.columns); }
            catch (const std::exception& e) { throw std::invalid_argument(name + ": " + e.what()); }
            auto scratch = source.codes.range.bytes;
            add(scratch, source.scales.range.bytes);
            add(scratch, bytes);
            for (unsigned i = 0; i < sizeof(float); ++i) add(scratch, source.rows * source.columns);
            if (bytes > std::numeric_limits<std::size_t>::max())
                throw std::invalid_argument(name + ": payload exceeds host address space");
            if (name.empty() || std::ranges::any_of(name, [](unsigned char c) { return c <= 32U || c == 127U; }))
                throw std::invalid_argument("selected tensor name contains whitespace or control bytes");
            if (!names.insert(name).second) throw std::invalid_argument("duplicate packed tensor " + name);
            result.scratch = std::max(result.scratch, scratch);
            add(result.compact, bytes);
            add(result.roles[match->role].compact, bytes);
            result.jobs.push_back({std::move(name), match->role, std::move(source), match->format, bytes});
        };
        replace(tensor, *match);
        if (match->bank) {
            ms::ExpertSlabView bank;
            require(wei::bind_pack_bank(checkpoint, tensor, match->order, bank), tensor.name);
            replace(tensor, *match, "_scale");
            if (bank.format == ms::MatrixFormat::nvfp4) {
                replace(tensor, *match, "_amax");
                replace(tensor, *match, "_outer_scale_denominator");
            }
            if (bank.experts > 65536U) throw std::invalid_argument(tensor.name + ": too many expert slices");
            for (std::size_t e = 0; e < bank.experts; ++e) {
                wei::PackMatrixSource matrix;
                require(wei::pack_expert(bank, e, matrix), tensor.name);
                emit(wei::packed_expert_name(tensor.name, e), std::move(matrix));
            }
        } else {
            wei::PackMatrixSource matrix;
            require(wei::PackMatrixSource::bind(checkpoint, tensor, matrix), tensor.name);
            if (matrix.scales.source) replace(tensor, *match, "_scale_inv");
            emit(tensor.name, std::move(matrix));
        }
    }
    for (const auto& rule : rules)
        if (!rule.matched && !rule.format.name.empty())
            throw std::invalid_argument("rule for " + rule.role + " matched no checkpoint tensors: " + rule.prefix + "*" + rule.suffix);
    if (result.jobs.empty()) throw std::invalid_argument("description selects no tensors to pack");
    for (const auto& tensor : checkpoint.tensors()) {
        const auto it = keep.find(tensor.name);
        if (replaced.contains(tensor.name)) {
            if (it != keep.end()) throw std::invalid_argument("keep rule selects a required packed plane: " + tensor.name);
            continue;
        }
        const auto* rule = it == keep.end() ? nullptr : it->second;
        const std::string_view role = rule ? std::string_view(rule->role) : "unassigned";
        const std::string_view reason = rule && !rule->reason.empty() ? std::string_view(rule->reason) :
                                        rule ? "role is keep" : "no matching role rule";
        result.retained.emplace(tensor.name, std::pair{std::string(role), std::string(reason)});
        add(result.kept, tensor.payload.range.bytes);
        auto& total = result.roles[std::string(role)];
        add(total.source, tensor.payload.range.bytes);
        add(total.retained, tensor.payload.range.bytes);
    }
    return result;
}

void print_plan(const Conversion& plan, const wei::PackCheckpoint& checkpoint) {
    std::cout << "role\tsource_bytes\tpacked_bytes\tretained_bytes\n";
    for (const auto& [role, size] : plan.roles)
        std::cout << role << '\t' << size.source << '\t' << size.compact << '\t' << size.retained << '\n';
    auto resident = plan.compact;
    add(resident, plan.kept);
    std::cout << "planned_tensors=" << plan.jobs.size() << " packed_payload_bytes=" << plan.compact
              << " retained_source_bytes=" << plan.kept << " matrix_scratch_bound_bytes=" << plan.scratch << '\n'
              << "resident_weight_bytes_one_node=" << resident << " ideal_half_shard_bytes_per_node="
              << resident / 2U + resident % 2U << " (excludes KV, activations, cache and replication)\n";
    for (const auto& [tensor, why] : plan.retained)
        std::cout << "keep " << tensor << ' ' << why.first << ' ' << checkpoint.find(tensor)->payload.range.bytes
                  << " bytes: " << why.second << '\n';
    std::cout << std::flush;
}

struct ErrorSums {
    double error{}, signal{};
};

template <class S>
void encode(const Job& job, std::span<const float> weights, unsigned threads,
            std::vector<std::uint8_t>& bytes, sp::Header& header, ErrorSums* quality) {
    if (std::ranges::any_of(weights, [](float v) { return !std::isfinite(v); }))
        throw std::runtime_error(job.name + ": nonfinite source weight");
    const auto n = job.source.rows, k = job.source.columns, pitch = S::row_bytes(k);
    header = sp::make_header<S>(weights.data(), n, k, sp::default_table<S::bits>());
    constexpr std::size_t rows_per_task = 64;
    const auto tasks = (n - 1U) / rows_per_task + 1U;
    std::vector<ErrorSums> sums(quality ? tasks : 0U);
    wei::lowbit::gptq::detail::parallel_for(tasks, threads, [&](std::size_t task) {
        const wei::lowbit::NearestRounding nearest;
        const auto row = task * rows_per_task;
        const auto rows = std::min(n - row, rows_per_task);
        const auto* src = weights.data() + row * k;
        auto* dst = bytes.data() + row * pitch;
        sp::encode_rows<S>(src, rows, k, header, dst, nullptr, nullptr, sp::ScaleSelection::nearest_max);
        if (!quality) return;
        auto& total = sums[task];
        for (std::size_t b = 0; b < rows * (k / S::elements); ++b) {
            const auto* block = dst + b * S::bytes;
            for (unsigned g = 0; g < S::groups; ++g) {
                const auto scale = static_cast<double>(sp::scale_value(S::scale_byte(block, g), header.base));
                for (unsigned j = g * S::group; j < (g + 1U) * S::group; ++j) {
                    const auto code = S::code(block, j);
                    const auto m = static_cast<double>(header.table[code & (S::levels - 1U)]) * scale;
                    const auto decoded = code >> (S::bits - 1U) ? -m : m;
                    const auto value = static_cast<double>(src[b * S::elements + j]);
                    const auto diff = value - decoded;
                    total.signal += value * value;
                    total.error += diff * diff;
                }
            }
        }
    });
    if (quality)
        for (const auto& sum : sums) {
            quality->error += sum.error;
            quality->signal += sum.signal;
        }
    sp::mark_foldable<S>(header, bytes.data(), n, k);
}

using Options = std::map<std::string, std::string, std::less<>>;

std::string option(const Options& options, std::string_view key) {
    const auto it = options.find(key);
    return it == options.end() ? std::string{} : it->second;
}

std::string required(const Options& options, std::string_view key) {
    auto value = option(options, key);
    if (value.empty()) throw std::invalid_argument("need " + std::string(key));
    return value;
}

Options parse(int argc, char** argv, std::initializer_list<std::string_view> keys,
              std::initializer_list<std::string_view> flags = {}) {
    Options options;
    for (int i = 2; i < argc; ++i) {
        const std::string_view key = argv[i];
        const bool flag = std::ranges::find(flags, key) != flags.end();
        if (!flag && std::ranges::find(keys, key) == keys.end())
            throw std::invalid_argument("unknown option " + std::string(key));
        if (!flag && (i + 1 == argc || std::string_view(argv[i + 1]).starts_with("--")))
            throw std::invalid_argument("missing value for " + std::string(key));
        if (!options.emplace(key, flag ? "1" : argv[++i]).second)
            throw std::invalid_argument("duplicate option " + std::string(key));
    }
    return options;
}

std::uint64_t peak_rss() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS memory{};
    if (!K32GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory)))
        throw std::runtime_error("cannot query peak working set");
    return memory.PeakWorkingSetSize;
#else
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("cannot query peak RSS");
#ifdef __APPLE__
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024U;
#endif
#endif
}

std::uint64_t positive_integer(std::string_view text, std::string_view key) {
    std::uint64_t value = 0;
    const auto end = text.data() + text.size();
    const auto [position, error] = std::from_chars(text.data(), end, value);
    if (error != std::errc{} || position != end || value == 0)
        throw std::invalid_argument(std::string(key) + " requires a positive integer");
    return value;
}

void convert(const Options& options) {
    const auto model_path = required(options, "--model");
    const auto description_path = required(options, "--description");
    auto rules = description(description_path);
    const bool plan_only = options.contains("--plan-only");
    if (plan_only && options.contains("--out")) throw std::invalid_argument("--plan-only and --out are mutually exclusive");
    const auto output = plan_only ? std::string{} : required(options, "--out");
    const auto thread_option = option(options, "--threads");
    unsigned threads = std::clamp(std::thread::hardware_concurrency(), 1U, 8U);
    if (!thread_option.empty()) {
        const auto count = positive_integer(thread_option, "--threads");
        if (count > 64U)
            throw std::invalid_argument("--threads must be in [1, 64]");
        threads = static_cast<unsigned>(count);
    }
    const auto start = Clock::now();
    wei::PackCheckpoint checkpoint;
    require(wei::PackCheckpoint::open(model_path, checkpoint), "index checkpoint");
    const auto conversion = plan(checkpoint, std::move(rules));
    print_plan(conversion, checkpoint);
    const auto scratch_option = option(options, "--scratch-mib");
    constexpr std::uint64_t mib = 1U << 20U;
    const auto scratch_mib = scratch_option.empty() ? 8192U : positive_integer(scratch_option, "--scratch-mib");
    if (scratch_mib > UINT64_MAX / mib || conversion.scratch > scratch_mib * mib)
        throw std::invalid_argument("planned matrix exceeds --scratch-mib (requires " +
                                    std::to_string(conversion.scratch) + " bytes)");
    if (plan_only) return;
    if (std::filesystem::exists(std::filesystem::symlink_status(output)))
        throw std::invalid_argument("pack output already exists: " + output);
    wei::Digest256 checkpoint_digest{};
    require(checkpoint.digest(checkpoint_digest), "fingerprint checkpoint");
    std::vector<std::string_view> names;
    names.reserve(conversion.jobs.size());
    for (const auto& job : conversion.jobs) names.push_back(job.name);
    wei::LowbitPackWriter writer(checkpoint_digest, output, names, wei::PackVersion::checked);
    std::map<std::string, ErrorSums, std::less<>> qualities;
    const bool quality = options.contains("--quality");
    std::size_t completed = 0;
    for (const auto& job : conversion.jobs) {
        wei::Digest256 source{};
        std::vector<float> weights;
        require(job.source.decode(weights, &source), job.name);
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(job.bytes));
        sp::Header header;
        auto* sums = quality ? &qualities[job.role] : nullptr;
        if (job.format == wei::kSP4WeightStorage) encode<sp::SP4Storage>(job, weights, threads, bytes, header, sums);
        else if (job.format == wei::kSP3WeightStorage) encode<sp::SP3Storage>(job, weights, threads, bytes, header, sums);
        else encode<sp::SP2Storage>(job, weights, threads, bytes, header, sums);
        writer.add(job.name, job.format, job.source.rows, job.source.columns, source, std::move(bytes), header,
                   {job.role, "rtn-max-fixed@1", {}});
        if (++completed % 256U == 0)
            std::cerr << "encoded " << completed << '/' << conversion.jobs.size() << '\n';
    }
    require(writer.finish(), "publish pack");
    for (const auto& [role, sums] : qualities)
        std::cout << "quality " << role << " squared_error=" << sums.error << " signal=" << sums.signal
                  << " sqnr_db=" << 10.0 * std::log10(sums.signal / sums.error) << '\n';
    std::cout << "converted tensors=" << conversion.jobs.size() << " file_bytes=" << std::filesystem::file_size(output)
              << " seconds=" << std::chrono::duration<double>(Clock::now() - start).count()
              << " peak_rss_bytes=" << peak_rss() << '\n';
}

void verify(const Options& options) {
    const auto input = required(options, "--pack");
    const auto model = option(options, "--model");
    wei::LowbitPack pack;
    if (model.empty()) require(wei::LowbitPack::inspect(input, pack), "open pack");
    else {
        wei::PackCheckpoint checkpoint;
        wei::Digest256 digest{};
        require(wei::PackCheckpoint::open(model, checkpoint), "index checkpoint");
        require(checkpoint.digest(digest), "fingerprint checkpoint");
        require(wei::LowbitPack::open_host(input, digest, pack), "open pack");
    }
    require(pack.verify_identity(), "verify pack");
    require(pack.verify_payloads(), "verify tensors");
    // V2 has no digest stored on disk. An external recorded identity is required to attest its original content.
    constexpr char digits[] = "0123456789abcdef";
    std::string actual;
    for (const auto byte : pack.identity().bytes) {
        actual += digits[byte >> 4U];
        actual += digits[byte & 15U];
    }
    if (const auto expected = option(options, "--expect"); !expected.empty() && actual != expected)
        throw std::runtime_error("pack SHA-256 does not match --expect (got " + actual + ")");
    std::cout << (pack.version() == wei::PackVersion::checked ? "v3 whole-file digest verified" : "v2 framing and stable identity verified")
              << " tensors=" << pack.entries().size() << " sha256=" << actual << '\n';
}

void probe(const Options& options) {
    const auto name = required(options, "--tensor");
    const auto path = option(options, "--pack");
    const auto source = option(options, "--model");
    if (path.empty() == source.empty()) throw std::invalid_argument("probe needs either --pack or --model");
    if (!path.empty() && options.contains("--mmap")) throw std::invalid_argument("--mmap applies to a source checkpoint");
    const auto start = Clock::now();
    std::uint64_t size = 0;
    unsigned first = 0;
    if (!path.empty()) {
        wei::LowbitPack pack;
        require(wei::LowbitPack::inspect(path, pack), "open pack");
        wei::MappedBytes view;
        require(pack.map_tensor(name, view), name);
        size = view.bytes.size();
        first = std::to_integer<unsigned>(view.bytes.front());
    } else {
        wei::PackCheckpoint checkpoint;
        require(wei::PackCheckpoint::open(source, checkpoint), "index checkpoint");
        const auto* tensor = checkpoint.find(name);
        if (!tensor || !tensor->payload.range.bytes) throw std::invalid_argument("missing or empty source tensor " + name);
        size = tensor->payload.range.bytes;
        if (options.contains("--mmap")) {
            wei::MappedBytes view;
            require(tensor->payload.source->map(tensor->payload.range, view), name);
            first = std::to_integer<unsigned>(view.bytes.front());
        } else {
            if (size > std::numeric_limits<std::size_t>::max()) throw std::invalid_argument("probe tensor exceeds host address space");
            std::vector<wei::Byte> bytes(static_cast<std::size_t>(size));
            require(wei::read_slice(tensor->payload, bytes), name);
            first = std::to_integer<unsigned>(bytes.front());
        }
    }
    std::cout << std::setprecision(9) << "first_tensor_ms="
              << std::chrono::duration<double, std::milli>(Clock::now() - start).count()
              << " peak_rss_bytes=" << peak_rss() << " tensor_bytes=" << size << " first_byte=" << first << '\n';
}

void usage() {
    std::cout << "strix-pack convert --model CHECKPOINT --description JSON [--out NEW_PACK | --plan-only] [--threads N] [--scratch-mib N] [--quality]\n"
                 "strix-pack verify --pack PACK [--model CHECKPOINT] [--expect SHA256]\n"
                 "strix-pack probe (--pack PACK | --model CHECKPOINT [--mmap]) --tensor NAME\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string_view(argv[1]) == "--help") {
            usage();
            return argc < 2 ? 1 : 0;
        }
        const std::string_view command = argv[1];
        if (command == "convert") convert(parse(argc, argv, {"--model", "--description", "--out", "--threads", "--scratch-mib"}, {"--plan-only", "--quality"}));
        else if (command == "verify") verify(parse(argc, argv, {"--pack", "--model", "--expect"}));
        else if (command == "probe") probe(parse(argc, argv, {"--pack", "--model", "--tensor"}, {"--mmap"}));
        else throw std::invalid_argument("unknown subcommand " + std::string(command));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "strix-pack: " << e.what() << '\n';
        return 1;
    }
}
