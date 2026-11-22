#pragma once

#include "strix/ds41_rtx_drafter_select.hpp"
#include "strix/token_sampling.hpp"
// strix-cuda-ds41-replica --serve: the persistent worker's spool protocol (design notes) and the
// runtime / load-time option split. Host-only; unit-tested by tests/ds41_replica_serve_test.cpp.
//
//   DIR/queue/NAME.req   one argument per line (CR stripped, empty lines skipped); writers create NAME.tmp and rename it
//   DIR/run/NAME.req     the request being served (moved there atomically before it starts)
//   DIR/done/NAME.req    the request after it ran, next to DIR/done/NAME.json (rc, wall, error; written .tmp + rename)
//   DIR/ready.json       written once after the load; DIR/lock exists while a worker owns the spool
// Control requests: a single line "--quit" ends the worker, "--ping" only answers.
// Generation requests can set --temperature, --top-k, --top-p, --min-p and --seed. A positive temperature uses plain chat.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace strix::ds41::replica_serve {

namespace fs = std::filesystem;

// A request that cannot run as given (bad or load-time option, missing file, capacity): the worker answers rc 2 and
// continues; the process state was not touched.
struct RequestReject : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct SamplingReject : RequestReject {
    using RequestReject::RequestReject;
};

// Spool sampling options belong to the request alone and are removed before parsing the worker's runtime options.
inline std::optional<TokenSampler> take_sampling_args(std::vector<std::string>& args) {
    TokenSamplingPolicy policy;
    bool present = false;
    for (std::size_t i = 0; i < args.size();) {
        const std::string_view arg = args[i];
        const auto eq = arg.find('=');
        const auto field = arg.substr(0, eq);
        const auto read = [&](auto& value) {
            const std::string_view input = eq != std::string_view::npos ? arg.substr(eq + 1U) :
                i + 1U < args.size() ? std::string_view(args[i + 1U]) : std::string_view("");
            const auto* end = input.data() + input.size();
            const auto parsed = std::from_chars(input.data(), end, value);
            if (parsed.ec != std::errc{} || parsed.ptr != end)
                throw SamplingReject(std::string(field) + " has an invalid or missing value");
        };
        if (field == "--temperature") read(policy.temperature);
        else if (field == "--top-k" || field == "--top_k") read(policy.top_k);
        else if (field == "--top-p" || field == "--top_p") read(policy.top_p);
        else if (field == "--min-p" || field == "--min_p") read(policy.min_p);
        else if (field == "--seed") read(policy.seed);
        else { ++i; continue; }
        present = true;
        const auto first = args.begin() + static_cast<std::ptrdiff_t>(i);
        args.erase(first, first + (eq == std::string_view::npos ? 2 : 1));
    }
    if (!present) return std::nullopt;
    try {
        validate_token_sampling_policy(policy);
    } catch (const std::invalid_argument& ex) {
        throw SamplingReject(ex.what());
    }
    if (!token_sampling_enabled(policy)) return std::nullopt;
    // An explicit seed is replayable across spool requests and worker restarts.
    return std::optional<TokenSampler>(std::in_place, policy, TokenSamplerOptions{});
}

inline std::string sampled_chat_phase(const std::string& phase) {
    if (phase.starts_with("chats") || phase.starts_with("dchats")) return phase;
    // A speculative chat can run plain. Exact/spec qualification phases carry a greedy reference contract.
    if (phase.starts_with("schats")) return rtx::plain_phase(phase);
    throw SamplingReject("--phases: " + phase + " is unavailable with --temperature above zero");
}

inline std::uint32_t sample_final_row(TokenSampler& sampler, std::span<const float> logits,
                                      std::uint32_t rows, std::uint64_t absolute_position) {
    if (rows == 0 || logits.empty() || logits.size() % rows != 0)
        throw std::logic_error("sampling: missing final logits row");
    return sampler.decide(logits.last(logits.size() / rows), absolute_position, nullptr).token;
}

inline std::string sampling_receipt_fields(const TokenSamplingPolicy& policy) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<float>::max_digits10)
        << ",\"sampling\":\"sampling-v1\",\"sampling_policy\":{\"temperature\":" << policy.temperature
        << ",\"top_k\":" << policy.top_k << ",\"top_p\":" << policy.top_p << ",\"min_p\":" << policy.min_p
        << ",\"seed\":" << policy.seed << '}';
    return out.str();
}

// The options a request may carry (section 2): everything else needs a new worker. Values only change what the next
// phases do; the per-request reset restores every one of them to the launch value first.
inline const char* eviction_error(const std::string& policy, bool supported, bool auto_supported = false) {
    if (policy != "lru" && policy != "scored" && policy != "auto") return "--evict lru | scored | auto";
    if (policy == "auto" && (!supported || !auto_supported))
        return "--evict auto: this tierstream has no ts_dg_end_rows; rebuild against a supporting library";
    if (policy == "scored" && !supported)
        return "--evict scored: this tierstream has no ts_dg_set_policy; rebuild against a supporting library";
    return nullptr;
}

inline const std::set<std::string>& runtime_options() {
    static const std::set<std::string> s{
        "--phases", "--steps", "--ref", "--chats", "--chat-prompt", "--chat-steps", "--seqs-file", "--json", "--timeline",
        "--qual-dump", "--draft-trace", "--verify", "--warm", "--chat-logits", "--pressure-free", "--pressure-recent", "--prefetch-layer",
        "--prefetch-token", "--window", "--controller-scope", "--controller-prior", "--controller-save",
        "--controller-prior-alt", "--controller-costs", "--controller-explore", "--controller-learn", "--probe-phase",
        "--draft-vocab", "--winpf-l1-cap", "--winpf-pmin", "--winpf-prior-cap", "--winpf-prior-pmin", "--winpf-revert",
        "--wb-alt", "--import-state", "--reset-cache", "--env", "--qual",
        // runtime when the linked tierstream has ts_dg_set_sched (the worker refuses them otherwise)
        "--wb-defer", "--no-wb-defer", "--flush-ready", "--no-flush-ready", "--evict", "--census", "--no-census"};
    return s;
}
inline bool is_runtime_option(const std::string& a) { return runtime_options().count(a) != 0U; }

// Environment variables a request may set with --env NAME=VALUE (read per request by the replica).
inline bool is_request_env(const std::string& name) {
    return name == "STRIX_DS41_CUDA_ENGRAM_CHECK" || name == "STRIX_DS41_REPLICA_WINDOW_ENGRAM_PREFETCH";
}

// Throws RequestReject unless every option of `args` is a runtime option (values are skipped by the replica's parser;
// here only tokens starting with "--" that are not values of the previous option are checked).
inline void check_request_args(const std::vector<std::string>& args, bool sampled = false) {
    static const std::set<std::string> flags{"--chat-logits", "--controller-prior-alt", "--controller-learn", "--qual", "--wb-defer",
                                             "--no-wb-defer", "--flush-ready", "--no-flush-ready", "--census", "--no-census"};
    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto& a = args[i];
        if (a.rfind("--", 0) != 0) throw RequestReject("unexpected argument '" + a + "' (options only)");
        if (a == "--serve" || a.rfind("--serve-", 0) == 0) throw RequestReject("option " + a + " is a launch option");
        if (!is_runtime_option(a)) throw RequestReject("load-time option " + a + " needs a reload (a new worker)");
        if (sampled && (a == "--window" || a == "--probe-phase" || a == "--draft-vocab" ||
                        a.starts_with("--controller-") || a.starts_with("--winpf-")))
            throw SamplingReject(a + " is unavailable with --temperature above zero");
        if (flags.count(a)) continue;
        if (i + 1 >= args.size()) throw RequestReject("missing value of " + a);
        if (a == "--env") {
            const auto& v = args[i + 1];
            const auto eq = v.find('=');
            if (eq == std::string::npos || eq == 0 || !is_request_env(v.substr(0, eq))) throw RequestReject("--env NAME=VALUE with NAME one of the request variables, got '" + v + "'");
        }
        ++i;
    }
}

// One argument per line; CR stripped; empty lines and lines starting with '#' skipped.
inline std::vector<std::string> read_args_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw RequestReject("cannot read " + p.string());
    std::vector<std::string> out;
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        out.push_back(line);
    }
    return out;
}

// A JSON string literal (quotes, backslashes, control characters escaped).
inline std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (const char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        if (c == '"' || c == '\\') { o += '\\'; o += ch; }
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += ch;
    }
    return o + "\"";
}

// Writes `text` to `p` through `p`.tmp + rename (readers never see a partial file).
inline void write_atomic(const fs::path& p, const std::string& text) {
    const fs::path tmp = p.string() + ".tmp";
    {
        std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
        if (!o) throw std::runtime_error("cannot write " + tmp.string());
        o << text;
        if (!o.flush()) throw std::runtime_error("cannot write " + tmp.string());
    }
    std::error_code ec;
    for (int attempt = 0; attempt < 50; ++attempt) {  // a scanner may hold the target briefly (Windows)
        fs::rename(tmp, p, ec);
        if (!ec) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    throw std::runtime_error("cannot rename " + tmp.string() + ": " + ec.message());
}

// The spool of one worker. open() creates the layout and takes DIR/lock (a directory: creation is atomic; an existing
// lock = another worker or a stale one: the harness removes it, never the worker).
class Spool {
public:
    explicit Spool(fs::path dir) : dir_(std::move(dir)) {}
    ~Spool() { release(); }
    Spool(const Spool&) = delete;
    Spool& operator=(const Spool&) = delete;

    void open() {
        for (const char* d : {"", "queue", "run", "done"}) fs::create_directories(dir_ / d);
        std::error_code ec;
        if (!fs::create_directory(dir_ / "lock", ec) || ec) throw std::runtime_error("spool " + dir_.string() + " is locked (DIR/lock exists)");
        locked_ = true;
        // a request left in run/ by a crashed worker is not retried: it is answered as failed
        for (const auto& e : fs::directory_iterator(dir_ / "run")) {
            if (e.path().extension() != ".req") continue;
            const auto name = e.path().stem().string();
            finish(name, "{\"name\":" + json_str(name) + ",\"rc\":3,\"error\":\"left in run/ by an earlier worker\"}");
        }
    }
    void release() {
        if (!locked_) return;
        std::error_code ec;
        fs::remove(dir_ / "lock", ec);
        locked_ = false;
    }
    [[nodiscard]] const fs::path& dir() const noexcept { return dir_; }

    // The lexicographically first DIR/queue/*.req moved to DIR/run/ (nullopt: none). Files that vanish or cannot be
    // moved yet (a writer or a scanner holding them) are retried at the next poll.
    std::optional<std::string> take() {
        std::vector<std::string> names;
        std::error_code ec;
        for (fs::directory_iterator it(dir_ / "queue", ec), end; !ec && it != end; it.increment(ec))
            if (it->path().extension() == ".req") names.push_back(it->path().stem().string());
        std::sort(names.begin(), names.end());
        for (const auto& n : names) {
            std::error_code rc;
            fs::rename(dir_ / "queue" / (n + ".req"), dir_ / "run" / (n + ".req"), rc);
            if (!rc) return n;
        }
        return std::nullopt;
    }
    [[nodiscard]] fs::path running(const std::string& name) const { return dir_ / "run" / (name + ".req"); }
    // The answer (done/NAME.json, atomic) and the request moved to done/.
    void finish(const std::string& name, const std::string& json) {
        write_atomic(dir_ / "done" / (name + ".json"), json + "\n");
        std::error_code ec;
        fs::remove(dir_ / "done" / (name + ".req"), ec);
        fs::rename(running(name), dir_ / "done" / (name + ".req"), ec);
    }
    void ready(const std::string& json) { write_atomic(dir_ / "ready.json", json + "\n"); }

private:
    fs::path dir_;
    bool locked_ = false;
};

}  // namespace strix::ds41::replica_serve
