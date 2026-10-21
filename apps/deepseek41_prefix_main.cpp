#include "strix/deepseek41_attention_state.hpp"
#include "strix/deepseek41_batch_forward.hpp"
#include "strix/deepseek41_prefill.hpp"
#include "strix/deepseek41_prefix_cache.hpp"
#include "strix/dist/communicator.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

// deepseek41 prefix cache gate: a prompt prefilled after its beginning was cached (DevicePrefill resumed from the
// cached block states: import_state + extend) against the same prompt prefilled fresh, bitwise: the greedy token after
// the prompt, the FNV of the last row's logits and the digest of the full attention state (every ring's live window,
// compressed KV, index keys, carries, tokens, position, next token; the last row's top-k is left out, nothing reads it).
// With --decode G both states are imported into a BatchForward (two slots, the prefill reading its weights in place)
// and decode G greedy tokens side by side, which must agree.
//
//   strix-deepseek41-prefix MODEL.gguf --prompt FILE (comma-separated ids) [--rows N] [--layers L]
//       [--world 1 [--stream-experts] | --transport emulate | --rank R --transport usb4-auto]
//       [--block 256] [--max-rows 4096] [--cache-mb 0] [--warm W1,W2,...] [--decode G] [--repeat-hit K]
//
// Scenarios, each on an emptied cache: "miss" (the prompt through the cache: pieces at the last full block, then the
// tail), and for every W of --warm (W < 0: N + W) a first prompt of W tokens through the cache, then the full prompt
// (a hit at the last boundary <= W cached by the first prompt; multi-turn chat when W = N - suffix). --repeat-hit K
// repeats the hit K times (timing; each hit resumes from the same snapshot). Times: "ttft" = lookup + state assembly +
// import + every prefill piece + the final export, i.e. what a server waits before its first token (without the
// tokenizer and the slot import, printed apart).
namespace {

namespace ds = strix::deepseek41;
using Clock = std::chrono::steady_clock;

struct Args {
    std::string model, prompt_path, transport, warm = "-64,-256";
    std::uint32_t rank = 0, world = 1, layers = 40, rows = 0, block = 256, max_rows = 4096, decode = 0, repeat_hit = 1;
    std::uint64_t cache_mb = 0;
    unsigned connect_timeout = 60;
    bool stream_experts = false, shared = true;
};

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

std::uint64_t fnv(const std::vector<float>& v) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const float f : v) {
        std::uint32_t b;
        std::memcpy(&b, &f, 4);
        for (int k = 0; k < 4; ++k, b >>= 8U) h = (h ^ (b & 0xffU)) * 1099511628211ULL;
    }
    return h;
}

std::vector<std::uint32_t> read_prompt(const std::string& path, std::uint32_t rows) {
    std::ifstream in(path);
    if (!in) throw std::invalid_argument("cannot read " + path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<std::uint32_t> ids;
    std::stringstream ss(text);
    for (std::string t; std::getline(ss, t, ',');)
        if (!t.empty()) ids.push_back(static_cast<std::uint32_t>(std::stoul(t)));
    if (rows != 0U) {
        if (rows > ids.size()) {  // repeat the prompt up to `rows` tokens (long-context gates from a shorter file)
            const auto base = ids;
            while (ids.size() < rows) ids.insert(ids.end(), base.begin(), base.end());
        }
        ids.resize(rows);
    }
    return ids;
}

std::vector<long> parse_list(const std::string& s) {
    std::vector<long> out;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');)
        if (!t.empty()) out.push_back(std::stol(t));
    return out;
}

struct Outcome {
    std::uint32_t next_token = 0, resume = 0;
    std::uint64_t logits = 0, digest = 0;
    std::vector<std::pair<std::string, std::uint64_t>> hashes;
    ds::AttentionState state;
    double seconds = 0.0, lookup_seconds = 0.0, import_seconds = 0.0, export_seconds = 0.0;
    std::vector<std::uint32_t> pieces;
};

class Gate {
public:
    Gate(const Args& a, std::uint32_t rank, strix::dist::Communicator* comm, bool print, std::size_t prompt_rows) : a_(a), print_(print) {
        const auto positions = static_cast<std::uint32_t>(prompt_rows + a.decode + 16U);
        if (a.decode != 0U) {
            ds::BatchForwardOptions bo;
            bo.layers = a.layers;
            bo.max_positions = positions;
            bo.slots = 2;
            bo.rank = rank;
            bo.world = comm ? 2U : 1U;
            bo.communicator = comm;
            bo.tp_attention = comm != nullptr;
            bo.engram_direct_io = true;
            batch_ = std::make_unique<ds::BatchForward>(a.model, bo);
        }
        ds::PrefillOptions o;
        o.layers = a.layers;
        o.max_positions = positions;
        o.max_rows = std::min<std::uint32_t>(a.max_rows, positions);
        o.rank = rank;
        o.world = comm ? 2U : 1U;
        o.communicator = comm;
        o.stream_experts = a.stream_experts;
        min_rows_ = comm ? o.window_rows + 1U : 1U;
        if (batch_ && a.shared) o.shared_weights = batch_->file_form_weights();
        const auto t0 = Clock::now();
        prefill_ = std::make_unique<ds::DevicePrefill>(a.model, o);
        if (print_)
            std::printf("rank %u: prefill weights %.2f GiB loaded + %.2f GiB shared in %.1f s%s\n", rank, static_cast<double>(prefill_->weight_bytes()) / 1073741824.0,
                        static_cast<double>(prefill_->shared_weight_bytes()) / 1073741824.0, since(t0),
                        batch_ ? " (BatchForward holds the shared ones)" : "");
        ds::PrefixCacheOptions co;
        co.block = a.block;
        co.budget_bytes = a.cache_mb << 20U;
        cache_ = std::make_unique<ds::PrefixCache>(co);
    }

    // The prompt prefilled fresh: one run of max_rows pieces (extend), no cache.
    Outcome fresh(const std::vector<std::uint32_t>& prompt) {
        Outcome o;
        const auto t0 = Clock::now();
        ds::PrefillResult r;
        std::uint32_t at = 0;
        for (const auto end : ds::plan_prefill_pieces(0U, 0U, static_cast<std::uint32_t>(prompt.size()), a_.max_rows, a_.max_rows, min_rows_)) {
            const std::span<const std::uint32_t> piece(prompt.data() + at, end - at);
            r = at == 0U ? prefill_->run(piece) : prefill_->extend(piece);
            at = end;
            o.pieces.push_back(at);
        }
        o.next_token = r.next_token;
        o.logits = fnv(r.logits);
        const auto t1 = Clock::now();
        o.state = prefill_->export_state();
        o.state.tokens = prompt;
        o.state.next_token = r.next_token;
        o.export_seconds = since(t1);
        o.seconds = since(t0);
        finish(o);
        return o;
    }

    // The prompt through the cache: resume from the longest snapshot, run the planned pieces, insert every boundary.
    Outcome cached(const std::vector<std::uint32_t>& prompt) {
        Outcome o;
        const auto t0 = Clock::now();
        const auto N = static_cast<std::uint32_t>(prompt.size());
        const auto hit = cache_->lookup(prompt, min_rows_);
        o.resume = hit.resume;
        if (hit.resume == N) {  // the whole prompt is a cached boundary: no prefill
            o.state = cache_->state(prompt, hit.resume, hit.key);
            o.next_token = hit.next_token;
            o.logits = hit.logits_fnv;
            o.lookup_seconds = since(t0);
            o.seconds = since(t0);
            finish(o);
            return o;
        }
        if (hit.resume > 0U) {
            const auto s = cache_->state(prompt, hit.resume, hit.key);
            o.lookup_seconds = since(t0);
            const auto t1 = Clock::now();
            prefill_->import_state(s);
            o.import_seconds = since(t1);
        } else {
            o.lookup_seconds = since(t0);
        }
        o.pieces = ds::plan_prefill_pieces(hit.resume, hit.shared, N, a_.block, a_.max_rows, min_rows_);
        ds::PrefillResult r;
        std::uint32_t at = hit.resume;
        for (const auto end : o.pieces) {
            const std::span<const std::uint32_t> piece(prompt.data() + at, end - at);
            r = at == 0U ? prefill_->run(piece) : prefill_->extend(piece);
            at = end;
            if (end % a_.block == 0U) {
                const auto t1 = Clock::now();
                auto s = prefill_->export_state();
                s.tokens.assign(prompt.begin(), prompt.begin() + end);
                s.next_token = r.next_token;
                (void)cache_->insert(s, fnv(r.logits));
                o.export_seconds += since(t1);
                if (end == N) o.state = std::move(s);
            }
        }
        o.next_token = r.next_token;
        o.logits = fnv(r.logits);
        if (o.state.layers.empty()) {
            const auto t1 = Clock::now();
            o.state = prefill_->export_state();
            o.state.tokens = prompt;
            o.state.next_token = r.next_token;
            o.export_seconds += since(t1);
        }
        o.seconds = since(t0);
        finish(o);
        return o;
    }

    // Both states decoded side by side in a BatchForward (slots 0 and 1); returns the two token lists.
    std::pair<std::vector<std::uint32_t>, std::vector<std::uint32_t>> decode(const Outcome& x, const Outcome& y, double& import_seconds) {
        auto& f = *batch_;
        const auto t0 = Clock::now();
        for (std::uint32_t s = 0; s < 2U; ++s) {
            if (!f.reserve(s, f.capacity(s))) throw std::runtime_error("reserve failed");
            f.import_state(s, s == 0U ? x.state : y.state);
        }
        import_seconds = since(t0) / 2.0;
        std::vector<std::uint32_t> tx{x.next_token}, ty{y.next_token};
        for (std::uint32_t g = 1; g < a_.decode; ++g) {
            const std::vector<ds::BatchRow> rows{{0U, tx.back()}, {1U, ty.back()}};
            const auto& t = f.step(rows);
            tx.push_back(t[0]);
            ty.push_back(t[1]);
        }
        return {tx, ty};
    }

    ds::PrefixCache& cache() { return *cache_; }
    [[nodiscard]] bool has_batch() const { return batch_ != nullptr; }

private:
    void finish(Outcome& o) const {
        o.digest = o.state.digest(true);
        o.hashes = o.state.hashes();
    }
    const Args& a_;
    bool print_;
    std::uint32_t min_rows_ = 1;  // world 2 (row split): every run gives each rank a row
    std::unique_ptr<ds::BatchForward> batch_;
    std::unique_ptr<ds::DevicePrefill> prefill_;
    std::unique_ptr<ds::PrefixCache> cache_;
};

std::string list(const std::vector<std::uint32_t>& v) {
    std::string s;
    for (const auto x : v) s += (s.empty() ? "" : ",") + std::to_string(x);
    return s.empty() ? "-" : s;
}

// Compares an outcome with the fresh one; prints one GATE line; returns the number of differences.
int compare(std::uint32_t rank, const std::string& name, const Outcome& ref, const Outcome& o) {
    int bad = 0;
    if (o.next_token != ref.next_token) ++bad;
    if (o.logits != ref.logits) ++bad;
    if (o.digest != ref.digest) ++bad;
    std::size_t diff = 0;
    for (std::size_t i = 0; i < ref.hashes.size() && i < o.hashes.size(); ++i)
        if (ref.hashes[i] != o.hashes[i] && ref.hashes[i].first.rfind("topk", 0) != 0) {
            if (diff < 6) std::printf("  rank %u %s: state %s differs\n", rank, name.c_str(), ref.hashes[i].first.c_str());
            ++diff;
        }
    std::printf("GATE rank %u %-14s resume %6u pieces %-24s token %6u logits %016llx digest %016llx  %s  ttft %.3f s (lookup+state %.3f, import %.3f, "
                "export+insert %.3f)\n",
                rank, name.c_str(), o.resume, list(o.pieces).c_str(), o.next_token, static_cast<unsigned long long>(o.logits),
                static_cast<unsigned long long>(o.digest), bad == 0 ? "EQUAL" : "DIFFERENT", o.seconds, o.lookup_seconds, o.import_seconds,
                o.export_seconds);
    std::fflush(stdout);
    return bad;
}

int run(const Args& a, const std::vector<std::uint32_t>& prompt, std::uint32_t rank, strix::dist::Communicator* comm, bool print) {
    Gate g(a, rank, comm, print, prompt.size());
    const auto N = static_cast<std::uint32_t>(prompt.size());
    int bad = 0;
    const auto ref = g.fresh(prompt);
    std::printf("GATE rank %u %-14s resume %6u pieces %-24s token %6u logits %016llx digest %016llx  ref    time %.3f s (export %.3f)\n", rank, "fresh",
                0U, list(ref.pieces).c_str(), ref.next_token, static_cast<unsigned long long>(ref.logits), static_cast<unsigned long long>(ref.digest),
                ref.seconds, ref.export_seconds);
    // miss through the cache, then the same prompt again (a hit at its last full block, or a full hit)
    g.cache().clear();
    const auto miss = g.cached(prompt);
    bad += compare(rank, "miss", ref, miss);
    const auto again = g.cached(prompt);
    bad += compare(rank, "same-prompt", ref, again);
    Outcome last_hit = again;
    for (const long w : parse_list(a.warm)) {
        const long W = w < 0 ? static_cast<long>(N) + w : w;
        if (W <= 0 || W >= static_cast<long>(N)) continue;
        g.cache().clear();
        const std::vector<std::uint32_t> first(prompt.begin(), prompt.begin() + W);
        (void)g.cached(first);
        const auto st = g.cache().stats();
        for (std::uint32_t k = 0; k < a.repeat_hit; ++k) {
            const auto hit = g.cached(prompt);
            bad += compare(rank, "warm" + std::to_string(W) + (a.repeat_hit > 1U ? "#" + std::to_string(k) : ""), ref, hit);
            last_hit = hit;
            if (k + 1U < a.repeat_hit) {  // the next repetition resumes from the same snapshot: drop the full prompt's
                g.cache().clear();
                (void)g.cached(first);
            }
        }
        std::printf("  cache after the first prompt (%ld tokens): %llu B in %llu blocks + %llu snapshots = %.0f B per cached token\n", W,
                    static_cast<unsigned long long>(st.bytes), static_cast<unsigned long long>(st.blocks), static_cast<unsigned long long>(st.snapshots),
                    static_cast<double>(st.bytes) / static_cast<double>(W / a.block * a.block));
    }
    if (a.decode != 0U && g.has_batch()) {
        double import_seconds = 0.0;
        const auto [tx, ty] = g.decode(ref, last_hit, import_seconds);
        const bool equal = tx == ty;
        bad += equal ? 0 : 1;
        std::printf("GATE rank %u decode %u tokens from the fresh and the cached state: %s (slot import %.3f s each); tokens %s\n", rank, a.decode,
                    equal ? "EQUAL" : "DIFFERENT", import_seconds, list(tx).c_str());
        if (!equal) std::printf("  cached: %s\n", list(ty).c_str());
    }
    const auto st = g.cache().stats();
    std::printf("GATE rank %u %s (%d differences); cache %llu lookups %llu hits %llu inserts %llu refused\n", rank, bad == 0 ? "PASS" : "FAIL", bad,
                static_cast<unsigned long long>(st.lookups), static_cast<unsigned long long>(st.hits), static_cast<unsigned long long>(st.inserts),
                static_cast<unsigned long long>(st.refused));
    std::fflush(stdout);
    return bad;
}

std::unique_ptr<strix::dist::Transport> open_transport(const std::string& spec, std::uint32_t rank, unsigned timeout) {
    if (spec != "usb4-auto") throw std::invalid_argument("unknown transport " + spec);
    strix::dist::Usb4AutoOptions o;
    o.rank = rank;
    o.timeout = std::chrono::milliseconds(std::uint64_t{timeout} * 1000U);
    strix::dist::Usb4Topology topology;
    auto t = strix::dist::usb4_host_auto(o, &topology);
    std::printf("usb4 topology: %s\n", topology.receipt().c_str());
    return t;
}

}  // namespace

int main(int argc, char** argv) try {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
            return std::string(argv[++i]);
        };
        const auto u32 = [&] { return static_cast<std::uint32_t>(std::stoul(next())); };
        if (s == "--prompt") a.prompt_path = next();
        else if (s == "--rows") a.rows = u32();
        else if (s == "--layers") a.layers = u32();
        else if (s == "--world") a.world = u32();
        else if (s == "--rank") a.rank = u32(), a.world = 2;
        else if (s == "--transport") a.transport = next(), a.world = 2;
        else if (s == "--stream-experts") a.stream_experts = true;
        else if (s == "--block") a.block = u32();
        else if (s == "--max-rows") a.max_rows = u32();
        else if (s == "--cache-mb") a.cache_mb = std::stoull(next());
        else if (s == "--warm") a.warm = next();
        else if (s == "--decode") a.decode = u32();
        else if (s == "--repeat-hit") a.repeat_hit = std::max<std::uint32_t>(1U, u32());
        else if (s == "--no-shared-weights") a.shared = false;
        else if (s == "--connect-timeout") a.connect_timeout = static_cast<unsigned>(std::stoul(next()));
        else if (a.model.empty()) a.model = s;
        else throw std::invalid_argument("unknown argument " + s);
    }
    if (a.model.empty() || a.prompt_path.empty()) throw std::invalid_argument("usage: strix-deepseek41-prefix MODEL.gguf --prompt FILE ...");
    if (a.stream_experts && a.decode != 0U) throw std::invalid_argument("--decode needs the experts resident (not --stream-experts)");
    const auto prompt = read_prompt(a.prompt_path, a.rows);
    std::printf("prompt %zu tokens, block %u, max rows %u, layers %u\n", prompt.size(), a.block, a.max_rows, a.layers);
    if (a.world == 1U) return run(a, prompt, 0U, nullptr, true) == 0 ? 0 : 1;
    if (a.transport == "emulate") {
        auto [t0, t1] = strix::dist::memory_pair();
        std::exception_ptr failure;
        int bad1 = 0;
        std::thread peer([&, t = std::move(t1)]() mutable {
            try {
                strix::dist::Communicator comm(strix::dist::Group{1U, 2U, 1U, 0x70726678ULL}, std::move(t), std::size_t{22} << 20U);
                bad1 = run(a, prompt, 1U, &comm, false);
            } catch (...) {
                failure = std::current_exception();
            }
        });
        int bad0 = 0;
        try {
            strix::dist::Communicator comm(strix::dist::Group{0U, 2U, 1U, 0x70726678ULL}, std::move(t0), std::size_t{22} << 20U);
            bad0 = run(a, prompt, 0U, &comm, true);
        } catch (...) {
            peer.join();
            throw;
        }
        peer.join();
        if (failure) std::rethrow_exception(failure);
        return bad0 == 0 && bad1 == 0 ? 0 : 1;
    }
    ::alarm(a.connect_timeout + 30U);
    auto transport = open_transport(a.transport, a.rank, a.connect_timeout);
    ::alarm(0);
    strix::dist::Communicator comm(strix::dist::Group{a.rank, 2U, 1U, 0x70726678ULL}, std::move(transport), std::size_t{22} << 20U);
    return run(a, prompt, a.rank, &comm, true) == 0 ? 0 : 1;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
