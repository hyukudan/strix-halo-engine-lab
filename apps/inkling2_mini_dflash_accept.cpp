// Offline drafter acceptance on dumped target taps (Inkling 2 Mini): the shared block-diffusion drafter (DFlash 2) and
// the bundled DSpark drafter read the SAME committed taps of a greedy target run and propose from the same anchors.
//
//   strix-inkling2-mini-dflash-accept --model MODEL.safetensors --dflash DIR [--dspark MTP.safetensors]
//       --case LABEL=TAPS.bin,GENERATED.ids [--case ...] [--k 7] [--threads N] [--path greedy|viterbi] [--all] [--max-anchors N]
//
// TAPS.bin: the CUDA CLI's STRIX_INKLING_DUMP_TAPS file (prompt ids + BF16 taps of every committed position).
// GENERATED.ids: the run's generated ids (the CLI's "generated" line). The full sequence S = prompt + generated is the
// greedy target trace, so a proposal is accepted while it equals S.
// loop: the speculative loop at fixed k (anchor P, accept a, next anchor P + a + 1): tokens per verify = a + 1
//   (the published "acceptance length"). --all: every anchor (teacher-forced), accepted-length histogram per drafter.
// No GPU, no target forward: the target's embedding, embedding norm and head are read from MODEL once.
#include "core/bf16_reference.hpp"
#include "inkling2/inkling2_mini_checkpoint.hpp"
#include "inkling2/inkling2_mini_config.hpp"
#include "inkling2/inkling2_mini_dspark_reference.hpp"
#include "strix/spec/dflash_drafter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace strix;
using namespace strix::inkling2;

void require(const weight::Status& status, const std::string& what) {
    if (!status) throw std::runtime_error(what + ": " + status.detail);
}

float bf16_to_float(std::uint16_t v) {
    const std::uint32_t bits = std::uint32_t{v} << 16U;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

// The target pieces both drafters read: embedding table + embedding norm gain, head [vocab_out][hidden] (FP32, once).
class InklingTarget final : public spec::dflash::Target {
public:
    InklingTarget(const std::string& model, unsigned threads) : threads_(threads) {
        require(weights_.open(model), "target checkpoint");
        embed_norm_.resize(kHiddenWidth);
        require(weights_.read(name("embed_norm.weight"), embed_norm_), "embed_norm");
        head_.resize(std::size_t{kOutputVocabulary} * kHiddenWidth);
        require(weights_.read(name("unembed.weight"), head_), "unembed");
    }
    static std::string name(std::string_view suffix) { return "model.llm." + std::string(suffix); }
    // DSpark's ReadTarget: the head from memory, everything else from the file.
    void read(std::string_view suffix, std::span<float> out, std::size_t first) const {
        if (suffix == "unembed.weight" && first + out.size() <= head_.size()) {
            std::copy_n(head_.begin() + static_cast<std::ptrdiff_t>(first), out.size(), out.begin());
            return;
        }
        require(weights_.read(name(suffix), out, first), std::string(suffix));
    }
    void embed(std::uint32_t token, std::span<float> row) override {
        // The target's own boundary: table row, then its embedding norm with BF16 rounding (as its token rows).
        require(weights_.read(name("embed.weight"), row, std::size_t{token} * kHiddenWidth), "embed");
        core::rms_norm_bf16_reference(row, embed_norm_, 1e-6F);
    }
    [[nodiscard]] std::uint32_t vocab_out() const override { return kOutputVocabulary; }
    void head(std::span<const float> rows, std::uint32_t count, std::span<float> logits) override {
        const unsigned workers = std::max(1U, threads_);
        std::vector<std::thread> pool;
        const std::uint32_t chunk = (kOutputVocabulary + workers - 1U) / workers;
        for (unsigned w = 0; w < workers; ++w)
            pool.emplace_back([&, w] {
                const std::uint32_t begin = std::min(kOutputVocabulary, w * chunk), end = std::min(kOutputVocabulary, begin + chunk);
                for (std::uint32_t v = begin; v < end; ++v) {
                    const float* wrow = head_.data() + std::size_t{v} * kHiddenWidth;
                    for (std::uint32_t r = 0; r < count; ++r) {
                        const float* x = rows.data() + std::size_t{r} * kHiddenWidth;
                        float s[8]{};
                        for (std::uint32_t i = 0; i < kHiddenWidth; i += 8U)
                            for (unsigned l = 0; l < 8U; ++l) s[l] += wrow[i + l] * x[i + l];
                        logits[std::size_t{r} * kOutputVocabulary + v] = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
                    }
                }
            });
        for (auto& t : pool) t.join();
    }

private:
    MiniTextWeights weights_;
    std::vector<float> embed_norm_, head_;
    unsigned threads_;
};

struct Case {
    std::string label;
    std::uint32_t prompt = 0, positions = 0;
    std::vector<std::uint32_t> sequence;  // prompt + generated
    std::vector<float> taps;              // [positions][width] FP32
};

Case load_case(const std::string& spec) {
    const auto eq = spec.find('='), comma = spec.find(',');
    if (eq == std::string::npos || comma == std::string::npos || comma < eq) throw std::invalid_argument("--case LABEL=TAPS,IDS");
    Case c;
    c.label = spec.substr(0, eq);
    std::ifstream in(spec.substr(eq + 1U, comma - eq - 1U), std::ios::binary);
    char magic[8]{};
    std::uint32_t head[4]{};
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(head), sizeof head);
    if (!in || std::memcmp(magic, "INKTAPS1", 8) != 0 || head[1] != MiniDSparkTapPlan::width)
        throw std::runtime_error("not an Inkling tap dump: " + spec);
    c.positions = head[0];
    c.prompt = head[2];
    c.sequence.resize(c.prompt);
    in.read(reinterpret_cast<char*>(c.sequence.data()), static_cast<std::streamsize>(c.prompt * sizeof(std::uint32_t)));
    std::vector<std::uint16_t> raw(std::size_t{c.positions} * head[1]);
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(std::uint16_t)));
    if (!in) throw std::runtime_error("short tap dump: " + spec);
    c.taps.resize(raw.size());
    std::transform(raw.begin(), raw.end(), c.taps.begin(), bf16_to_float);
    std::ifstream ids(spec.substr(comma + 1U));
    std::uint32_t id;
    while (ids >> id) c.sequence.push_back(id);
    if (c.positions < c.prompt || c.positions > c.sequence.size())
        throw std::runtime_error("taps / ids disagree (positions " + std::to_string(c.positions) + ", sequence " +
                                 std::to_string(c.sequence.size()) + ")");
    return c;
}

// One drafter seen through the same interface: context rows up to an anchor, proposals from it.
class Proposer {
public:
    virtual ~Proposer() = default;
    virtual void context(const Case& c, std::uint32_t upto) = 0;  // append taps rows [have, upto)
    virtual std::vector<std::uint32_t> propose(std::uint32_t anchor, std::uint32_t position, std::uint32_t k) = 0;
};

class DFlashProposer final : public Proposer {
public:
    DFlashProposer(spec::dflash::Drafter& d, spec::dflash::Path path) : d_(d), path_(path) {}
    void context(const Case& c, std::uint32_t upto) override {
        const std::uint32_t have = d_.positions();
        if (upto > have)
            d_.append_context(std::span<const float>(c.taps).subspan(std::size_t{have} * MiniDSparkTapPlan::width,
                                                                     std::size_t{upto - have} * MiniDSparkTapPlan::width),
                              upto - have);
    }
    std::vector<std::uint32_t> propose(std::uint32_t anchor, std::uint32_t, std::uint32_t k) override {
        spec::dflash::Draft out;
        d_.draft(anchor, k, out, path_);
        return out.tokens;
    }

private:
    spec::dflash::Drafter& d_;
    spec::dflash::Path path_;
};

class DSparkProposer final : public Proposer {
public:
    DSparkProposer(const std::string& file, const InklingTarget& target)
        : ref_(file, [&target](std::string_view name, std::span<float> out, std::size_t first) { target.read(name, out, first); }) {}
    void context(const Case& c, std::uint32_t upto) override {
        if (upto > have_)
            ref_.append_context(std::span<const float>(c.taps).subspan(std::size_t{have_} * MiniDSparkTapPlan::width,
                                                                       std::size_t{upto - have_} * MiniDSparkTapPlan::width));
        have_ = std::max(have_, upto);
    }
    std::vector<std::uint32_t> propose(std::uint32_t anchor, std::uint32_t position, std::uint32_t k) override {
        std::vector<std::uint32_t> ids(MiniDSparkReference::policy().block, 0U), proposals;
        ids.front() = anchor;
        std::vector<float> confidence;
        ref_.forward({.ids = ids, .anchor_pos = position, .prefix = 0, .k = k}, proposals, confidence);
        return proposals;
    }

private:
    MiniDSparkReference ref_;
    std::uint32_t have_ = 0;
};

std::uint32_t accepted(const Case& c, std::uint32_t position, const std::vector<std::uint32_t>& drafts) {
    std::uint32_t a = 0;
    while (a < drafts.size() && position + 1U + a < c.sequence.size() && drafts[a] == c.sequence[position + 1U + a]) ++a;
    return a;
}

struct Tally {
    std::vector<std::uint64_t> hist;  // accepted drafts 0..k
    std::uint64_t steps = 0, tokens = 0;
    double draft_ms = 0.0;
};

void report(const std::string& what, const std::string& drafter, const Tally& t, std::uint32_t k) {
    std::ostringstream h;
    for (std::uint32_t a = 0; a <= k; ++a) h << (a ? "," : "") << t.hist[a];
    std::ostringstream ek;  // E(K) = mean(min(a, K) + 1) for K = 1 .. k (prefix-consistent proposals)
    for (std::uint32_t kk = 1; kk <= k; ++kk) {
        double s = 0.0;
        for (std::uint32_t a = 0; a <= k; ++a) s += static_cast<double>(t.hist[a]) * (std::min(a, kk) + 1U);
        ek << (kk > 1 ? "," : "") << std::fixed;
        ek.precision(3);
        ek << (t.steps ? s / static_cast<double>(t.steps) : 0.0);
    }
    std::printf("ACCEPT %s %s steps %llu tokens %llu E %.3f hist[a=0..%u] %s E(K=1..%u) %s draft_ms %.1f\n", what.c_str(),
                drafter.c_str(), static_cast<unsigned long long>(t.steps), static_cast<unsigned long long>(t.tokens),
                t.steps ? static_cast<double>(t.tokens) / static_cast<double>(t.steps) : 0.0, k, h.str().c_str(), k,
                ek.str().c_str(), t.steps ? t.draft_ms / static_cast<double>(t.steps) : 0.0);
    std::fflush(stdout);
}

// loop: the speculative loop at fixed k; all: every anchor of the generated span.
Tally run(Proposer& p, const Case& c, std::uint32_t k, bool all, std::uint32_t max_anchors) {
    Tally t;
    t.hist.assign(k + 1U, 0U);
    std::uint32_t position = c.prompt;  // the first generated token is the first anchor
    std::uint32_t n = 0;
    while (position < c.positions && position + 1U < c.sequence.size() && n < max_anchors) {
        p.context(c, position);
        const auto start = std::chrono::steady_clock::now();
        const auto drafts = p.propose(c.sequence[position], position, k);
        t.draft_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const auto a = accepted(c, position, drafts);
        ++t.hist[a];
        ++t.steps;
        t.tokens += a + 1U;
        ++n;
        position += all ? 1U : a + 1U;
    }
    return t;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string model, dflash_dir, dspark;
        std::vector<std::string> cases;
        std::uint32_t k = 7U, max_anchors = ~0U;
        unsigned threads = std::max(1U, std::thread::hardware_concurrency() / 2U);
        bool all = false;
        auto path = spec::dflash::Path::greedy;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            const bool more = i + 1 < argc;
            if (a == "--model" && more) model = argv[++i];
            else if (a == "--dflash" && more) dflash_dir = argv[++i];
            else if (a == "--dspark" && more) dspark = argv[++i];
            else if (a == "--case" && more) cases.emplace_back(argv[++i]);
            else if (a == "--k" && more) k = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            else if (a == "--threads" && more) threads = static_cast<unsigned>(std::stoul(argv[++i]));
            else if (a == "--max-anchors" && more) max_anchors = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            else if (a == "--all") all = true;
            else if (a == "--path" && more) path = std::string(argv[++i]) == "viterbi" ? spec::dflash::Path::viterbi : spec::dflash::Path::greedy;
            else throw std::invalid_argument("unknown argument " + a);
        }
        if (model.empty() || (dflash_dir.empty() && dspark.empty()) || cases.empty() || k == 0U || k > 7U)
            throw std::invalid_argument("usage: --model MODEL --dflash DIR [--dspark FILE] --case LABEL=TAPS,IDS [--k 1..7] [--all]");
        InklingTarget target(model, threads);
        std::optional<spec::dflash::Config> config;
        if (!dflash_dir.empty()) {
            std::ifstream f(std::filesystem::path(dflash_dir) / "config.json");
            std::stringstream text;
            text << f.rdbuf();
            config = spec::dflash::Config::from_json_text(text.str());
            if (config->target_layers.size() != MiniDSparkTapPlan::count)
                throw std::invalid_argument("DFlash taps differ from the target's tap plan");
            for (std::uint32_t i = 0; i < MiniDSparkTapPlan::count; ++i)
                if (config->target_layers[i] != MiniDSparkTapPlan::visits[i]) throw std::invalid_argument("DFlash tap visits differ");
            std::printf("DFLASH config hidden %u layers %u block %u window %u taps %u conv %u/%u selector %u/%u softcap %.1f %s\n",
                        config->hidden, config->layers, config->block, config->window, config->tap_width(), config->conv_taps,
                        config->conv_group, config->selector_rank, config->selector_top_k, static_cast<double>(config->softcap),
                        config->softcap_after_correction ? "after" : "before");
        }
        std::unique_ptr<spec::dflash::Weights> loaded;
        if (config) loaded = spec::dflash::load_weights(*config, dflash_dir);
        // One weight set shared by the per-case drafters (the drafter owns a Weights; a view forwards to the loaded set).
        class View final : public spec::dflash::Weights {
        public:
            explicit View(const spec::dflash::Weights& w) : w_(w) {}
            [[nodiscard]] std::span<const float> tensor(const std::string& n) const override { return w_.tensor(n); }

        private:
            const spec::dflash::Weights& w_;
        };
        for (const auto& spec_text : cases) {
            const auto c = load_case(spec_text);
            std::printf("CASE %s prompt %u positions %u sequence %zu\n", c.label.c_str(), c.prompt, c.positions, c.sequence.size());
            const std::string what = std::string(all ? "all " : "loop ") + c.label;
            if (config) {
                spec::dflash::Drafter d(*config, std::make_unique<View>(*loaded), target, threads);
                DFlashProposer p(d, path);
                report(what, path == spec::dflash::Path::viterbi ? "dflash2-viterbi" : "dflash2", run(p, c, k, all, max_anchors), k);
            }
            if (!dspark.empty()) {
                DSparkProposer p(dspark, target);
                report(what, "dspark", run(p, c, k, all, max_anchors), k);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
