// Host invariants of the shared block-diffusion drafter (strix::spec::dflash).
#include "strix/spec/dflash_drafter.hpp"
#include "strix/token_sampling.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace strix::spec::dflash;

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}

// The published DFlash 2 drafter config of Inkling 2 Mini (trimmed to the read keys).
constexpr const char* kInklingConfig = R"({
  "architectures": ["DFlash2DraftModel"],
  "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "final_logit_softcapping": 30.0,
    "mask_token_id": 200056, "selector_rank": 256, "selector_softcap_mode": "after_correction", "selector_top_k": 16,
    "target_layer_ids": [5, 23, 41, 48]},
  "head_dim": 128, "hidden_act": "silu", "hidden_size": 2304, "intermediate_size": 6912, "is_causal": false,
  "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
  "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}, "sliding_window": 2048,
  "use_sliding_window": true, "vocab_size": 201024})";

std::string tiny_config(bool conv, bool selector) {
    std::string d = R"("block_size": 4, "mask_token_id": 63, "target_layer_ids": [1, 2])";
    if (conv) d += R"(, "conv_kernel_size": 2, "conv_group_size": 8)";
    if (selector) d += R"(, "selector_rank": 8, "selector_top_k": 4, "final_logit_softcapping": 30.0, "selector_softcap_mode": "after_correction")";
    return R"({"hidden_size": 32, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "intermediate_size": 48,
      "num_hidden_layers": 2, "vocab_size": 64, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "use_sliding_window": true,
      "sliding_window": 6, "dflash_config": {)" + d + "}}";
}

class TestWeights final : public Weights {
public:
    std::map<std::string, std::vector<float>> t;
    [[nodiscard]] std::span<const float> tensor(const std::string& name) const override { return t.at(name); }
};

std::unique_ptr<TestWeights> copy_weights(const Config& config, std::uint64_t seed) {
    auto source = synthetic_weights(config, seed);
    auto out = std::make_unique<TestWeights>();
    for (const auto& [name, extents] : inventory(config)) {
        const auto v = source->tensor(name);
        out->t[name].assign(v.begin(), v.end());
    }
    return out;
}

class TinyTarget final : public Target {
public:
    explicit TinyTarget(std::uint32_t hidden) : hidden_(hidden), table_(64U * hidden), head_(64U * hidden) {
        std::mt19937_64 rng(7);
        std::uniform_real_distribution<float> d(-1.0F, 1.0F);
        for (auto& x : table_) x = d(rng);
        for (auto& x : head_) x = d(rng);
    }
    void embed(std::uint32_t token, std::span<float> row) override {
        for (std::uint32_t i = 0; i < hidden_; ++i) row[i] = table_[std::size_t{token} * hidden_ + i];
    }
    [[nodiscard]] std::uint32_t vocab_out() const override { return 60U; }  // the padded tail is never proposed
    void head(std::span<const float> rows, std::uint32_t count, std::span<float> logits) override {
        for (std::uint32_t r = 0; r < count; ++r)
            for (std::uint32_t v = 0; v < 60U; ++v) {
                float s = 0.0F;
                for (std::uint32_t i = 0; i < hidden_; ++i) s += rows[std::size_t{r} * hidden_ + i] * head_[std::size_t{v} * hidden_ + i];
                logits[std::size_t{r} * 60U + v] = s;
            }
    }

private:
    std::uint32_t hidden_;
    std::vector<float> table_, head_;
};

std::vector<float> taps_for(const Config& c, std::uint32_t rows, std::uint64_t seed) {
    std::vector<float> taps(std::size_t{rows} * c.tap_width());
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> d(0.0F, 1.0F);
    for (auto& x : taps) x = d(rng);
    return taps;
}

Draft run(const Config& c, std::unique_ptr<Weights> w, Target& target, const std::vector<float>& taps, std::uint32_t rows,
          unsigned threads, Path path = Path::greedy, double temperature = 0.0, std::uint64_t seed = 0) {
    Drafter drafter(c, std::move(w), target, threads);
    drafter.append_context(taps, rows);
    Draft d;
    drafter.draft(5U, c.block - 1U, d, path, temperature, seed);
    return d;
}

// A drafter directory written from synthetic weights (BF16 safetensors + config.json): the device loader's host half.
std::filesystem::path write_drafter_dir(const Config& c, const std::string& config_text, std::uint64_t seed) {
    const auto dir = std::filesystem::temp_directory_path() / ("strix-dflash-test-" + std::to_string(seed));
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "config.json", std::ios::binary) << config_text;
    const auto weights = synthetic_weights(c, seed);
    std::string header = "{";
    std::vector<std::uint16_t> data;
    for (const auto& [name, extents] : inventory(c)) {
        const auto v = weights->tensor(name);
        const auto first = data.size() * 2U;
        for (const float x : v) {
            std::uint32_t u = 0;
            std::memcpy(&u, &x, sizeof(u));
            data.push_back(static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U));
        }
        std::string shape;
        for (const auto e : extents) shape += (shape.empty() ? "" : ",") + std::to_string(e);
        header += (header.size() > 1 ? "," : "") + std::string("\"") + name + "\":{\"dtype\":\"BF16\",\"shape\":[" + shape +
                  "],\"data_offsets\":[" + std::to_string(first) + "," + std::to_string(data.size() * 2U) + "]}";
    }
    header += "}";
    while (header.size() % 8U) header += ' ';
    std::ofstream file(dir / "model.safetensors", std::ios::binary);
    const std::uint64_t n = header.size();
    file.write(reinterpret_cast<const char*>(&n), sizeof(n));
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * 2U));
    return dir;
}

bool same(const Draft& a, const Draft& b) { return a.tokens == b.tokens && a.q == b.q && a.scores == b.scores; }

}  // namespace

int main() {
    {  // the published file's geometry and inventory
        const auto c = Config::from_json_text(kInklingConfig);
        check(c.hidden == 2304 && c.heads == 32 && c.kv_heads == 8 && c.head_dim == 128 && c.intermediate == 6912 && c.layers == 5,
              "inkling config geometry");
        check(c.block == 8 && c.window == 2048 && c.mask_token == 200056 && c.vocab == 201024, "inkling config block/window/mask");
        check(c.target_layers == std::vector<std::uint32_t>{5, 23, 41, 48} && c.tap_width() == 9216, "inkling config taps");
        check(c.conv_taps == 2 && c.conv_group == 16 && c.conv_groups() == 144, "inkling config conv");
        check(c.selector_rank == 256 && c.selector_top_k == 16 && c.softcap == 30.0F && c.softcap_after_correction, "inkling config selector");
        check(inventory(c).size() == 81U, "inkling inventory = 81 tensors");
        const auto p = Drafter::policy(c);
        check(p.block == 8 && p.fill_token == 200056 && p.ring == 2048 && p.target_layer_count == 4, "inkling policy");
        bool threw = false;
        try {
            std::string causal = kInklingConfig;
            causal.replace(causal.find("\"is_causal\": false"), 18, "\"is_causal\": true ");
            (void)Config::from_json_text(causal);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "causal block rows are refused");
    }
    const auto full = Config::from_json_text(tiny_config(true, true));
    TinyTarget target(full.hidden);
    const std::uint32_t rows = 9U;
    const auto taps = taps_for(full, rows, 11);
    const auto base = run(full, copy_weights(full, 3), target, taps, rows, 1U);
    check(base.tokens.size() == 3U && base.q.size() == 3U && base.candidates.size() == 12U, "proposal shape");
    for (std::size_t j = 0; j < base.tokens.size(); ++j) {
        bool in = false;
        for (std::uint32_t i = 0; i < 4U; ++i) in = in || base.candidates[j * 4U + i] == base.tokens[j];
        check(in && base.tokens[j] < 60U, "greedy path stays on the candidates, inside the valid vocabulary");
    }
    check(same(base, run(full, copy_weights(full, 3), target, taps, rows, 4U)), "threads do not change bits");
    {  // per-position context: truncate + re-append = fresh
        Drafter drafter(full, copy_weights(full, 3), target, 2U);
        drafter.append_context(taps, rows);
        drafter.truncate(5U);
        drafter.append_context(std::span<const float>(taps).subspan(std::size_t{5} * full.tap_width()), rows - 5U);
        Draft d;
        drafter.draft(5U, 3U, d);
        check(same(d, base), "truncate + re-append is exact");
    }
    {  // window 6 at P = 9: rows see j > 9 + r - 6 >= 3, so position 3 and below are never read; position 8 is
        auto changed = taps;
        for (std::size_t i = 0; i < std::size_t{4} * full.tap_width(); ++i) changed[i] += 3.0F;
        check(same(base, run(full, copy_weights(full, 3), target, changed, rows, 1U)), "out-of-window positions are not read");
        changed = taps;
        for (std::size_t i = std::size_t{8} * full.tap_width(); i < taps.size(); ++i) changed[i] += 3.0F;
        check(!same(base, run(full, copy_weights(full, 3), target, changed, rows, 1U)), "the last context position is read");
    }
    {  // identity convolution (base tap 0 = 1, others 0, no projection) = the drafter without the conv trait
        const auto plain_c = Config::from_json_text(tiny_config(false, true));
        auto with = copy_weights(full, 3);
        auto without = std::make_unique<TestWeights>();
        for (const auto& [name, extents] : inventory(plain_c)) without->t[name] = with->t.at(name);
        for (auto& [name, v] : with->t) {
            if (name.find("kernel_projection") != std::string::npos) std::fill(v.begin(), v.end(), 0.0F);
            if (name.find("base_kernel") != std::string::npos)
                for (std::size_t i = 0; i < v.size(); ++i) v[i] = (i / full.hidden) % full.conv_taps == 0 ? 1.0F : 0.0F;
        }
        check(same(run(full, std::move(with), target, taps, rows, 1U), run(plain_c, std::move(without), target, taps, rows, 1U)),
              "identity convolution = no convolution");
    }
    {  // zero codebooks: the selector reduces to the head's own argmax rows
        const auto noselect = Config::from_json_text(tiny_config(true, false));
        auto with = copy_weights(full, 3);
        auto without = std::make_unique<TestWeights>();
        for (const auto& [name, extents] : inventory(noselect)) without->t[name] = with->t.at(name);
        for (auto& [name, v] : with->t)
            if (name.find("codebook") != std::string::npos) std::fill(v.begin(), v.end(), 0.0F);
        const auto a = run(full, std::move(with), target, taps, rows, 1U);
        const auto b = run(noselect, std::move(without), target, taps, rows, 1U);
        check(a.tokens == b.tokens, "zero codebooks: selector path = argmax rows");
        const auto v = run(full, copy_weights(full, 3), target, taps, rows, 1U, Path::viterbi);
        check(v.tokens.size() == 3U, "viterbi path shape");
    }
    {  // sampled path: q is the selector softmax of the drafted token; first-step frequencies follow q
        std::map<std::uint32_t, int> count;
        std::map<std::uint32_t, double> q0;
        const int draws = 400;
        Drafter drafter(full, copy_weights(full, 3), target, 1U);
        drafter.append_context(taps, rows);
        for (int i = 0; i < draws; ++i) {
            Draft d;
            drafter.draft(5U, 3U, d, Path::greedy, 1.0, static_cast<std::uint64_t>(i) + 1U);
            ++count[d.tokens[0]];
            q0[d.tokens[0]] = d.q[0];
        }
        double chi = 0.0, total_q = 0.0;
        for (const auto& [token, n] : count) {
            const double e = q0[token] * draws;
            chi += (n - e) * (n - e) / e;
            total_q += q0[token];
        }
        check(total_q <= 1.0 + 1e-6 && chi < 30.0, "sampled first step follows q (chi-square)");
        Draft a, b;
        drafter.draft(5U, 3U, a, Path::greedy, 1.0, 99U);
        drafter.draft(5U, 3U, b, Path::greedy, 1.0, 99U);
        check(same(a, b), "a seed reproduces the sampled path");
        // The proposal q of every step: a distribution over the step's candidates whose drafted entry is Draft::q.
        for (std::uint32_t j = 0; j < a.tokens.size(); ++j) {
            const auto q = proposal(a, j, full.selector_top_k, 1.0);
            double sum = 0.0, drafted = -1.0;
            for (std::size_t i = 0; i < q.tokens.size(); ++i) {
                sum += q.probabilities[i];
                if (q.tokens[i] == a.tokens[j]) drafted = q.probabilities[i];
            }
            check(std::abs(sum - 1.0) < 1e-9 && std::abs(drafted - a.q[j]) < 1e-6, "proposal q matches the drafted q");
        }
    }
    {  // Lossless: drafts d ~ q (the selector's sparse support) decided by the shared rule give exactly p.
        Drafter drafter(full, copy_weights(full, 3), target, 1U);
        drafter.append_context(taps, rows);
        Draft d;
        drafter.draft(5U, 1U, d, Path::greedy, 0.8, 7U);
        const auto q = proposal(d, 0U, full.selector_top_k, 0.8);
        const auto qd = strix::make_token_distribution_from_probabilities(q.tokens, q.probabilities);
        // A target p over 24 ids that overlaps q's support partly (mass inside and outside it).
        std::vector<float> logits(24U);
        std::mt19937_64 rng(11U);
        std::uniform_real_distribution<float> u(-2.0F, 2.0F);
        for (auto& x : logits) x = u(rng);
        std::vector<std::uint32_t> ids(logits.size());
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = i < q.tokens.size() ? q.tokens[i] : 100000U + static_cast<std::uint32_t>(i);
        const strix::TokenSamplingPolicy policy{.temperature = 0.8F};
        const auto p = strix::make_token_distribution(logits, ids, policy);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        std::map<std::uint32_t, int> seen;
        const int draws = 200000;
        int accepted = 0;
        for (int n = 0; n < draws; ++n) {
            strix::TokenDraft draft;
            draft.token = qd.sample(uniform(rng));
            draft.distribution = qd;
            const auto decision = strix::decide_token(p, &draft, uniform(rng), uniform(rng), uniform(rng));
            accepted += decision.accepted ? 1 : 0;
            ++seen[decision.token];
        }
        double chi = 0.0;
        for (std::size_t i = 0; i < p.tokens().size(); ++i) {
            const double e = p.probabilities()[i] * draws;
            const double o = seen.count(p.tokens()[i]) ? seen[p.tokens()[i]] : 0.0;
            chi += (o - e) * (o - e) / e;
        }
        std::size_t outside = 0;
        for (const auto& [token, n] : seen)
            if (p.probability(token) == 0.0) outside += static_cast<std::size_t>(n);
        // 23 degrees of freedom: chi-square above 70 has probability < 1e-6 under p.
        check(chi < 70.0 && outside == 0U && accepted > 0 && accepted < draws, "speculative decision with q equals p");
    }
    {  // Device loader's host half: the resident plan built from the drafter directory binds every inventory tensor by
       // name (the names the plan reads live through make(); a dangling view crashed the first device ticket, df3).
        const auto text = tiny_config(true, true);
        const auto c = Config::from_json_text(text);
        const auto dir = write_drafter_dir(c, text, 42U);
        const auto loaded = load_config(dir);
        check(loaded.selector_top_k == c.selector_top_k && loaded.conv_taps == c.conv_taps, "load_config reads config.json");
        const auto lookup = open_lookup(c, dir);
        const auto plan = resident_plan(c, lookup, 256U);
        std::vector<std::string> clobber(4096, std::string(96, '#'));  // reuse freed heap before the names are read
        std::vector<std::uint8_t> host(plan.bytes() + 256U);
        const auto misalign = reinterpret_cast<std::uintptr_t>(host.data()) % 256U;
        const auto storage = std::span<const std::uint8_t>(host).subspan(misalign ? 256U - misalign : 0U, plan.bytes());
        const auto slices = strix::weight::ResidentSlices::bind(plan, storage);
        bool all = true;
        for (const auto& [name, extents] : inventory(c)) {
            std::uint64_t count = 1;
            for (const auto e : extents) count *= e;
            try { all = all && slices.bf16(name).size() == count; } catch (...) { all = false; }
        }
        check(all && !clobber.empty(), "resident plan binds every inventory tensor by name");
        const auto host_weights = load_weights(c, dir);
        check(host_weights->tensor("norm.weight").size() == c.hidden, "host loader reads the same directory");
        std::error_code ignored;
        std::filesystem::remove_all(dir, ignored);
    }
    if (failures == 0) std::puts("PASS strix-spec-dflash-tests");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
