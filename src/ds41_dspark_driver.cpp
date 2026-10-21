#include "strix/ds41_dspark_driver.hpp"
#include "strix/ds41_dspark_vocab.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

namespace strix::ds41::dspark {
namespace {

using clock = std::chrono::steady_clock;
double since(clock::time_point t) { return std::chrono::duration<double>(clock::now() - t).count(); }

} // namespace

void HostDrafter::append(std::span<const float> main_hidden, std::uint32_t rows) {
    for (std::uint32_t r0 = 0; r0 < rows; r0 += 128U) {
        const std::uint32_t n = std::min(128U, rows - r0);
        const auto part = main_hidden.subspan(static_cast<std::size_t>(r0) * main_width, static_cast<std::size_t>(n) * main_width);
        dspark::append(weights_, state_, main_rows(weights_, part, n), n);
    }
}
Draft HostDrafter::draft(std::uint32_t token) { return dspark::draft(weights_, state_, token, false, vocabulary_); }
void HostDrafter::add_context(std::span<const std::uint32_t> ids) {
    if (vocabulary_.core < vocab) (void)dspark::add_context(vocabulary_, ids);
}

std::uint32_t SurvivalController::bin(float confidence) {
    static constexpr float edges[bins - 1U] = {-2.0F, -1.0F, 0.0F, 1.0F, 2.0F, 4.0F, 8.0F};
    std::uint32_t b = 0;
    while (b < bins - 1U && confidence >= edges[b]) ++b;
    return b;
}
float SurvivalController::probability(std::uint32_t depth, float confidence) const {
    const auto b = bin(confidence);
    return (static_cast<float>(hits_[depth][b]) + 1.0F) / (static_cast<float>(seen_[depth][b]) + 2.0F);
}
std::uint32_t SurvivalController::choose(const Draft& d) {
    if (windows_++ % 16U == 0U) return block_size;
    float survival = 1.0F, expected = 1.0F, best_value = 1.0F / cost_[0];
    std::uint32_t best = 0;
    for (std::uint32_t k = 1; k <= block_size; ++k) {
        survival = survival * probability(k - 1U, d.confidence[k - 1U]);
        expected = expected + survival;
        if (expected / cost_[k] > best_value) { best_value = expected / cost_[k]; best = k; }
    }
    return best;
}
void SurvivalController::observe(const Draft& d, std::uint32_t verified, std::uint32_t accepted) {
    for (std::uint32_t j = 0; j < verified && j <= accepted; ++j) {
        const auto b = bin(d.confidence[j]);
        ++seen_[j][b];
        if (j < accepted) ++hits_[j][b];
    }
}

double WindowController::verify_cost(std::uint32_t rows) const {
    std::array<double, max_rows + 1U> v{};
    for (std::uint32_t r = 0; r <= max_rows; ++r) v[r] = static_cast<double>(costs_.verify[r]);
    const auto x = static_cast<double>(rows);
    if (rows >= 1U && rows <= max_rows && v[rows] > 0.0) return v[rows];
    std::uint32_t lo = 0, hi = 0;
    for (std::uint32_t r = 1; r < rows && r <= max_rows; ++r)
        if (v[r] > 0.0) lo = r;
    for (std::uint32_t r = rows + 1U; r <= max_rows; ++r)
        if (v[r] > 0.0) { hi = r; break; }
    const auto xl = static_cast<double>(lo), xh = static_cast<double>(hi);
    if (lo == 0U && hi == 0U) return 0.075 * (1.0 + 0.15 * (x - 1.0));  // nothing measured yet
    if (lo != 0U && hi != 0U) return v[lo] + (v[hi] - v[lo]) * (x - xl) / (xh - xl);
    if (lo == 0U) return v[hi] * (1.0 + 0.15 * (x - 1.0)) / (1.0 + 0.15 * (xh - 1.0));
    std::uint32_t lo2 = 0;  // the next measured size below lo: the slope of the two largest
    for (std::uint32_t r = 1; r < lo; ++r)
        if (v[r] > 0.0) lo2 = r;
    const double slope = lo2 != 0U ? (v[lo] - v[lo2]) / (xl - static_cast<double>(lo2)) : 0.15 * v[lo] / (1.0 + 0.15 * (xl - 1.0));
    return v[lo] + slope * (x - xl);
}

double WindowController::probability(std::uint32_t depth, float confidence) const {
    const auto b = SurvivalController::bin(confidence);
    std::uint64_t bs = 0, bh = 0;  // the bin over every depth
    for (std::uint32_t j = 0; j < block_size; ++j) bs += seen_[j][b], bh += hits_[j][b];
    const double q = (static_cast<double>(bh) + 1.0) / (static_cast<double>(bs) + 2.0);
    return (static_cast<double>(hits_[depth][b]) + 4.0 * q) / (static_cast<double>(seen_[depth][b]) + 4.0);
}

double WindowController::pooled(std::uint32_t depth) const {
    std::uint64_t s = 0, h = 0;
    for (std::uint32_t b = 0; b < bins; ++b) s += seen_[depth][b], h += hits_[depth][b];
    return (static_cast<double>(h) + 1.0) / (static_cast<double>(s) + 2.0);
}

double WindowController::window_rate(std::uint32_t drafts, const std::array<double, block_size>& p) const {
    double survival = 1.0, expected = 1.0;
    for (std::uint32_t j = 0; j < drafts; ++j) survival *= p[j], expected += survival;
    return expected / (costs_.draft + verify_cost(drafts + 1U) + costs_.extra);
}

bool WindowController::draft_next() {
    explore_ = false;
    const bool exploring = ++decisions_ % explore_every == 0U;
    if (sweep_ < block_size) return true;  // the initial sweep drafts
    std::array<double, block_size> p{};
    for (std::uint32_t j = 0; j < block_size; ++j) p[j] = pooled(j);
    double best = 0.0;
    for (std::uint32_t k = 1; k <= block_size; ++k) best = std::max(best, window_rate(k, p));
    const bool draft = best > 1.0 / verify_cost(1U);
    if (exploring) {
        if (!draft) return true;  // single steps lead: try a drafted window
        explore_ = true;          // drafting leads: the next window size is the least measured
    }
    return draft;
}

std::uint32_t WindowController::choose(const Draft& d) {
    if (sweep_ < block_size) return block_size - sweep_++;
    if (explore_) {
        std::uint32_t k = 1;
        for (std::uint32_t c = 2; c <= block_size; ++c)
            if (costs_.samples[c + 1U] < costs_.samples[k + 1U]) k = c;
        return k;
    }
    std::array<double, block_size> p{};
    for (std::uint32_t j = 0; j < block_size; ++j) p[j] = probability(j, d.confidence[j]);
    std::uint32_t best = 0;
    double best_rate = window_rate(0U, p);
    for (std::uint32_t k = 1; k <= block_size; ++k)
        if (const double r = window_rate(k, p); r > best_rate) best_rate = r, best = k;
    return best;
}

void WindowController::observe(const Draft& d, std::uint32_t verified, std::uint32_t accepted) {
    for (std::uint32_t j = 0; j < verified && j <= accepted && j < block_size; ++j) {
        const auto b = SurvivalController::bin(d.confidence[j]);
        ++seen_[j][b];
        if (j < accepted) ++hits_[j][b];
    }
}

std::uint32_t WindowPolicy::choose(const Draft& d) const {
    if (controller) return controller->choose(d);
    if (survival) return survival->choose(d);
    if (!automatic) return drafts;
    return choose_drafts(d.confidence, cost, 0U);
}
void WindowPolicy::observe(const Draft& d, std::uint32_t verified, std::uint32_t accepted) const {
    if (controller) controller->observe(d, verified, accepted);
    if (survival) survival->observe(d, verified, accepted);
}

bool dspark_enabled_from_env() {
    const char* v = std::getenv("STRIX_DS41_DSPARK");
    return v && v[0] == '1';
}

WindowPolicy window_policy_from_env() {
    WindowPolicy p;
    const char* v = std::getenv("STRIX_DS41_DSPARK_WINDOW");
    if (!v || !*v) return p;
    const std::string s(v);
    if (s == "controller") {
        p.controller = std::make_shared<WindowController>();
    } else if (s.rfind("fixed:", 0) == 0) {
        p.drafts = static_cast<std::uint32_t>(std::stoul(s.substr(6)));
        if (p.drafts > block_size) throw std::invalid_argument("STRIX_DS41_DSPARK_WINDOW=fixed:N needs N <= 5");
    } else if (s.rfind("auto", 0) == 0 || s.rfind("survival", 0) == 0) {
        const bool survival = s.rfind("survival", 0) == 0;
        const std::size_t name = survival ? 8U : 4U;
        p.automatic = !survival;
        if (s.size() > name + 1U && s[name] == ':') {
            std::stringstream ss(s.substr(name + 1U));
            std::string item;
            for (std::uint32_t k = 0; k <= block_size; ++k) {
                if (!std::getline(ss, item, ',')) throw std::invalid_argument("STRIX_DS41_DSPARK_WINDOW=auto:c0,..,c5 needs six costs");
                p.cost[k] = std::stof(item);
            }
        }
        if (survival) p.survival = std::make_shared<SurvivalController>(p.cost);
    } else {
        throw std::invalid_argument("STRIX_DS41_DSPARK_WINDOW must be fixed:N, auto[:c0,..,c5], survival[:c0,..,c5] or controller");
    }
    return p;
}

std::vector<std::uint32_t> generate_greedy(SpeculativeTarget& target, std::span<const std::uint32_t> prompt, std::uint32_t count) {
    if (prompt.empty()) throw std::invalid_argument("generate_greedy needs a prompt");
    std::uint32_t next = 0;
    for (const auto t : prompt) next = target.step(t);
    std::vector<std::uint32_t> out;
    while (out.size() < count) {
        out.push_back(next);
        if (out.size() < count) next = target.step(next);
    }
    return out;
}

std::vector<std::uint32_t> generate_speculative(SpeculativeTarget& target, Drafter& drafter, std::span<const std::uint32_t> prompt,
                                                std::uint32_t count, const WindowPolicy& policy, SpeculativeStats* stats) {
    if (prompt.empty()) throw std::invalid_argument("generate_speculative needs a prompt");
    SpeculativeStats local;
    auto& st = stats ? *stats : local;
    std::vector<float> rows, row(main_width);
    // Prompt rows: every one is committed.
    std::uint32_t t = 0;
    auto t0 = clock::now();
    for (const auto token : prompt) {
        t = target.step(token);
        ++st.target_steps;
        target.main_hidden(row);
        rows.insert(rows.end(), row.begin(), row.end());
    }
    st.target_seconds += since(t0);
    t0 = clock::now();
    drafter.append(rows, static_cast<std::uint32_t>(prompt.size()));
    drafter.add_context(prompt);
    const std::uint32_t first[1] = {t};
    drafter.add_context(first);
    st.draft_seconds += since(t0);
    std::vector<std::uint32_t> out{t};
    while (out.size() < count) {
        t0 = clock::now();
        const auto d = drafter.draft(t);
        st.draft_seconds += since(t0);
        const std::uint32_t k = policy.choose(d);
        ++st.windows;
        st.verified += k;
        for (std::uint32_t j = 0; j < k; ++j) ++st.verified_at_depth[j];
        // Row 0 is t; row j feeds draft j only if draft j equalled the target's greedy token after row j - 1.
        t0 = clock::now();
        rows.clear();
        std::uint32_t g = target.step(t);
        ++st.target_steps;
        target.main_hidden(row);
        rows.insert(rows.end(), row.begin(), row.end());
        std::uint32_t accepted = 0;
        while (accepted < k && d.tokens[accepted + 1U] == g && out.size() < count) {
            out.push_back(g);
            ++st.accepted_at_depth[accepted];
            ++accepted;
            if (out.size() >= count) break;
            g = target.step(d.tokens[accepted]);
            ++st.target_steps;
            target.main_hidden(row);
            rows.insert(rows.end(), row.begin(), row.end());
        }
        st.target_seconds += since(t0);
        st.accepted += accepted;
        for (std::uint32_t j = 0; j < k && j <= accepted; ++j) ++st.reached_at_depth[j];
        policy.observe(d, k, accepted);
        if (out.size() >= count) break;
        out.push_back(g);
        t0 = clock::now();
        drafter.append(rows, static_cast<std::uint32_t>(rows.size() / main_width));
        drafter.add_context(std::span<const std::uint32_t>(out).last(accepted + 1U));
        st.draft_seconds += since(t0);
        t = g;
    }
    out.resize(count);
    return out;
}

} // namespace strix::ds41::dspark
