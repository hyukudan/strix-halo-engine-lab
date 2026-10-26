#include "strix/ds41_dspark_driver.hpp"
#include "strix/ds41_dspark_vocab.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
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
    ++decisions_;
    const bool exploring = explore_every_ != 0U && decisions_ % explore_every_ == 0U;
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

BatchWindowController::BatchWindowController(std::uint32_t slots) : slot_seen_(slots), slot_hits_(slots) {
    if (slots == 0U) throw std::invalid_argument("BatchWindowController needs at least one slot");
}

void BatchWindowController::reset(std::uint32_t slot) {
    slot_seen_.at(slot).fill(0U);
    slot_hits_.at(slot).fill(0U);
}

double BatchWindowController::plain_cost(std::uint32_t n) const {
    if (n == 0U) return 0.0;
    std::array<double, max_requests + 1U> v{};
    for (std::uint32_t r = 0; r <= max_requests; ++r) v[r] = static_cast<double>(costs_.plain[r]);
    if (n <= max_requests && v[n] > 0.0) return v[n];
    std::uint32_t lo = 0, hi = 0;
    for (std::uint32_t r = 1; r < n && r <= max_requests; ++r)
        if (v[r] > 0.0) lo = r;
    for (std::uint32_t r = n + 1U; r <= max_requests; ++r)
        if (v[r] > 0.0) { hi = r; break; }
    const double x = n;
    if (lo != 0U && hi != 0U) return v[lo] + (v[hi] - v[lo]) * (x - lo) / (static_cast<double>(hi) - lo);
    // unmeasured: a request more costs about 1.5 window rows (its row plus its own segment's launches)
    const double row = costs_.row > 0.0F ? costs_.row : (lo != 0U ? 0.13 * v[lo] : hi != 0U ? 0.13 * v[hi] : 0.005);
    if (lo != 0U) return v[lo] + 1.5 * row * (x - lo);
    if (hi != 0U) return std::max(0.5 * v[hi], v[hi] - 1.5 * row * (static_cast<double>(hi) - x));
    return 0.075 + 1.5 * row * (x - 1.0);  // nothing measured yet
}

double BatchWindowController::step_cost(std::span<const std::uint32_t> rows, std::uint32_t drafting) const {
    const auto n = static_cast<std::uint32_t>(rows.size());
    std::uint32_t total = 0;
    for (const auto r : rows) total += r;
    const double plain = plain_cost(n);
    // defaults until measured: a window row ~13 % of a one-request step, a draft ~20 %, append + commit 1.5 ms
    const double row = costs_.row > 0.0F ? costs_.row : 0.13 * plain_cost(1U);
    if (drafting == 0U) return plain + row * (total - n);
    const double draft = costs_.draft > 0.0F ? costs_.draft : 0.2 * plain_cost(1U);
    const double extra = costs_.draft > 0.0F ? costs_.extra : 0.0015;  // measured drafts: `extra` as given (0 = inside the walls)
    const std::uint32_t drafters = std::max(costs_.drafters, 1U);
    return plain + row * (total - n) + draft * ((drafting + drafters - 1U) / drafters) + (drafters > 1U ? costs_.round : 0.0F) + extra * n;
}

double BatchWindowController::probability(std::uint32_t depth, float confidence) const {
    const auto b = SurvivalController::bin(confidence);
    std::uint64_t bs = 0, bh = 0;  // the bin over every depth
    for (std::uint32_t j = 0; j < block_size; ++j) bs += seen_[j][b], bh += hits_[j][b];
    const double q = (static_cast<double>(bh) + 1.0) / (static_cast<double>(bs) + 2.0);
    return (static_cast<double>(hits_[depth][b]) + 4.0 * q) / (static_cast<double>(seen_[depth][b]) + 4.0);
}

double BatchWindowController::slot_survival(std::uint32_t slot, std::uint32_t depth) const {
    std::uint64_t s = 0, h = 0;  // pooled over every request and confidence
    for (std::uint32_t b = 0; b < WindowController::bins; ++b) s += seen_[depth][b], h += hits_[depth][b];
    const double prior = (static_cast<double>(h) + 1.0) / (static_cast<double>(s) + 2.0);
    return (static_cast<double>(slot_hits_.at(slot)[depth]) + 8.0 * prior) / (static_cast<double>(slot_seen_.at(slot)[depth]) + 8.0);
}

std::vector<std::uint32_t> BatchWindowController::search(const std::vector<std::vector<double>>& expected, std::span<const std::uint32_t> limit,
                                                         std::uint32_t budget, int drafting) const {
    const auto n = static_cast<std::uint32_t>(expected.size());
    const auto total = [](const std::vector<std::uint32_t>& rows) {
        std::uint32_t t = 0;
        for (const auto r : rows) t += r;
        return t;
    };
    const auto rate = [&](const std::vector<std::uint32_t>& rows) {
        double e = 0.0;
        std::uint32_t d = 0;
        for (std::uint32_t i = 0; i < n; ++i) e += expected[i][rows[i] - 1U], d += rows[i] > 1U ? 1U : 0U;
        return e / step_cost(rows, drafting < 0 ? d : static_cast<std::uint32_t>(drafting));
    };
    std::vector<std::uint32_t> best(n, 1U);
    double best_rate = rate(best);
    if (n <= 4U) {  // exhaustive, in a fixed order (ties keep the earlier, fewer-row plan)
        std::vector<std::uint32_t> rows(n, 1U);
        for (;;) {
            std::uint32_t i = 0;
            while (i < n && rows[i] == limit[i]) rows[i++] = 1U;
            if (i == n) break;
            ++rows[i];
            if (total(rows) <= budget) {
                const double r = rate(rows);
                if (r > best_rate) best_rate = r, best = rows;
            }
        }
        return best;
    }
    for (std::uint32_t u = 1; u <= max_window; ++u) {  // local search from every uniform start
        std::vector<std::uint32_t> rows(n);
        for (std::uint32_t i = 0; i < n; ++i) rows[i] = std::min(u, limit[i]);
        while (total(rows) > budget) *std::max_element(rows.begin(), rows.end()) -= 1U;
        double cur = rate(rows);
        for (;;) {
            std::vector<std::uint32_t> cand;
            double cand_rate = cur;
            const auto consider = [&](const std::vector<std::uint32_t>& t) {
                if (total(t) > budget) return;
                const double r = rate(t);
                if (r > cand_rate) cand_rate = r, cand = t;
            };
            for (std::uint32_t i = 0; i < n; ++i) {
                auto t = rows;
                if (t[i] < limit[i]) ++t[i], consider(t);
                t = rows;
                if (t[i] > 1U) --t[i], consider(t);
                for (std::uint32_t j = 0; j < n; ++j) {
                    t = rows;
                    if (j != i && t[i] < limit[i] && t[j] > 1U) ++t[i], --t[j], consider(t);
                }
            }
            if (cand.empty()) break;
            rows = cand;
            cur = cand_rate;
        }
        if (cur > best_rate) best_rate = cur, best = rows;
    }
    return best;
}

std::vector<bool> BatchWindowController::plan_drafts(std::span<const std::uint32_t> slots, std::span<const std::uint32_t> limit,
                                                     std::uint32_t budget) {
    const auto n = static_cast<std::uint32_t>(slots.size());
    if (limit.size() != n || budget < n) throw std::invalid_argument("BatchWindowController: one row per request at least");
    std::vector<bool> out(n, false);
    if (n == 0U) return out;
    if (++plans_ % explore_every == 0U) {  // exploration: one request with room drafts (round-robin over the plans)
        const auto first = static_cast<std::uint32_t>((plans_ / explore_every) % n);
        for (std::uint32_t j = 0; j < n && budget > n; ++j)
            if (const auto i = (first + j) % n; limit[i] > 1U) {
                out[i] = true;
                break;
            }
        return out;
    }
    std::vector<std::vector<double>> expected(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        double survival = 1.0, e = 1.0;
        expected[i].push_back(1.0);
        for (std::uint32_t R = 2; R <= limit[i]; ++R) {
            survival *= slot_survival(slots[i], R - 2U);
            e += survival;
            expected[i].push_back(e);
        }
    }
    const auto rows = search(expected, limit, budget, -1);
    for (std::uint32_t i = 0; i < n; ++i) out[i] = rows[i] > 1U;
    return out;
}

std::vector<std::uint32_t> BatchWindowController::choose_rows(std::span<const std::uint32_t> slots, std::span<const std::uint32_t> limit,
                                                              std::uint32_t budget, const std::vector<bool>& drafted,
                                                              std::span<const Draft> drafts) const {
    const auto n = static_cast<std::uint32_t>(slots.size());
    if (limit.size() != n || drafted.size() != n || drafts.size() != n || budget < n)
        throw std::invalid_argument("BatchWindowController: plan geometry");
    std::uint32_t drafting = 0;
    std::vector<std::uint32_t> lim(limit.begin(), limit.end());
    std::vector<std::vector<double>> expected(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        expected[i].push_back(1.0);
        if (!drafted[i]) {
            lim[i] = 1U;
            continue;
        }
        ++drafting;
        double survival = 1.0, e = 1.0;
        for (std::uint32_t R = 2; R <= lim[i]; ++R) {
            survival *= probability(R - 2U, drafts[i].confidence[R - 2U]);
            e += survival;
            expected[i].push_back(e);
        }
    }
    return search(expected, lim, budget, static_cast<int>(drafting));
}

void BatchWindowController::observe(std::uint32_t slot, const Draft& d, std::uint32_t verified, std::uint32_t accepted) {
    for (std::uint32_t j = 0; j < verified && j <= accepted && j < block_size; ++j) {
        const auto b = SurvivalController::bin(d.confidence[j]);
        ++seen_[j][b];
        ++slot_seen_.at(slot)[j];
        if (j < accepted) ++hits_[j][b], ++slot_hits_.at(slot)[j];
    }
}

WindowController::Table WindowController::table() const {
    Table t;
    t.seen = seen_;
    t.hits = hits_;
    return t;
}

void WindowController::set_table(const Table& table) {
    for (std::uint32_t j = 0; j < block_size; ++j)
        for (std::uint32_t b = 0; b < bins; ++b)
            if (table.hits[j][b] > table.seen[j][b]) throw std::invalid_argument("survival table: hits above seen");
    seen_ = table.seen;
    hits_ = table.hits;
}

namespace {
WindowController::Table parse_survival_table(std::istream& in, const std::string& path, std::uint32_t cap);
// Node probe prior (6 lines), the adopted default warm start
constexpr const char* kBuiltinSurvival = "ds41-dspark-survival-v1 bins 8\n"
    "d1 22 343 1008 1069 760 840 930 893 0 80 411 647 624 815 925 892\n"
    "d2 41 431 921 819 557 577 600 431 8 101 393 513 479 558 599 431\n"
    "d3 45 348 637 562 390 404 438 248 6 95 291 381 332 390 437 248\n"
    "d4 25 270 392 362 296 325 366 136 4 72 169 245 248 315 365 136\n"
    "d5 21 161 238 269 237 298 261 62 4 53 114 184 197 289 260 62\n";
}  // namespace

WindowController::Table builtin_survival_table(std::uint32_t cap) {
    std::istringstream in(kBuiltinSurvival);
    return parse_survival_table(in, "<builtin node prior>", cap);
}

WindowController::Table read_survival_table(const std::string& path, std::uint32_t cap) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("survival table: cannot read " + path);
    return parse_survival_table(in, path, cap);
}

namespace {
WindowController::Table parse_survival_table(std::istream& in, const std::string& path, std::uint32_t cap) {
    std::string magic, word;
    std::uint32_t nbins = 0;
    if (!(in >> magic >> word >> nbins) || magic != "ds41-dspark-survival-v1" || word != "bins" || nbins != WindowController::bins)
        throw std::runtime_error("survival table: bad header in " + path);
    WindowController::Table t;
    for (std::uint32_t j = 0; j < block_size; ++j) {
        if (!(in >> word) || word != "d" + std::to_string(j + 1U)) throw std::runtime_error("survival table: bad depth line in " + path);
        for (auto& v : t.seen[j])
            if (!(in >> v)) throw std::runtime_error("survival table: short seen counts in " + path);
        for (auto& v : t.hits[j])
            if (!(in >> v)) throw std::runtime_error("survival table: short hit counts in " + path);
        std::uint64_t total = 0;
        for (std::uint32_t b = 0; b < WindowController::bins; ++b) {
            if (t.hits[j][b] > t.seen[j][b]) throw std::runtime_error("survival table: hits above seen in " + path);
            total += t.seen[j][b];
        }
        if (cap > 0U && total > cap) {
            const double f = static_cast<double>(cap) / static_cast<double>(total);
            for (std::uint32_t b = 0; b < WindowController::bins; ++b) {
                const auto s = static_cast<std::uint32_t>(std::lround(t.seen[j][b] * f));
                const auto h = static_cast<std::uint32_t>(std::lround(t.hits[j][b] * f));
                t.seen[j][b] = s;
                t.hits[j][b] = std::min(h, s);
            }
        }
    }
    if (in >> word) throw std::runtime_error("survival table: trailing data in " + path);
    return t;
}
}  // namespace

void write_survival_table(const std::string& path, const WindowController::Table& table) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("survival table: cannot write " + path);
    out << "ds41-dspark-survival-v1 bins " << WindowController::bins << '\n';
    for (std::uint32_t j = 0; j < block_size; ++j) {
        out << 'd' << j + 1U;
        for (const auto v : table.seen[j]) out << ' ' << v;
        for (const auto v : table.hits[j]) out << ' ' << v;
        out << '\n';
    }
    if (!out) throw std::runtime_error("survival table: write failed " + path);
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
