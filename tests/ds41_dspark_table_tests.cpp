// the window controller's survival table (export, import with a cap, the text format) and the token
// identity of speculative decoding whatever table the controller starts from (window sizes change, tokens never).
// CPU only (no weights, no device).
#include "strix/ds41_dspark_driver.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix::ds41::dspark;

namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

// A deterministic toy target: the next token is a hash of the last three tokens and the position.
std::uint32_t toy_next(const std::vector<std::uint32_t>& seq) {
    std::uint64_t h = 1469598103934665603ULL ^ seq.size();
    for (std::size_t i = seq.size() >= 3U ? seq.size() - 3U : 0U; i < seq.size(); ++i) h = (h ^ seq[i]) * 1099511628211ULL;
    return static_cast<std::uint32_t>(h % 97U);
}
struct ToyTarget final : SpeculativeTarget {
    std::vector<std::uint32_t> seq;
    std::uint32_t step(std::uint32_t token) override {
        seq.push_back(token);
        return toy_next(seq);
    }
    void main_hidden(std::span<float> out) const override {
        std::fill(out.begin(), out.end(), 0.0F);
        out[1] = static_cast<float>(seq.back());
    }
};
// Guesses the toy target with errors; confidence informative but noisy.
struct ToyDrafter final : Drafter {
    std::vector<std::uint32_t> history;
    std::mt19937 random{11};
    float p_ok;
    explicit ToyDrafter(float p) : p_ok(p) {}
    void append(std::span<const float> rows, std::uint32_t count) override {
        for (std::uint32_t r = 0; r < count; ++r) history.push_back(static_cast<std::uint32_t>(rows[static_cast<std::size_t>(r) * main_width + 1U]));
    }
    Draft draft(std::uint32_t token) override {
        Draft d;
        d.tokens[0] = token;
        std::vector<std::uint32_t> guess(history);
        guess.push_back(token);
        std::uniform_real_distribution<float> u(0.0F, 1.0F);
        for (std::uint32_t j = 0; j < block_size; ++j) {
            std::uint32_t g = toy_next(guess);
            const bool wrong = u(random) > p_ok;
            if (wrong) g = (g + 1U) % 97U;
            guess.push_back(g);
            d.tokens[j + 1U] = g;
            d.confidence[j] = wrong ? u(random) * 4.0F - 3.0F : u(random) * 4.0F;
        }
        return d;
    }
};

std::string temp_file(const char* name) { return (std::filesystem::temp_directory_path() / name).string(); }

void test_table_io() {
    WindowController c;
    Draft d{};
    for (std::uint32_t i = 0; i < 300; ++i) {
        for (std::uint32_t j = 0; j < block_size; ++j) d.confidence[j] = static_cast<float>(static_cast<int>((i * 7U + j * 3U) % 13U) - 4);
        c.observe(d, 1U + i % 5U, i % 4U);
    }
    const auto t = c.table();
    const auto path = temp_file("draftacc-table-test.txt");
    write_survival_table(path, t);
    const auto back = read_survival_table(path);
    expect(back.seen == t.seen && back.hits == t.hits, "table: write/read round trip");
    // A controller given the table decides like the one that learned it.
    WindowController c2;
    c2.set_table(back);
    bool same = true;
    for (std::uint32_t i = 0; i < 64; ++i) {
        for (std::uint32_t j = 0; j < block_size; ++j) d.confidence[j] = static_cast<float>(static_cast<int>(i % 11U) - 3);
        for (std::uint32_t j = 0; j < block_size; ++j) same = same && c.probability(j, d.confidence[j]) == c2.probability(j, d.confidence[j]);
    }
    expect(same, "table: imported table gives the same survival probabilities");
    // Cap: every depth's seen counts sum to at most the cap (rounding), hits <= seen.
    const auto capped = read_survival_table(path, 40);
    for (std::uint32_t j = 0; j < block_size; ++j) {
        std::uint32_t s = 0;
        for (std::uint32_t b = 0; b < WindowController::bins; ++b) {
            s += capped.seen[j][b];
            expect(capped.hits[j][b] <= capped.seen[j][b], "table: capped hits <= seen");
        }
        expect(s <= 40U + WindowController::bins / 2U, "table: capped depth total near the cap");
    }
    // Malformed files throw.
    const auto bad = temp_file("draftacc-table-bad.txt");
    for (const char* text : {"ds41-dspark-survival-v2 bins 8\n", "ds41-dspark-survival-v1 bins 7\n",
                             "ds41-dspark-survival-v1 bins 8\nd1 1 1 1 1 1 1 1 1 2 0 0 0 0 0 0 0\n",
                             "ds41-dspark-survival-v1 bins 8\nd1 1 1\n"}) {
        {
            std::ofstream o(bad, std::ios::binary);
            o << text;
        }
        bool threw = false;
        try {
            (void)read_survival_table(bad);
        } catch (const std::exception&) {
            threw = true;
        }
        expect(threw, std::string("table: malformed file throws: ") + text);
    }
    WindowController::Table wrong;
    wrong.hits[0][0] = 1;
    bool threw = false;
    try {
        c2.set_table(wrong);
    } catch (const std::exception&) {
        threw = true;
    }
    expect(threw, "table: set_table rejects hits above seen");
    std::filesystem::remove(path);
    std::filesystem::remove(bad);
    std::cout << "table: round trip, same decisions, cap, malformed input ok\n";
}

// Tokens are the target's greedy tokens whatever table the controller starts from; the window sizes differ.
void test_token_identity() {
    const std::vector<std::uint32_t> prompt{3, 1, 4, 1, 5, 9};
    for (const float p_ok : {0.3F, 0.7F, 0.95F}) {
        ToyTarget plain;
        const auto greedy = generate_greedy(plain, prompt, 400);
        std::vector<std::uint32_t> verified;
        for (int variant = 0; variant < 3; ++variant) {
            WindowPolicy policy;
            policy.controller = std::make_shared<WindowController>();
            WindowController::Costs costs;
            costs.verify = {0.0F, 0.050F, 0.056F, 0.062F, 0.068F, 0.074F, 0.080F};
            costs.draft = 0.004F;
            policy.controller->set_costs(costs);
            if (variant > 0) {  // 1: an optimistic prior (every draft right), 2: a pessimistic one (every draft wrong)
                WindowController::Table t;
                for (std::uint32_t j = 0; j < block_size; ++j)
                    for (std::uint32_t b = 0; b < WindowController::bins; ++b) {
                        t.seen[j][b] = 50;
                        t.hits[j][b] = variant == 1 ? 50 : 0;
                    }
                policy.controller->set_table(t);
            }
            ToyTarget target;
            ToyDrafter drafter(p_ok);
            SpeculativeStats stats;
            const auto tokens = generate_speculative(target, drafter, prompt, 400, policy, &stats);
            expect(tokens == greedy, "identity: p_ok " + std::to_string(p_ok) + " prior " + std::to_string(variant) + ": tokens == greedy");
            verified.push_back(stats.verified);
        }
        expect(verified[1] > verified[2], "identity: the prior changes the window sizes (optimistic verifies more)");
        std::cout << "identity p_ok " << p_ok << ": verified drafts none/optimistic/pessimistic prior " << verified[0] << "/" << verified[1]
                  << "/" << verified[2] << ", tokens == greedy\n";
    }
}

}  // namespace

// Seeded start: no sweep, no exploration; decisions come from the table and the given costs at once.
void test_seeded() {
    WindowController c;
    WindowController::Costs costs;
    costs.verify = {0.0F, 0.0171F, 0.0229F, 0.0327F, 0.0357F, 0.0409F, 0.0459F};
    costs.draft = 0.00173F;
    c.set_costs(costs);
    WindowController::Table t;
    for (std::uint32_t j = 0; j < block_size; ++j)
        for (std::uint32_t b = 0; b < WindowController::bins; ++b) t.seen[j][b] = 64, t.hits[j][b] = b >= 5U ? 63U : 8U;
    c.set_table(t);
    c.skip_sweep();
    c.set_explore_every(0);
    Draft d{};
    d.confidence.fill(5.0F);  // a high-confidence bin: long window at once (no sweep)
    expect(c.draft_next(), "seeded: drafts");
    expect(c.choose(d) == block_size, "seeded: high confidence -> five drafts on the first decision");
    d.confidence.fill(-3.0F);  // low-confidence bin
    expect(c.choose(d) == 0U, "seeded: low confidence -> no draft verified");
    std::uint32_t explored = 0;
    d.confidence.fill(5.0F);
    for (int i = 0; i < 200; ++i) {
        (void)c.draft_next();
        explored += c.choose(d) != block_size ? 1U : 0U;
    }
    expect(explored == 0U, "seeded: exploration off -> no other window size");
    // token identity with the seeded controller
    const std::vector<std::uint32_t> prompt{2, 7, 1, 8};
    ToyTarget plain;
    const auto greedy = generate_greedy(plain, prompt, 300);
    WindowPolicy policy;
    policy.controller = std::make_shared<WindowController>(c);
    ToyTarget target;
    ToyDrafter drafter(0.8F);
    expect(generate_speculative(target, drafter, prompt, 300, policy) == greedy, "seeded: tokens == greedy");
    std::cout << "seeded controller: no sweep, no exploration, tokens == greedy\n";
}

int main() {
    try {
        test_table_io();
        test_token_identity();
        test_seeded();
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 2;
    }
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ds41 dspark table tests: all passed\n";
    return 0;
}
