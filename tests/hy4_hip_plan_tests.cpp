// Hy4-flash HIP binding plan (src/hy4/hip/hy4_hip_plan.hpp): every stage names a shared entry, the layer order follows
// the oracle's residual order (Engram, attention sublayer, FFN sublayer), and the cache plan uses the shared
// cache-row layouts the readers expect. Optional argument: the first GGUF part of the pack (real layer schedule).
#include "strix/hy4/presets.hpp"
#include "hy4/hip/hy4_hip_plan.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/pair/layer_pipeline.hpp"
#include "strix/profiles/hy4.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <random>
#include <filesystem>
#include <string>

namespace hb = strix::hy4::hipbind;
namespace cr = strix::attn::cache_rows;

namespace {

int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

bool has(const std::vector<hb::Step>& s, hb::Stage st) {
    for (const auto& x : s)
        if (x.stage == st) return true;
    return false;
}

void check_config(const strix::hy4::Config& c, const char* name) {
    unsigned indexed = 0;
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const auto s = hb::layer_steps(c, l);
        bool engram = false;
        for (const auto e : c.engram_layers) engram = engram || e == l;
        for (const auto& x : s) {
            CHECK(!x.entry.empty() && !x.what.empty());
        }
        // residual order: Engram first (when present), the attention HC mix before the FFN HC mix, expand last
        CHECK((s.front().stage == hb::Stage::engram_rows) == engram);
        CHECK(s.back().stage == hb::Stage::hc_expand);
        CHECK(has(s, hb::Stage::index_select) == (c.indexed[l] != 0U));
        CHECK(has(s, hb::Stage::route) == (c.dense[l] == 0U));
        indexed += c.indexed[l] != 0U ? 1U : 0U;
        // caches: decode (1 row) and a 64-row chunk
        for (const std::uint32_t rows : {1U, 64U}) {
            const auto k = hb::layer_caches(c, l, 8192U, rows);
            CHECK(k.latent.width == c.kv_lora + c.qk_rope);
            if (c.indexed[l] == 0U) {
                CHECK(cr::is_pow2(k.latent.ring) && k.latent.ring >= c.window - 1U + rows && k.latent_rows == k.latent.ring);
                CHECK(k.token_rows == 0U && k.group_rows == 0U);
            } else {
                CHECK(k.latent.ring == 0U && k.latent_rows == 8192U);
                CHECK(k.token_keys.width == c.index_dim && k.token_keys.ring == c.index_ratio - 1U + rows);
                CHECK(k.group_rows == 8192U / c.index_ratio && k.group_keys.ring == 0U);
            }
        }
    }
    const auto head = hb::edge_steps(true);
    CHECK(hb::edge_steps(false).front().stage == hb::Stage::embed && head.back().stage == hb::Stage::argmax);
    for (const auto& x : head) CHECK(!x.entry.empty() && !x.what.empty());
    const double gib = static_cast<double>(hb::cache_bytes(c, 131072U, 1U)) / 1073741824.0;
    std::printf("%s: %u layers, %u indexed; caches at 128K decode %.3f GiB\n", name, c.layers, indexed, gib);
}

// Indexer binding constants (hy4_hip_plan.hpp): the device scan's head-weight scale is the oracle's fp32, the flash list
// length the host passes equals the shared contract's expanded selection, and the token ring holds every slot a pool of
// a launch of `rows` positions reads.
// Batched prompt plan: windows start 8-aligned, are multiples of 8 and >= 16 (grouped expert prefill admission), cover
// every position exactly once with the lead / tail steps; the indexed dense / select split of a window.
void check_prompt_plan() {
    for (const std::uint32_t rmax : {16U, 24U, 512U, 2048U, 4096U}) {
        for (const std::uint32_t pos : {0U, 3U, 8U, 2045U}) {
            for (const std::uint32_t count : {1U, 7U, 15U, 16U, 17U, 31U, 2048U, 4096U, 4100U, 4111U}) {
                const auto p = hb::prompt_plan(pos, count, rmax);
                std::uint32_t at = pos + p.lead, covered = p.lead + p.tail;
                CHECK(p.lead < 8U || p.lead == count);
                for (const auto& w : p.windows) {
                    CHECK(w.p0 == at && w.p0 % 8U == 0U && hb::prompt_window_ok(w.rows) && w.rows <= rmax);
                    at += w.rows;
                    covered += w.rows;
                }
                CHECK(covered == count && p.tail < 16U);
            }
        }
    }
    const auto off = hb::prompt_plan(0U, 100U, 12U);  // inadmissible window: every position is a step
    CHECK(off.lead == 100U && off.windows.empty() && off.tail == 0U);
    CHECK(hb::index_dense_rows(0U, 2048U) == 2048U && hb::index_dense_rows(2040U, 16U) == 8U && hb::index_dense_rows(2048U, 16U) == 0U);
    for (std::uint32_t t = 2040U; t < 2056U; ++t)
        CHECK((t < 2040U + hb::index_dense_rows(2040U, 16U)) == strix::attn::group_indexer::dense(strix::hy4::preset::indexer, t));
    // Verify rows flash split: window layers batch every row; indexed layers batch the causal-prefix rows, which come first
    // (dense is a prefix property), and row r of the batch attends iota[p0 + 1 + r] = p0 + r + 1 keys = its own launch's
    // index_list_tokens(p0 + r); the selecting rows follow one launch each.
    for (const std::uint32_t p0 : {0U, 100U, 2040U, 2044U, 2047U, 2048U, 3000U})
        for (std::uint32_t rows = 1U; rows <= 8U; ++rows) {
            const auto w = hb::vrows_flash_split(false, p0, rows);
            CHECK(w.batched == rows && w.per_row == 0U);
            const auto d = hb::vrows_flash_split(true, p0, rows);
            CHECK(d.batched + d.per_row == rows);
            for (std::uint32_t r = 0; r < rows; ++r) {
                const bool dense = strix::attn::group_indexer::dense(strix::hy4::preset::indexer, p0 + r);
                CHECK(dense == (r < d.batched));
                if (dense) {
                    CHECK(p0 + 1U + r == hb::index_list_tokens(p0 + r));
                }
            }
        }
    CHECK(hb::vrows_flash_split(true, 2044U, 8U).batched == 4U && hb::vrows_flash_split(true, 2044U, 8U).per_row == 4U);
    static_assert(hb::kKbPadInput % 128U == 0U && hb::kKbPadInput >= 192U);
}

// The pair invariant: the shared layer pipeline's windows (the same on every stage), each run as stage_window_plan,
// concatenate to the one-node prompt_plan from position 0 (same rows windows, same single steps, in order), so a stage
// of the pair executes exactly the one-node prefill's work in the same order and its bits equal it.
void check_stage_windows() {
    for (const std::uint32_t rmax : {16U, 24U, 512U, 1024U, 2048U}) {
        for (const std::uint32_t count : {1U, 7U, 15U, 16U, 17U, 31U, 32U, 1000U, 2048U, 4096U, 4100U, 4111U, 4119U, 16384U, 32768U}) {
            const auto one = hb::prompt_plan(0U, count, rmax);
            std::vector<hb::PromptWindow> wins;
            std::uint32_t steps_mid = 0, tail = 0, at = 0;
            const auto ws = strix::pair::layers::windows(count, rmax);
            for (std::size_t k = 0; k < ws.size(); ++k) {
                CHECK(ws[k].start == at);
                const auto p = hb::stage_window_plan(ws[k].start, ws[k].rows, rmax);
                CHECK(p.lead == 0U && p.windows.size() <= 1U);
                for (const auto& w : p.windows) wins.push_back(w);
                (k + 1U == ws.size() ? tail : steps_mid) += p.tail;
                at += ws[k].rows;
            }
            CHECK(at == count && steps_mid == 0U && tail == one.tail && one.lead == 0U);
            CHECK(wins.size() == one.windows.size());
            for (std::size_t k = 0; k < std::min(wins.size(), one.windows.size()); ++k)
                CHECK(wins[k].p0 == one.windows[k].p0 && wins[k].rows == one.windows[k].rows);
        }
    }
    bool refused = false;
    try {
        (void)hb::stage_window_plan(8U, 16U, 16U);  // a window off the shared grid
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    CHECK(refused);
    CHECK(hb::pair_cut(42U) == 21U);
    // The flow-shop picker over a calibration (synthetic costs: the wide window 2.3 % cheaper per row).
    const hb::PairWindowCost costs[] = {{1024U, 1000.0}, {2048U, 1954.0}};
    CHECK(hb::pair_window_rows(16U, costs) == 1024U && hb::pair_window_rows(4096U, costs) == 1024U &&
          hb::pair_window_rows(32768U, costs) == 1024U);
    CHECK(hb::pair_window_rows(65536U, costs) == 2048U);
    CHECK(hb::pair_window_rows(4096U, {}) == 0U);  // no calibration: the caller keeps its width
    const hb::PairWindowCost unusable[] = {{0U, 5.0}, {1024U, 0.0}};
    CHECK(hb::pair_window_rows(4096U, unusable) == 0U);
    // The interleaved pair plan over the measured cost (shared planner): narrower windows on short prompts, more stages
    // per rank and wider windows on long ones; an explicit width keeps it; one window keeps the two-stage cut.
    // Picks of the receipts' fit: 2K rows 1024 v 2, 4K 1024 v 4, 8K 2048 v 4, 32K 4096 v 4.
    CHECK(hb::pair_stage_plan(2048U, 42U).choice.chunk == 1024U && hb::pair_stage_plan(2048U, 42U).map.virtual_stages() == 2U);
    CHECK(hb::pair_stage_plan(4096U, 42U).choice.chunk == 1024U && hb::pair_stage_plan(4096U, 42U).map.virtual_stages() == 4U);
    CHECK(hb::pair_stage_plan(8192U, 42U).choice.chunk == 2048U && hb::pair_stage_plan(8192U, 42U).map.virtual_stages() == 4U);
    CHECK(hb::pair_stage_plan(32768U, 42U).choice.chunk == 4096U && hb::pair_stage_plan(32768U, 42U).map.virtual_stages() == 4U);
    // 8K, 2048 rows: v = 5 sends two 172 ms frames per 311 ms stage over the one link (measured seal waits 0.3-0.6 s,
    // 3.7 % slower than v = 3): the model prices it above v = 3 and v = 4.
    {
        const auto c8 = hb::kPairPipeCost.at(8192U);
        const double v3 = strix::pair::interleaved_wall_ms(8192U, 2048U, 3U, c8);
        const double v4 = strix::pair::interleaved_wall_ms(8192U, 2048U, 4U, c8);
        const double v5 = strix::pair::interleaved_wall_ms(8192U, 2048U, 5U, c8);
        CHECK(v5 > v3 && v4 < v3);
    }
    CHECK(hb::pair_stage_plan(8192U, 42U, 1024U).choice.chunk == 1024U);
    CHECK(hb::pair_stage_plan(1000U, 42U).choice.chunk == 0U && hb::pair_stage_plan(1000U, 42U).map.text() == "0,21,42");
}

// The tier word through the shared resolver: default leaves the plain invocation (prompt position by position); the
// batched prompt failed its gate, so fast leaves it pending (runs only with --allow-pending, stamped unqualified); an
// explicit --prompt-rows wins and stamps the run unqualified; turbo is refused (no version).
void check_profile() {
    namespace sp = strix::profile;
    const auto& engine = strix::hy4::profile::engine();
    const auto rows = [&](std::vector<std::string> args, std::string* status) {
        sp::Inputs in;
        in.env = [](const char*) -> const char* { return nullptr; };
        sp::take_options(engine, args, in);
        const auto r = sp::resolve(engine, in);
        const auto* e = r.find("prompt_rows");
        if (e == nullptr) return std::string("missing");
        *status = std::string(sp::status_name(e->status)) + " / " + r.qualification();
        return e->value;
    };
    const auto is = [](sp::Status s, const char* q) { return std::string(sp::status_name(s)) + " / " + q; };
    std::string st;
    CHECK(rows({}, &st) == "0" && st == is(sp::Status::builtin, "qualified"));
    CHECK(rows({"--profile", "default"}, &st) == "0" && st == is(sp::Status::builtin, "qualified"));
    CHECK(rows({"--profile", "fast"}, &st) == "0" && st == is(sp::Status::pending, "qualified"));
    CHECK(rows({"--profile", "fast", "--allow-pending"}, &st) == "4096" &&
          st.rfind(is(sp::Status::unqualified, "unqualified: prompt_rows"), 0) == 0);
    CHECK(rows({"--profile", "fast", "--prompt-rows", "512"}, &st) == "512" &&
          st == is(sp::Status::explicit_cli, "unqualified: prompt_rows"));
    CHECK(rows({"--prompt-rows", "2048"}, &st) == "2048" && st == is(sp::Status::explicit_cli, "unqualified: prompt_rows"));
    CHECK(rows({"--prompt-rows", "0"}, &st) == "0" && st == is(sp::Status::off, "qualified"));
    bool refused = false;
    try {
        (void)rows({"--profile", "turbo"}, &st);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    CHECK(refused);
}

// Per regime (strix/profiles/hy4.hpp resolve): fast applies the windows only where a gate receipt covers the device,
// the topology and the prompt length; elsewhere the cell is pending with the reason printed; an explicit qualified value
// keeps the run qualified; an unknown device and the pair stay pending (the pair passes its rows explicitly).
void check_profile_regime() {
    namespace sp = strix::profile;
    namespace hp = strix::hy4::profile;
    std::uint64_t gated = 0;
    for (const auto& q : hp::qualified)
        if (q.lever == "prompt_rows" && q.device == hp::Device::gfx1151 && q.topology == hp::Topology::one_node)
            gated = std::max(gated, q.max_prompt_tokens);
    CHECK(gated >= 32568U);  // the long-prompt gate (hy4_fast2): the 2K..32K scoreboard prompts are in the qualified range
    const auto run = [](hp::Device d, hp::Topology t, std::uint64_t tokens, std::vector<std::string> args, std::string* status) {
        sp::Inputs in;
        in.env = [](const char*) -> const char* { return nullptr; };
        sp::take_options(hp::engine(), args, in);
        hp::Regime r;
        r.device = d;
        r.topology = t;
        r.prompt_tokens = tokens;
        const auto res = hp::resolve(r, in);
        const auto* e = res->find("prompt_rows");
        if (e == nullptr) return std::string("missing");
        *status = std::string(sp::status_name(e->status)) + " / " + res->resolution.qualification() + (e->reason.empty() ? "" : " +reason");
        return e->value;
    };
    const auto g = hp::Device::gfx1151;
    const auto one = hp::Topology::one_node;
    std::string st;
    CHECK(run(g, one, 17, {"--profile", "fast"}, &st) == "4096" && st == "profile / qualified");
    CHECK(run(g, one, gated, {"--profile", "fast"}, &st) == "4096" && st == "profile / qualified");
    CHECK(run(g, one, 16384, {"--profile", "fast"}, &st) == "4096" && st == "profile / qualified");
    CHECK(run(g, one, gated + 1U, {"--profile", "fast"}, &st) == "0" && st == "pending / qualified +reason");
    CHECK(run(hp::Device::unknown, one, 17, {"--profile", "fast"}, &st) == "0" && st == "pending / qualified +reason");
    // the pair inherits each width's one-node gate by identity (same bits at the same width), over the same prompt range
    const auto pair = hp::Topology::pair;
    CHECK(run(g, pair, 17, {"--profile", "fast"}, &st) == "4096" && st == "profile / qualified");
    CHECK(run(g, pair, gated, {"--profile", "fast"}, &st) == "4096" && st == "profile / qualified");
    CHECK(run(g, pair, gated + 1U, {"--profile", "fast"}, &st) == "0" && st == "pending / qualified +reason");
    CHECK(run(hp::Device::unknown, pair, 17, {"--profile", "fast"}, &st) == "0" && st == "pending / qualified +reason");
    for (const char* w : {"1024", "2048", "4096"}) {
        CHECK(run(g, pair, 8192, {"--prompt-rows", w}, &st) == w && st == "explicit-cli / qualified");
        CHECK(run(g, one, 8192, {"--prompt-rows", w}, &st) == w && st == "explicit-cli / qualified");
        CHECK(run(g, pair, gated + 1U, {"--prompt-rows", w}, &st) == w && st == "explicit-cli / unqualified: prompt_rows");
    }
    CHECK(run(g, pair, 8192, {"--prompt-rows", "512"}, &st) == "512" && st == "explicit-cli / unqualified: prompt_rows");
    CHECK(run(g, one, 17, {}, &st) == "0" && st == "builtin / qualified");
    CHECK(run(g, one, 17, {"--prompt-rows", "4096"}, &st) == "4096" && st == "explicit-cli / qualified");
    CHECK(run(g, one, 32568, {"--prompt-rows", "4096"}, &st) == "4096" && st == "explicit-cli / qualified");
    // 2048 is in the table value's same_bits class: qualified wherever 4096 is, unqualified past the gated range
    CHECK(run(g, one, 17, {"--prompt-rows", "2048"}, &st) == "2048" && st == "explicit-cli / qualified");
    CHECK(run(g, one, gated + 1U, {"--prompt-rows", "2048"}, &st) == "2048" && st == "explicit-cli / unqualified: prompt_rows");
    CHECK(run(g, one, 17, {"--prompt-rows", "512"}, &st) == "512" && st == "explicit-cli / unqualified: prompt_rows");
    CHECK(hp::device_of("gfx1151:sramecc+:xnack-") == g && hp::device_of("gfx1100") == hp::Device::unknown);
    // head_q8: no prompt bound, one node and pair; unknown device pending
    const auto head = [](hp::Device d, hp::Topology t, std::uint64_t tokens) {
        sp::Inputs in;
        in.env = [](const char*) -> const char* { return nullptr; };
        std::vector<std::string> args{"--profile", "fast"};
        sp::take_options(hp::engine(), args, in);
        hp::Regime r;
        r.device = d;
        r.topology = t;
        r.prompt_tokens = tokens;
        const auto res = hp::resolve(r, in);
        const auto* e = res->find("head_q8");
        return e == nullptr ? std::string("missing") : e->value + "/" + sp::status_name(e->status);
    };
    CHECK(head(g, one, 1U << 20) == "1/profile");
    CHECK(head(g, hp::Topology::pair, 32768) == "1/profile");
    CHECK(head(hp::Device::unknown, one, 17) == "0/pending");
}

// pf_mha (environment lever STRIX_HY4_PF_MHA): fast sets it, and strix::profile::apply exports it, exactly where a
// `qualified` row covers the regime; elsewhere it stays pending and nothing is exported; an explicit value wins.
void check_profile_pf_mha() {
    namespace sp = strix::profile;
    namespace hp = strix::hy4::profile;
    const auto run = [](hp::Device d, hp::Topology t, std::uint64_t tokens, const char* env, std::string* exported) {
        sp::Inputs in;
        in.env = [env](const char* k) -> const char* { return std::strcmp(k, "STRIX_HY4_PF_MHA") == 0 ? env : nullptr; };
        std::vector<std::string> args{"--profile", "fast"};
        sp::take_options(hp::engine(), args, in);
        hp::Regime r;
        r.device = d;
        r.topology = t;
        r.prompt_tokens = tokens;
        const auto res = hp::resolve(r, in);
        exported->clear();
        sp::apply(res->resolution, [exported](const char* k, const char* v) {
            if (std::strcmp(k, "STRIX_HY4_PF_MHA") == 0) *exported = v;
            return true;
        });
        const auto* e = res->find("pf_mha");
        return e == nullptr ? std::string("missing") : e->value + "/" + sp::status_name(e->status);
    };
    const auto g = hp::Device::gfx1151;
    const auto one = hp::Topology::one_node;
    const auto pair = hp::Topology::pair;
    for (const auto t : {one, pair})
        for (const std::uint64_t tokens : {std::uint64_t{17}, std::uint64_t{8192}, std::uint64_t{32568}, std::uint64_t{1} << 20}) {
            hp::Regime r;
            r.device = g;
            r.topology = t;
            r.prompt_tokens = tokens;
            const bool gated = hp::qualification("pf_mha", "1", r, hp::qualified) != nullptr;
            std::string ex;
            const auto got = run(g, t, tokens, nullptr, &ex);
            CHECK(gated ? (got == "1/profile" && ex == "1") : (got == "0/pending" && ex.empty()));
            CHECK(run(g, t, tokens, "0", &ex) == "0/off" && ex.empty());
            CHECK(run(g, t, tokens, "1", &ex) == "1/explicit-env" && ex.empty());
        }
    std::string ex;
    CHECK(run(hp::Device::unknown, one, 17, nullptr, &ex) == "0/pending" && ex.empty());
}

void check_indexer_binding() {
    namespace gi = strix::attn::group_indexer;
    float scale = 0.0F;
    const float oracle = (1.0F / strix::ds41::declared_sqrt(16.0F)) * (1.0F / strix::ds41::declared_sqrt(128.0F));
    std::memcpy(&scale, &hb::kIndexHeadScaleBits, sizeof scale);
    CHECK(scale == oracle);
    CHECK(scale == gi::head_weight_scale(strix::hy4::preset::indexer));
    static_assert(hb::kIndexListStride == 2048U);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0F, 4.0F);
    const std::uint32_t ts[] = {0U, 1U, 3U, 2043U, 2046U, 2047U, 2048U, 2049U, 2050U, 2051U, 2052U, 2055U, 3001U, 4095U, 4096U, 16383U, 32767U};
    for (const auto t : ts) {
        const unsigned P = gi::visible_groups(strix::hy4::preset::indexer, t);
        std::vector<float> scores(P);
        for (auto& x : scores) x = std::round(u(rng) * 8.0F) / 8.0F;  // coarse values: ties at the cut are exercised
        const auto list = gi::expand_tokens(strix::hy4::preset::indexer, gi::select_groups(strix::hy4::preset::indexer, scores.data(), P), t);
        CHECK(list.size() == hb::index_list_tokens(t));
        CHECK(list.size() <= hb::kIndexListStride);
        CHECK(!list.empty() && list.back() == t);
        if (gi::dense(strix::hy4::preset::indexer, t)) CHECK(list.size() == t + 1U);
    }
    // Token ring (layer_caches: ratio - 1 + rows slots): a launch of rows positions ending at e pools every group whose
    // 4th slot lies in the launch; the earliest such group starts at e - rows - 2, so ring >= rows + 3 keeps its keys.
    for (unsigned rows : {1U, 2U, 7U, 64U, 512U}) {
        const unsigned ring = gi::token_ring_slots(strix::hy4::preset::indexer, rows);
        for (std::uint32_t e = rows + 8U; e < rows + 24U; ++e) {
            const std::uint32_t lo = e + 1U - rows;
            for (std::uint32_t g = lo / 4U; g * 4U + 3U <= e; ++g) {
                if (g * 4U + 3U < lo) continue;
                CHECK(e - g * 4U < ring);  // slot 4g is not yet overwritten by slot e
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto tiny = strix::hy4::Config::tiny();
        check_config(tiny, "tiny");
        check_indexer_binding();
        check_prompt_plan();
        check_stage_windows();
        check_profile();
        check_profile_regime();
        check_profile_pf_mha();
        if (argc > 1 && std::filesystem::exists(argv[1])) {
            const auto real = strix::hy4::Config::from_gguf(argv[1]);
            check_config(real, "pack");
            unsigned indexed = 0;
            for (const auto x : real.indexed) indexed += x != 0U ? 1U : 0U;
            CHECK(real.layers == 42U && indexed == 12U);
            // 12 indexed layers x (1152 B latent + 64 B pooled key per token) + 30 sliding rings of 512 x 1152 B + 12
            // token rings of 4 x 256 B (decode)
            const std::uint64_t want = 12ULL * 131072ULL * 1152ULL + 12ULL * (131072ULL / 4ULL) * 256ULL + 30ULL * 512ULL * 1152ULL +
                                       12ULL * 4ULL * 256ULL;
            CHECK(hb::cache_bytes(real, 131072U, 1U) == want);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "hy4 hip plan: %d failures\n", failures);
        return 1;
    }
    std::printf("hy4 hip plan: PASS\n");
    return 0;
}
