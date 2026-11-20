// ONE resolver of the qwen35 HIP speculation (hip_spec_defaults.hpp + hip_dense_spec_costs.hpp): both entry points
// (strix-hip-qwen35-decode, strix-hip-qwen35-spec) parse the same flags with the same parser and end with the same plan,
// costs and block config; the work model's costs are measured on the device (calibrated) with the compiled table only as
// a reported fallback. Host only: a stand-in engine drives hip_dense_measure_costs.
#include "strix/qwen35/hip_spec_defaults.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace q = strix::qwen35;

#define RC_CHECK(c, ...)                                     \
    do {                                                     \
        if (!(c)) {                                          \
            std::fprintf(stderr, "FAIL line %d: ", __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);               \
            std::fprintf(stderr, "\n");                      \
            std::exit(1);                                    \
        }                                                    \
    } while (0)

namespace {

// The parse loop of an entry point: the shared flags through the shared parser, everything else app-specific.
q::HipDenseSpecArgs parse(const std::vector<std::string>& argv, std::vector<std::string>* rest = nullptr) {
    q::HipDenseSpecArgs a;
    for (std::size_t i = 0; i < argv.size(); ++i) {
        auto next = [&]() -> std::string {
            if (i + 1 >= argv.size()) throw std::invalid_argument("missing value for " + argv[i]);
            return argv[++i];
        };
        if (!q::parse_hip_dense_spec_arg(argv[i], next, a) && rest) rest->push_back(argv[i]);
    }
    return a;
}

bool same(const strix::spec::BlockConfig& x, const strix::spec::BlockConfig& y) {
    return x.max_block == y.max_block && x.chain == y.chain && x.hybrid == y.hybrid && x.copy == y.copy &&
           x.fixed_costs == y.fixed_costs && x.chain_max_k == y.chain_max_k && x.chain_seed_step_ms == y.chain_seed_step_ms &&
           x.chain_seed_flush_ms == y.chain_seed_flush_ms;
}

struct FakeEngine {  // the members hip_dense_measure_costs uses
    enum class LogitRows { none, last, all };
    static constexpr std::uint32_t max_rows() { return 16U; }
    std::vector<std::uint32_t> windows;
    unsigned commits0 = 0, syncs = 0, all_rows = 0;
    int forward_rows(const std::uint32_t*, std::uint32_t rows, LogitRows want) {
        windows.push_back(rows);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));  // a measurable wall
        all_rows += want == LogitRows::all ? 1U : 0U;
        return 0;
    }
    void commit(std::uint32_t k) { commits0 += k == 0U ? 1U : 0U; }
    void sync() { ++syncs; }
    std::uint64_t weight_bytes() const { return 14'000'000'000ULL; }
    std::uint64_t mtp_weight_bytes() const { return 450'000'000ULL; }
    std::uint64_t head_weight_bytes() const { return 700'000'000ULL; }
};

std::string read(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    RC_CHECK(f.good(), "cannot read %s", path.c_str());
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

}  // namespace

int main() {
    // 1. One parser: the same flags mean the same thing for every entry point; bad values are refused.
    const auto d = parse({});
    RC_CHECK(d.mtp == "auto" && d.draft_rows && d.draft_packs.empty() && d.costs == q::HipDenseCostMode::calibrated, "defaults");
    std::vector<std::string> rest;
    const auto a = parse({"--model", "m.gguf", "--mtp", "off", "--draft-rows", "off", "--draft-pack", "a.ids,b.ids", "--spec-costs",
                          "table", "--chain-k", "5"},
                         &rest);
    RC_CHECK(a.mtp == "off" && !a.draft_rows && a.draft_packs == "a.ids,b.ids" && a.costs == q::HipDenseCostMode::table,
             "shared flags parsed");
    RC_CHECK(rest == std::vector<std::string>({"--model", "m.gguf", "--chain-k", "5"}), "app flags left to the app");
    RC_CHECK(parse({"--spec-costs", "fixed"}).costs == q::HipDenseCostMode::table, "fixed = table (previous name)");
    RC_CHECK(parse({"--spec-costs", "learned"}).costs == q::HipDenseCostMode::learned, "learned");
    // --cost-kv: the shared context cost switch (spec::set_context_cost), default online.
    RC_CHECK(d.cost_kv == "auto" && parse({"--cost-kv", "online"}).cost_kv == "online" && parse({"--cost-kv", "off"}).cost_kv == "off" && parse({"--cost-kv", "0.3"}).cost_kv == "0.3",
             "--cost-kv parsed");
    RC_CHECK(parse({"--spec-costs", "calibrated"}).costs_explicit && !parse({"--mtp", "off"}).costs_explicit,
             "an explicit --spec-costs is recorded (exact TP decode refuses an explicit non-table choice)");
    for (const auto& bad : std::vector<std::vector<std::string>>{{"--draft-rows", "yes"}, {"--spec-costs", "auto"}, {"--mtp"},
                                                                 {"--cost-kv", "maybe"}, {"--cost-kv", "-1"}}) {
        bool threw = false;
        try { (void)parse(bad); } catch (const std::invalid_argument&) { threw = true; }
        RC_CHECK(threw, "%s %s must be refused", bad[0].c_str(), bad.size() > 1 ? bad[1].c_str() : "");
    }

    // 2. One resolver: plan from (model, args, plain) only.
    const auto has = [](const std::string& m) { return m == "with-mtp.gguf"; };
    const auto pm = q::resolve_hip_dense_spec("with-mtp.gguf", d, false, has);
    RC_CHECK(pm.mtp_source == "with-mtp.gguf" && pm.mtp_reason == "model" && pm.draft_rows, "auto + layer = model");
    RC_CHECK(q::resolve_hip_dense_spec("plain.gguf", d, false, has).mtp_reason == "absent", "auto without layer = absent");
    const auto plain = q::resolve_hip_dense_spec("with-mtp.gguf", d, true, has);
    RC_CHECK(plain.mtp_source.empty() && plain.mtp_reason == "off" && !plain.draft_rows, "every-logit runs are plain");
    auto fa = d;
    fa.mtp = "side.gguf";
    fa.draft_packs = "x.ids,,y.ids";
    const auto pf = q::resolve_hip_dense_spec("plain.gguf", fa, false, has);
    RC_CHECK(pf.mtp_reason == "file" && pf.draft_packs == std::vector<std::string>({"x.ids", "y.ids"}), "file + packs");
    {  // the plan carries --cost-kv into the block config of both apps (fixed / calibrated model; learned walls ignore it)
        const auto costs = q::hip_dense_table_costs(true);
        const auto bd = q::hip_dense_block_config(pm, 1.0, costs);
        const auto bo = q::hip_dense_block_config(q::resolve_hip_dense_spec("with-mtp.gguf", parse({"--cost-kv", "off"}), false, has), 1.0, costs);
        const auto bk = q::hip_dense_block_config(q::resolve_hip_dense_spec("with-mtp.gguf", parse({"--cost-kv", "0.3"}), false, has), 1.0, costs);
        RC_CHECK(pm.cost_kv == "auto" && bd.fixed_costs && bd.arbiter.cost_fit && bd.arbiter.row_kv_ms_per_kpos == 0.0,
                 "default auto: table fallback -> online");
        const double cw[5] = {77.0, 80.0, 83.0, 95.0, 141.0};
        const auto cal = q::hip_dense_calibrated_costs(cw, 14'000'000'000ULL, 450'000'000ULL, 700'000'000ULL, true);
        RC_CHECK(cal.source == q::HipDenseCostSource::calibrated, "calibrated walls");
        const auto bc_auto = q::hip_dense_block_config(pm, 1.0, cal);
        const auto bc_on = q::hip_dense_block_config(q::resolve_hip_dense_spec("with-mtp.gguf", parse({"--cost-kv", "online"}), false, has), 1.0, cal);
        RC_CHECK(bc_auto.fixed_costs && !bc_auto.arbiter.cost_fit && bc_auto.arbiter.row_kv_ms_per_kpos == 0.0,
                 "default auto: calibrated seeds -> off");
        RC_CHECK(bc_on.arbiter.cost_fit, "explicit online kept with calibrated seeds");
        RC_CHECK(!bo.arbiter.cost_fit && bo.arbiter.row_kv_ms_per_kpos == 0.0, "off");
        RC_CHECK(!bk.arbiter.cost_fit && bk.arbiter.row_kv_ms_per_kpos == 0.3, "fixed");
    }

    // 3. Costs: calibrated from walls, table only as the fallback.
    const auto table = q::hip_dense_table_costs(true);
    RC_CHECK(table.source == q::HipDenseCostSource::table && table.bucket_ms[4] == q::kHipDenseBucketMsScalar[4] &&
                 table.draft_a == q::kHipDenseDraftA && table.catch_up == q::kHipDenseDraftCatchUp,
             "table = compiled receipt");
    const double walls[5] = {70.0, 72.0, 71.0, 90.0, 130.0};  // r4 measured below r2: noise
    const auto c = q::hip_dense_calibrated_costs(walls, 14'000'000'000ULL, 450'000'000ULL, 700'000'000ULL, true);
    RC_CHECK(c.source == q::HipDenseCostSource::calibrated, "valid walls -> calibrated");
    RC_CHECK(c.bucket_ms[0] == 70.0 && c.bucket_ms[1] == 72.0 && c.bucket_ms[2] == 72.0 && c.bucket_ms[4] == 130.0, "running max");
    RC_CHECK(std::fabs(c.draft_a - 450.0 * 70.0 / 14000.0) < 1e-9 && std::fabs(c.draft_b - 700.0 * 70.0 / 14000.0) < 1e-9 &&
                 c.catch_up == c.draft_a,
             "draft prices = bytes at the measured stream rate (a %.4f b %.4f)", c.draft_a, c.draft_b);
    for (std::uint32_t r = 1; r <= 16; ++r)
        RC_CHECK(q::hip_dense_verify_ms(c, r) == c.bucket_ms[q::hip_dense_bucket_index(r)], "bucket pricing r %u", r);
    const double bad_walls[5] = {70.0, NAN, 71.0, 90.0, 130.0};
    RC_CHECK(q::hip_dense_calibrated_costs(bad_walls, 1ULL, 1ULL, 1ULL, true).source == q::HipDenseCostSource::table, "NaN -> table");
    RC_CHECK(q::hip_dense_calibrated_costs(walls, 0ULL, 1ULL, 1ULL, true).source == q::HipDenseCostSource::table, "no bytes -> table");
    const auto nomtp = q::hip_dense_calibrated_costs(walls, 1ULL, 0ULL, 1ULL, true);
    RC_CHECK(nomtp.draft_a == q::kHipDenseDraftA, "no MTP layer: draft prices unused, table kept");

    // 4. Measurement protocol on a stand-in engine: discarded windows (commit 0), warm pass + timed pass, verify's request.
    FakeEngine e;
    const auto m = q::hip_dense_measure_costs(e, q::HipDenseCostMode::calibrated, true, 7U);
    RC_CHECK(e.windows == std::vector<std::uint32_t>({1, 2, 4, 8, 16, 1, 2, 4, 8, 16}) && e.commits0 == 10U && e.syncs == 10U &&
                 e.all_rows == 10U,
             "measurement windows");
    RC_CHECK(m.source == q::HipDenseCostSource::calibrated, "stand-in walls are positive: calibrated");
    FakeEngine e2;
    RC_CHECK(q::hip_dense_measure_costs(e2, q::HipDenseCostMode::table, true, 7U).source == q::HipDenseCostSource::table &&
                 e2.windows.empty(),
             "table mode runs no window");

    // 5. Block config: the same (plan, head fraction, costs) give the same config; costs reach the draft seeds.
    const auto b1 = q::hip_dense_spec_block_config(16U, true, true, 0.25, c);
    const auto b2 = q::hip_dense_spec_block_config(16U, true, true, 0.25, c);
    RC_CHECK(same(b1, b2) && b1.fixed_costs && b1.chain_max_k == 15U, "deterministic config");
    RC_CHECK(std::fabs(b1.chain_seed_step_ms - (c.draft_a + 0.25 * c.draft_b)) < 1e-12 &&
                 std::fabs(b1.chain_seed_flush_ms - b1.chain_seed_step_ms - c.catch_up) < 1e-12,
             "draft seeds from the costs");
    RC_CHECK(same(q::hip_dense_spec_block_config(16U, true, true, 0.25), q::hip_dense_spec_block_config(16U, true, true, 0.25, table)),
             "table overload = table costs");

    // 6. Source contract: both entry points use the shared parser, resolver, costs and block config, and neither keeps a
    //    second parse of the shared flags or its own defaults branch.
    const std::string root = STRIX_SPEC_RESOLVER_SOURCE_DIR;
    for (const char* app : {"/apps/hip_qwen35_decode_main.hip", "/apps/hip_qwen35_spec_main.hip"}) {
        const std::string src = read(root + app);
        for (const char* need : {"parse_hip_dense_spec_arg(a, next, spec_args)", "resolve_hip_dense_spec(model, spec_args",
                                 "hip_dense_measure_costs(*E, plan.costs", "hip_dense_block_config(plan, head_fraction,",
                                 "hip_dense_seed_verify(drafter, bc.max_block, bc.fixed_costs,"})
            RC_CHECK(src.find(need) != std::string::npos, "%s: missing shared call %s", app, need);
        for (const char* banned : {"a == \"--draft-rows\"", "a == \"--spec-costs\"", "a == \"--draft-pack\"", "sp::BlockConfig bc;",
                                   "drafter.seed_verify_ms("})
            RC_CHECK(src.find(banned) == std::string::npos, "%s: a second defaults branch (%s)", app, banned);
    }
    std::printf("hip dense spec resolver: all checks passed\n");
    return 0;
}
