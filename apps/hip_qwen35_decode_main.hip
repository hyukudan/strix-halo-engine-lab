// strix-hip-qwen35-decode: Qwen3.8-27B (qwen35 dense) decode on one HIP device over strix::qwen35::HipEngine.
// Prompt goes through windows of up to --rows tokens (forward_commit), then greedy decode: exact MTP speculation (shared
// BlockDrafter + MtpChain, verify windows bitwise equal to one row per step) when the shared resolver finds an MTP layer
// (hip_spec_defaults.hpp: --mtp auto|off|FILE, --draft-rows on|off), else one row per step. --dump-logits keeps one row per step.
// Home: specific = app glue; engine in include/strix/qwen35/hip_engine.hpp, kernels shared.
#include <algorithm>
#include <fstream>
#include <iterator>
#include <memory>
#include <span>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "strix/device/topology.hpp"
#include "strix/dist/open_transport.hpp"
#include "strix/pair/split_plan.hpp"
#include "strix/pair/tp_inputs.hpp"
#include "strix/qwen35/hip_engine.hpp"
#include "strix/profiles/qwen_dense_hip.hpp"
#include "strix/qwen35/hip_spec_defaults.hpp"
#include "strix/spec/engine_target.hpp"
#include "../src/core/dense_linear.hpp"
#include "../src/core/dense_linear_iu4.hpp"
#include "../src/xport/tp_gather_hip.hpp"

namespace {
std::string sysfs_clock() {
    for (int c = 0; c < 4; ++c) {
        std::ifstream f("/sys/class/drm/card" + std::to_string(c) + "/device/pp_dpm_sclk");
        std::string line, best;
        while (std::getline(f, line))
            if (line.find('*') != std::string::npos) best = line;
        if (!best.empty()) return best;
    }
    return "0: 0Mhz *";
}
int sysfs_busy() {
    for (int c = 0; c < 4; ++c) {
        std::ifstream f("/sys/class/drm/card" + std::to_string(c) + "/device/gpu_busy_percent");
        int v;
        if (f >> v) return v;
    }
    return -1;
}
std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> r;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ','))
        if (!t.empty()) {
            const auto id = strix::core::parse_u32(t);
            if (id > 0x7FFFFFFFU) throw std::invalid_argument("token ID exceeds signed index range");
            r.push_back(static_cast<int>(id));
        }
    return r;
}
}  // namespace

int main(int argc, char** argv) try {
    strix::qwen35::hip_profile::take(strix::qwen35::hip_profile::App::decode, argc, argv);  // --profile exact|fast
    const auto topo_cli = strix::device::take_topology_command_line(argc, argv);  // --topology FILE --node ID (TP rates)
    using namespace strix::qwen35;
    std::string model, dump, ids_arg = "1";
    HipDenseSpecArgs spec_args;  // the family's shared speculation flags (hip_dense_spec_costs.hpp)
    unsigned wide = 0;
    bool wide_exact = false;
    bool dump_window = false;
    unsigned tf_rows = 0;
    unsigned gen = 16, rows = 1, maxpos = 0, repeat = 1;
    int tp_rank = -1;
    std::string tp_spec = "usb4auto";
    std::vector<double> tp_rates;  // --tp-rates override, GB/s per rank; default: the topology declaration (shares follow the ratio)
    double tp_trip_us = 0.0;       // --tp-trip-us override of the measured trip latency; 0 = measured at start-up
    bool costs_flag = false;       // --spec-costs given explicitly
    bool tp_force = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--prompt-ids") ids_arg = next();
        else if (a == "--gen") gen = strix::core::parse_u32(next());
        else if (a == "--rows") rows = strix::core::parse_u32(next());
        else if (a == "--wide") wide = strix::core::parse_u32(next());  // prompt windows of N rows, last-row logits only (N > 16 = FAST wide prefill)
        else if (a == "--wide-exact") { wide = strix::core::parse_u32(next()); wide_exact = true; }  // the same windows, EXACT class (bitwise to --wide 16)
        else if (a == "--dump-logits") dump = next();
        else if (a == "--dump-window") dump_window = true;  // with --wide and --dump-logits: every row of the last wide window
        else if (a == "--tf-rows") tf_rows = strix::core::parse_u32(next());  // with --wide: the last N prompt tokens as <= 16-row windows (row path), all logits
        else if (a == "--iu4-reference") strix::core::dense_iu4::set_reference(true);  // exactness gate: dequantised-pack reference form
        else if (a == "--max-pos") maxpos = strix::core::parse_u32(next());
        else if (a == "--repeat") repeat = std::max(1U, strix::core::parse_u32(next()));  // prefills per process (reset between): first = cold, last = warm
        else if (a == "--no-mtp") spec_args.mtp = "off";
        else if (parse_hip_dense_spec_arg(a, next, spec_args)) costs_flag = costs_flag || a == "--spec-costs";  // --mtp, --draft-rows, --draft-pack, --spec-costs
        // Exact tensor-parallel decode over 2 ranks (HipEngine::enable_tp): each rank loads the model and prefills alone
        // (replicated, same bits), then decodes its column / head shares; ids == the one-rank ids. Rates = measured
        // per-rank read speed (column shares); spec needs fixed costs (both ranks must choose the same windows).
        else if (a == "--tp-rank") tp_rank = int(strix::core::parse_u32(next()));
        else if (a == "--tp-transport") tp_spec = next();  // usb4auto (default) | tcp-listen:PORT | tcp-connect:HOST:PORT
        else if (a == "--tp-rates") {
            tp_rates.clear();
            std::stringstream ss(next());
            for (std::string f; std::getline(ss, f, ',');) tp_rates.push_back(std::stod(f));
        }
        // Placement: the shared split plan (strix/pair/split_plan.hpp) of the group step (bytes / declared read rate x
        // share + trips x the trip cost measured at start-up over the gather form, strix/pair/tp_inputs.hpp) against the
        // best single rank; decode stays on one node when the group does not win, unless --tp-force.
        else if (a == "--tp-trip-us") tp_trip_us = std::stod(next());
        else if (a == "--tp-force") tp_force = true;
        else throw std::invalid_argument("unknown flag " + a);
    }
    if (tp_rank > 1) throw std::invalid_argument("--tp-rank 0|1");
    // Tensor-parallel decode: every rank must pick the same verify windows, so the speculation costs are the compiled
    // table (identical bits on every rank; calibrated or learned walls are per rank and could differ). An explicit
    // calibrated / learned request contradicts that and is refused; the default resolves to the table and says so.
    if (tp_rank >= 0 && costs_flag && spec_args.costs != HipDenseCostMode::table)
        throw std::invalid_argument("--spec-costs calibrated|learned with --tp-rank: the ranks must agree on windows (use --spec-costs table)");
    if (tp_rank >= 0) {
        spec_args.costs = HipDenseCostMode::table;
        // Read rate per rank: --tp-rates, else the topology declaration (memory_gbytes_ps of both nodes), else refused.
        std::vector<double> declared;
        if (!topo_cli.path.empty() || !strix::device::read_topology_environment().path.empty()) {
            const auto topo = strix::device::start_topology(topo_cli, tp_rank);
            if (topo.size() == 2U) for (const auto& nd : topo.nodes) declared.push_back(nd.memory_gbytes_ps);
        }
        const auto rr = strix::pair::tp::rank_rates(tp_rates, declared, 2U);
        tp_rates = rr.gbytes_ps;
        std::printf("tp rates: %.1f, %.1f GB/s (%s)\n", tp_rates[0], tp_rates[1], rr.source.c_str());
    }
    if (model.empty() || rows < 1U || rows > HipEngine::max_rows())
        throw std::invalid_argument("usage: --model GGUF [--prompt-ids 1,2,3] [--gen N] [--rows 1..16] [--dump-logits FILE] [--iu4-reference] [--mtp auto|off|F] [--draft-rows on|off] [--draft-pack F1,F2] [--spec-costs calibrated|table|learned] [--wide N] [--tf-rows N] [--dump-window] [--max-pos N] [--repeat N]; --prompt-ids @FILE reads the list from FILE");
    if (!ids_arg.empty() && ids_arg[0] == '@') {  // long prompts exceed the per-argument limit of the OS
        std::ifstream f(ids_arg.substr(1));
        if (!f) throw std::invalid_argument("cannot read " + ids_arg.substr(1));
        ids_arg.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        while (!ids_arg.empty() && std::isspace(static_cast<unsigned char>(ids_arg.back()))) ids_arg.pop_back();
    }
    const auto prompt = parse_ids(ids_arg);
    if (prompt.empty()) throw std::invalid_argument("empty prompt");
    strix::core::require_capacity(prompt.size(), gen, 0xFFFFFFFFULL);
    const auto required = prompt.size() + gen;
    if (!maxpos) maxpos = static_cast<unsigned>(required) + HipEngine::max_rows();  // + one verify window (speculation)
    strix::core::require_capacity(0, required, maxpos);

    HipEngineOptions opt;
    opt.max_positions = maxpos;
    opt.prefill_rows = wide;
    opt.wide_exact = wide_exact;
    const auto t_load0 = std::chrono::steady_clock::now();
    auto E = HipEngine::from_gguf(model, opt);
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
    std::printf("load time %.1f s\n", load_s);
    std::unique_ptr<strix::xport::tp_gather::HipPairGather> tpg;
    if (tp_rank >= 0) {  // after the load: the peer may load slower; open_pair waits up to its timeout
        auto opened = strix::dist::open_pair(tp_spec, std::uint32_t(tp_rank));
        if (!opened.receipt.empty()) std::printf("tp transport: %s\n", opened.receipt.c_str());
        // largest gather: 16 rows x the widest share (mlp ~I/2), or whole logits rows for --dump-logits
        const std::size_t cap = dump.empty() ? (2U << 20) : (16U << 20);
        tpg = std::make_unique<strix::xport::tp_gather::HipPairGather>(std::move(opened), std::uint32_t(tp_rank), cap);
    }
    const unsigned V = E->vocab();
    // Speculation: one shared resolver (hip_spec_defaults.hpp). --dump-logits needs every position's logits: one row per step.
    auto plan = resolve_hip_dense_spec(model, spec_args, !dump.empty());
    std::unique_ptr<HipMtpChain> chain;
    std::unique_ptr<strix::spec::DraftRows> drows;
    if (!plan.mtp_source.empty()) {
        E->load_mtp(plan.mtp_source);
        chain = std::make_unique<HipMtpChain>(*E, false, HipEngine::mtp_max_rows());
        if (plan.draft_rows) drows = std::make_unique<strix::spec::DraftRows>(hip_dense_draft_rows_config(V, plan));
    }
    std::printf("spec: %s (mtp %s%s%s), draft rows %s\n", chain ? "mtp" : "off", plan.mtp_reason.c_str(),
                plan.mtp_source.empty() ? "" : " ", plan.mtp_source.c_str(), drows ? "subset" : "full");
    std::vector<std::uint32_t> uprompt(prompt.begin(), prompt.end());
    std::vector<float> dumped;
    std::uint32_t top[HipEngine::max_rows()];
    auto forward = [&](unsigned n, const int* ids) {
        std::uint32_t t[HipEngine::max_rows()];
        for (unsigned i = 0; i < n; ++i) t[i] = std::uint32_t(ids[i]);
        const auto o = E->forward_commit(t, n, dump.empty() ? LogitRows::last : LogitRows::all);
        E->read_top(o, dump.empty() ? 1U : n, top);
        if (!dump.empty()) {
            const auto at = dumped.size();
            dumped.resize(at + std::size_t(n) * V);
            E->read_logits(o, n, dumped.data() + at);
        }
        return o;
    };
    unsigned last_n = 0;
    double pf_ms = 0.0;
    for (unsigned pass = 0; pass < repeat; ++pass) {
    if (pass) { E->reset(); dumped.clear(); }
    if (chain) chain->reset();
    const auto t_pf0 = std::chrono::steady_clock::now();
    if (wide) {
        // --tf-rows N: wide windows up to size - N (no logits), then N teacher-forced rows in <= 16-row windows (the
        // decode row path, every row's logits dumped): the state a FAST prefill hands to decode, scored on exact rows.
        const std::size_t wide_end = prompt.size() - std::min<std::size_t>(tf_rows, prompt.size() - 1U);
        for (std::size_t at = 0; at < wide_end;) {
            const unsigned n = unsigned(std::min<std::size_t>(wide, wide_end - at));
            std::vector<std::uint32_t> t(n);
            for (unsigned i = 0; i < n; ++i) t[i] = std::uint32_t(prompt[at + i]);
            const bool last = at + n == prompt.size();  // false when --tf-rows leaves a tail
            const bool rows_all = last && dump_window && !dump.empty();
            const auto o = E->forward_commit(t.data(), n, last ? (rows_all ? LogitRows::all : LogitRows::last) : LogitRows::none);
            if (last) {
                E->read_top(o, 1, top);
                const unsigned m = rows_all ? n : 1U;
                if (!dump.empty()) { dumped.resize(std::size_t(m) * V); E->read_logits(o, m, dumped.data()); }
            }
            if (chain) chain->observe_prefill(o, n, t.data(), last ? top[0] : uprompt[at + n]);
            at += n;
        }
        last_n = 1;
        for (std::size_t at = wide_end; at < prompt.size();) {
            last_n = unsigned(std::min<std::size_t>(HipEngine::max_rows(), prompt.size() - at));
            const auto o = forward(last_n, prompt.data() + at);
            if (chain) chain->observe_prefill(o, last_n, uprompt.data() + at, at + last_n == prompt.size() ? top[0] : uprompt[at + last_n]);
            at += last_n;
        }
    } else
    for (std::size_t at = 0; at < prompt.size();) {
        last_n = unsigned(std::min<std::size_t>(rows, prompt.size() - at));
        const auto o = forward(last_n, prompt.data() + at);
        if (chain) chain->observe_prefill(o, last_n, uprompt.data() + at, at + last_n == prompt.size() ? top[0] : uprompt[at + last_n]);
        at += last_n;
    }
    pf_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pf0).count();
    if (repeat > 1U)
        std::printf("prefill %u/%u (%s): %zu tokens %.1f ms = %.1f tok/s\n", pass + 1U, repeat,
                    pass == 0U ? "cold" : pass + 1U == repeat ? "warm" : "mid", prompt.size(), pf_ms, double(prompt.size()) * 1000.0 / pf_ms);
    }
    std::printf("prompt %zu tokens: %.1f ms\n", prompt.size(), pf_ms);
    std::vector<int> generated{int(dump.empty() ? top[0] : top[last_n - 1U])};
    // Speculation costs of this device (hip_spec_defaults.hpp hip_dense_measure_costs): after the prompt, before the decode
    // clock starts; discarded windows, the committed state is untouched.
    const bool scalar_x = strix::core::dense_iu4::scalar_x();
    HipDenseCosts costs = hip_dense_table_costs(scalar_x);
    if (chain) {
        const auto tc = std::chrono::steady_clock::now();
        costs = hip_dense_measure_costs(*E, plan.costs, scalar_x, std::uint32_t(generated[0]));
        std::printf("spec costs: %s (%s), %.0f ms\n", hip_dense_cost_mode_name(plan.costs),
                    costs.source == HipDenseCostSource::calibrated ? "measured on this device"
                    : tp_rank >= 0                                    ? "table (forced: ranks must agree)"
                                                                      : "compiled table, FALLBACK",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count());
    }
    if (tp_rank >= 0) {  // placement, then decode from here on: this rank's shares, the state sliced to its heads
        namespace psp = strix::pair::split;
        // One decode pass as split stages (strix/pair/split_plan.hpp): weights + recurrent state + KV read once, every
        // projection split by column shares, 4 all-gathers per layer + the argmax pick (exact joins).
        const std::uint32_t layers = E->config().layers;
        const double pass_bytes = double(E->weight_bytes()) + double(E->state_bytes_per_pass()) + double(E->kv_bytes_per_position()) * double(E->committed());
        // Joined row bytes of the 4 gathers per layer (FP16 core|ctx and mlp, FP32 hid twice), averaged per gather.
        const auto& cf = E->config();
        const double gather_row = (8.0 * cf.hidden + 2.0 * (double(cf.heads) * cf.head_dim + cf.intermediate)) / 4.0;
        const auto stages = psp::uniform_layers(layers, pass_bytes / double(layers), 1024U, 4U, gather_row, 1U);
        // Trip cost measured now over the gather form the decode runs, agreed by both ranks (identical bits).
        const auto lc = tpg->measure_link(static_cast<hipStream_t>(E->stream_handle()));
        const auto trip = strix::pair::tp::trip_cost(lc, tp_trip_us);
        std::printf("tp link cost: %s; trip %.1f us%s, %s\n", lc.describe().c_str(), trip.latency_ms * 1000.0,
                    tp_trip_us > 0.0 ? " (--tp-trip-us)" : " (measured)", strix::pair::gather_route_name(tpg->link().route()));
        std::vector<double> bpms;
        for (const double g : tp_rates) bpms.push_back(g * 1e6);  // GB/s -> bytes/ms
        const auto tp_plan = psp::plan(stages, psp::Pass{}, psp::Costs{bpms, trip, 0.0, false});
        const bool group_wins = tp_plan.tensor.admissible && tp_plan.tensor.ms < tp_plan.single.ms;
        std::printf("tp placement: one rank %.1f ms/pass, %zu ranks %.1f ms/pass (compute %.1f + %u trips %.1f) -> %s\n", tp_plan.single.ms,
                    tp_rates.size(), tp_plan.tensor.ms, tp_plan.tensor.compute_ms, tp_plan.tensor.trips, tp_plan.tensor.link_ms,
                    group_wins ? "decode on all ranks" : tp_force ? "group loses, forced" : "decode stays on one node");
        if (!group_wins && !tp_force) throw std::runtime_error("tp placement: the group does not win; run one node (or --tp-force)");
        E->enable_tp(std::uint32_t(tp_rank), tp_rates, HipEngine::TpGather{&tpg->link(), &strix::xport::tp_gather::HipPairGather::Link::fn}, !dump.empty());
        std::printf("%s\n", E->tp_receipt().c_str());
    }

    std::atomic<bool> run{true};
    std::atomic<long> clk_sum{0}, clk_n{0}, busy_sum{0};
    std::string last_clk;
    std::thread sampler([&] {
        while (run) {
            last_clk = sysfs_clock();
            clk_sum += std::atol(last_clk.c_str() + last_clk.find(' ') + 1);
            ++clk_n;
            busy_sum += std::max(0, sysfs_busy());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    std::uint64_t steps = 0, accepted = 0;
    std::string spec_lines;  // k histogram + strix-spec-stats (per-depth offered / accepted, verify walls per rows)
    const auto t0 = std::chrono::steady_clock::now();
    if (chain) {  // exact speculation: every window is verified; emitted tokens are the target's greedy tokens
        namespace sp = strix::spec;
        std::vector<std::uint32_t> out{std::uint32_t(generated[0])};
        if (drows) {  // the subset is chosen by the prompt first: its share of the head is the draft price (fixed costs)
            drows->begin(uprompt);
            drows->admit(std::span<const std::uint32_t>(out.data(), 1));
        }
        const double head_fraction = drows ? double(drows->rows()) / double(V) : 1.0;
        const auto bc = hip_dense_block_config(plan, head_fraction, costs);
        sp::BlockDrafter drafter(bc, chain.get());
        hip_dense_seed_verify(drafter, bc.max_block, bc.fixed_costs, costs);
        std::printf("spec config: costs %s (%s), verify ms r1 %.1f r2 %.1f r3 %.1f r4 %.1f r8 %.1f r16 %.1f (iu4 scalar_x %d), "
                    "draft flush %.2f step %.2f ms (head %.1f %%), chain cap %u, max block %u, sources copy%s%s, context cost %s\n",
                    hip_dense_cost_mode_name(plan.costs), costs.source == HipDenseCostSource::calibrated ? "calibrated" : "table",
                    hip_dense_verify_ms(costs, 1), hip_dense_verify_ms(costs, 2), hip_dense_verify_ms(costs, 3),
                    hip_dense_verify_ms(costs, 4), hip_dense_verify_ms(costs, 8), hip_dense_verify_ms(costs, 16), int(scalar_x),
                    bc.chain_seed_flush_ms, bc.chain_seed_step_ms, 100.0 * head_fraction, bc.chain_max_k, bc.max_block,
                    bc.chain ? ",chain" : "", bc.hybrid ? ",hybrid" : "", sp::context_cost_name(drafter.arbiter().config()).c_str());
        drafter.commit(uprompt);
        drafter.commit(std::span<const std::uint32_t>(out.data(), 1));
        std::uint64_t k_hist[HipEngine::max_rows()] = {};  // drafts verified per step
        E->set_draft_rows(drows.get());
        sp::EngineVerifyTarget<HipEngine> target(*E);
        target.set_observer([&](const HipRowsOut& o, std::uint32_t r) { chain->observe_verify(o, r); });
        sp::BlockSpeculator spec(drafter, target);
        spec.start(out[0]);
        while (out.size() < gen) {
            sp::StepParams p;
            p.remaining = gen - std::uint32_t(out.size());
            p.max_rows = HipEngine::max_rows();
            p.context = E->committed();
            const auto r = spec.step(out, p);
            if (r.emitted == 0) break;
            ++steps;
            accepted += r.accepted;
            ++k_hist[std::min<std::uint32_t>(r.proposed, HipEngine::max_rows() - 1U)];
            if (drows) drows->admit(std::span<const std::uint32_t>(out.data() + out.size() - r.emitted, r.emitted));
        }
        E->set_draft_rows(nullptr);
        generated.assign(out.begin(), out.end());
        std::string hist;
        for (std::uint32_t k = 0; k < HipEngine::max_rows(); ++k)
            if (k_hist[k]) hist += " " + std::to_string(k) + ":" + std::to_string(k_hist[k]);
        spec_lines = "k hist (drafts verified: steps):" + hist + "\n" + drafter.stats().line() + "\n";
    } else {
        for (unsigned s = 1; s < gen; ++s) {
            const int id = generated.back();
            forward(1, &id);
            generated.push_back(int(top[0]));
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    run = false;
    sampler.join();
    std::printf("generated:");
    for (int t : generated) std::printf(" %d", t);
    std::printf("\n");
    if (steps)
        std::printf("spec: %zu tokens in %llu steps, E %.2f tokens/step (accepted drafts %llu), %.2f ms/step%s\n", generated.size() - 1U,
                    (unsigned long long)steps, double(generated.size() - 1U) / double(steps), (unsigned long long)accepted,
                    ms / double(steps), drows ? (", " + drows->describe()).c_str() : "");
    std::fputs(spec_lines.c_str(), stdout);
    if (generated.size() > 1U) {
        const double per = ms / double(generated.size() - 1U);
        const double bytes = double(E->weight_bytes()) + double(E->state_bytes_per_pass()) + double(E->kv_bytes_per_position()) * double(E->committed());
        const double pass = steps ? ms / double(steps) : per;  // trunk passes: one per token, or one verify window per step
        std::printf("decode: %.2f ms/token, %.2f tok/s, %.1f GB/s (%.0f %% of 243), bytes/pass %.2f GB\n", per, 1000.0 / per,
                    bytes / pass / 1e6, bytes / pass / 1e6 / 243.0 * 100.0, bytes / 1e9);
        std::printf("gpu: sclk avg %ld MHz (last: %s), busy avg %ld %%\n", clk_n ? clk_sum / clk_n : 0L, last_clk.c_str(),
                    clk_n ? busy_sum / clk_n : 0L);
        if (tp_rank >= 0) {
            const double passes = steps ? double(steps) : double(generated.size() - 1U);
            // sync = host blocked in the stream wait before a staged exchange (GPU work in flight + drain); link = host
            // time in the exchange call (device routes: enqueue only); rest = everything else of the pass wall.
            const auto& tl = tpg->link();
            const double n = double(tl.calls());
            std::printf("tp link: %llu gathers, %.1f per pass, route %s, sync %.2f ms/pass, link %.2f ms/pass (%.1f us/gather), "
                        "rest %.2f ms/pass\n",
                        (unsigned long long)tl.calls(), n / passes, strix::pair::gather_route_name(tl.route()), tl.sync_ms() / passes,
                        tl.link_ms() / passes, n > 0.0 ? tl.link_ms() * 1000.0 / n : 0.0, (ms - tl.sync_ms() - tl.link_ms()) / passes);
        }
    }
    if (!dump.empty()) {
        std::ofstream f(dump, std::ios::binary);
        f.write(reinterpret_cast<const char*>(dumped.data()), std::streamsize(dumped.size() * 4U));
        std::printf("dumped %zu positions x %u to %s\n", dumped.size() / V, V, dump.c_str());
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
