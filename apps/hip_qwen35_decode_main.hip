// strix-hip-qwen35-decode: Qwen3.8-27B (qwen35 dense) decode on one HIP device over strix::qwen35::HipEngine.
// Prompt goes through the engine's resolved forward_commit windows (shared plan: strix/qwen35/prefill_windows.hpp; logits
// on the last window only; --rows / --dump-logits / --tf-rows keep their diagnostic row paths), then greedy decode:
// exact MTP speculation (shared BlockDrafter + MtpChain, verify windows bitwise equal to one row per step) when the
// shared resolver finds an MTP layer
// (hip_spec_defaults.hpp: --mtp auto|off|FILE, --draft-rows on|off), else one row per step. --dump-logits keeps one row per step.
// Home: specific = app glue; engine in include/strix/qwen35/hip_engine.hpp, kernels shared.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include <algorithm>
#include <cmath>
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
#include "strix/device/set.hpp"
#include "strix/pair/planner.hpp"
#include "strix/pair/split_plan.hpp"
#include "strix/pair/tp_inputs.hpp"
#include "strix/qwen35/hip_engine.hpp"
#include "strix/qwen35/prefill_windows.hpp"
#include "strix/profiles/qwen_dense_hip.hpp"
#include "strix/qwen35/hip_spec_defaults.hpp"
#include "strix/spec/engine_target.hpp"
#include "../src/core/dense_linear.hpp"
#include "../src/core/dense_linear_iu4.hpp"
#include "../src/xport/tp_gather_hip.hpp"
#include "strix/env_flags.hpp"
#include "strix/gguf_reader.hpp"
#include "device/hip_cost_identity.hpp"

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
    const auto topo_cli = strix::device::take_topology_command_line(argc, argv);  // --topology FILE --node ID (TP rates)
    using namespace strix::qwen35;
    const hip_profile::Command command(hip_profile::App::decode, argc, argv);
    const auto& args = command.args;
    bool inspect = false;
    std::string model, dump, ids_arg = "1";
    HipDenseSpecArgs spec_args;  // the family's shared speculation flags (hip_dense_spec_costs.hpp)
    PrefillFlags pf;  // prompt windows: engine defaults unless a width / row path is asked (prefill_windows.hpp)
    unsigned gen = 16, maxpos = 0, repeat = 1;
    int tp_rank = -1;
    std::string tp_spec = "usb4auto";
    std::vector<double> tp_rates;  // --tp-rates override, GB/s per rank; default: the topology declaration (shares follow the ratio)
    double tp_trip_us = 0.0;       // --tp-trip-us override of the measured trip latency; 0 = measured at start-up
    double tp_gap_us = -1.0;       // --tp-gap-us: measured host gap per join of the staged route (a receipt value); < 0 = unknown
    bool tp_force = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto& a = args[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= args.size()) throw std::invalid_argument("missing value for " + a);
            return args[++i];
        };
        if (a == "--print-effective-config") inspect = true;
        else if (a == "--model") model = next();
        else if (a == "--prompt-ids") ids_arg = next();
        else if (a == "--gen") gen = strix::core::parse_u32(next());
        else if (a == "--rows") pf.rows = strix::core::parse_u32(next());  // diagnostic: prompt on the decode row path, windows of N rows
        else if (hip_profile::take_window_flag(hip_profile::App::decode, a, [&] { return strix::core::parse_u32(next()); }, pf)) {}
        else if (a == "--dump-logits") { dump = next(); pf.dump = true; }
        else if (a == "--dump-window") pf.dump_window = true;  // wide path with --dump-logits: every row of the last wide window
        else if (a == "--tf-rows") pf.tf_rows = strix::core::parse_u32(next());  // wide path: the last N prompt tokens as <= 16-row windows (row path)
        else if (a == "--iu4-reference") strix::core::dense_iu4::set_reference(true);  // exactness gate: dequantised-pack reference form
        else if (a == "--max-pos") maxpos = strix::core::parse_u32(next());
        else if (a == "--repeat") repeat = std::max(1U, strix::core::parse_u32(next()));  // prefills per process (reset between): first = cold, last = warm
        else if (a == "--no-mtp") spec_args.mtp = "off";
        else if (parse_hip_dense_spec_arg(a, next, spec_args)) {}  // --mtp, --draft-rows, --draft-pack, --spec-costs
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
        // The staged route returns to the host inside every gather: its host gap per join (wake-up + relaunch) is priced
        // only when measured (e.g. a previous run's in-step link counters); unknown refuses the split (planner rule).
        else if (a == "--tp-gap-us") {
            tp_gap_us = std::stod(next());
            if (!std::isfinite(tp_gap_us) || tp_gap_us < 0.0) throw std::invalid_argument("--tp-gap-us must be finite and >= 0");
        }
        else if (a == "--tp-force") tp_force = true;
        else throw std::invalid_argument("unknown flag " + a);
    }
    // The profile resolves for this run (before load): prompt length, device class and product, the IU4 pack, one rank.
    if (!ids_arg.empty() && ids_arg[0] == '@') {  // long prompts exceed the per-argument limit of the OS
        std::ifstream f(ids_arg.substr(1));
        if (!f) throw std::invalid_argument("cannot read " + ids_arg.substr(1));
        ids_arg.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        while (!ids_arg.empty() && std::isspace(static_cast<unsigned char>(ids_arg.back()))) ids_arg.pop_back();
    }
    const auto prompt = parse_ids(ids_arg);
    const std::string device_product = inspect ? std::string{} : strix::device::hip_cost_device_identity();
    const std::string device_class = device_product.substr(0, std::min(device_product.find('|'), device_product.find(':')));
    bool iu4_pack = false;
    try {
        if (!inspect && !model.empty())
            for (const auto& t : strix::gguf::read_file(model).tensors) iu4_pack = iu4_pack || t.ggml_type == strix::gguf::type_strix_iu4_g64;
    } catch (const std::exception&) {
        iu4_pack = false;  // unreadable header: the load below reports it; no FAST cell qualifies
    }
    const std::uint64_t capacity = maxpos ? std::uint64_t{maxpos} : prompt.size() + gen + HipEngine::max_rows();
    const std::string settings = "mtp=" + spec_args.mtp + "|tp=" + (tp_rank < 0 ? "off" : "on");
    const std::string execution = inspect ? std::string{} : strix::profile::ladder_execution({"hip", device_product,
        {prompt.size(), capacity, 1U, static_cast<std::uint32_t>(std::min<std::size_t>(prompt.size(), 0xFFFFFFFFU)), 0U, 0U, 0U, gen},
        settings});
    auto profile_inputs = command.inputs;
    profile_inputs.regime = {strix::profile::Phase::prefill, prompt.size(), 1U, device_class, execution};
    const auto profile = hip_profile::resolve_prefill(hip_profile::App::decode, profile_inputs, pf, HipEngine::max_rows(),
                                                      {tp_rank < 0, iu4_pack});
    if (hip_profile::finish(hip_profile::App::decode, profile, inspect)) return 0;
    hip_profile::apply_window(hip_profile::App::decode, profile, pf);  // the window the profile applied runs
    if (tp_rank > 1) throw std::invalid_argument("--tp-rank 0|1");
    // Tensor-parallel decode: every rank must pick the same verify windows, so the speculation costs are the compiled
    // table (identical bits on every rank; calibrated or learned walls are per rank and could differ). An explicit
    // calibrated / learned request contradicts that and is refused; the default resolves to the table and says so.
    if (tp_rank >= 0 && spec_args.costs_explicit && spec_args.costs != HipDenseCostMode::table)
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
    if (model.empty() || pf.rows > HipEngine::max_rows())
        throw std::invalid_argument("usage: --model GGUF [--prompt-ids 1,2,3] [--gen N] [--rows 1..16 (row-path diagnostic)] [--dump-logits FILE] [--iu4-reference] [--mtp auto|off|F] [--draft-rows on|off] [--draft-pack F1,F2] [--spec-costs calibrated|table|learned] [--wide N | --wide-exact N] [--tf-rows N] [--dump-window] [--max-pos N] [--repeat N]; --prompt-ids @FILE reads the list from FILE");
    if (prompt.empty()) throw std::invalid_argument("empty prompt");
    strix::core::require_capacity(prompt.size(), gen, 0xFFFFFFFFULL);
    const auto required = prompt.size() + gen;
    if (!maxpos) maxpos = static_cast<unsigned>(required) + HipEngine::max_rows();  // + one verify window (speculation)
    strix::core::require_capacity(0, required, maxpos);

    HipEngineOptions opt;
    opt.max_positions = maxpos;
    apply_prefill_flags(pf, HipEngine::max_rows(), opt);
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
        tpg = std::make_unique<strix::xport::tp_gather::HipPairGather>(std::move(opened), std::uint32_t(tp_rank), cap,
                                                                       static_cast<hipStream_t>(E->stream_handle()),
                                                                       strix::envcfg::flag("STRIX_TP_CHANNEL", true));
    }
    const unsigned V = E->vocab();
    // Speculation: one shared resolver (hip_spec_defaults.hpp). --dump-logits needs every position's logits: one row per step.
    auto plan = resolve_hip_dense_spec(model, spec_args, !dump.empty());
    std::unique_ptr<HipMtpChain> chain;
    std::unique_ptr<strix::spec::DraftRows> drows;
    if (!plan.mtp_source.empty()) {
        E->load_mtp(plan.mtp_source);
        chain = std::make_unique<HipMtpChain>(*E, false, HipEngine::mtp_max_rows());
        chain->set_device_feed(plan.device_feed);  // --chain-feed (shared spec flags)
        if (plan.draft_rows) drows = std::make_unique<strix::spec::DraftRows>(hip_dense_draft_rows_config(V, plan));
    }
    std::printf("spec: %s (mtp %s%s%s), draft rows %s, chain feed %s\n", chain ? "mtp" : "off", plan.mtp_reason.c_str(),
                plan.mtp_source.empty() ? "" : " ", plan.mtp_source.c_str(), drows ? "subset" : "full",
                chain && chain->device_feed() ? "device" : "host");
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
    // Prompt windows (shared plan, prefill_windows.hpp): chunked by the width the engine resolved, logits only where the
    // plan asks (the last window; every row on the diagnostic dump paths).
    const PrefillPlan pplan = plan_prefill(pf, prompt.size(), E->prefill_rows(), HipEngine::max_rows());
    std::size_t logit_windows = 0;
    for (const auto& w : pplan.windows) logit_windows += w.tops() ? 1U : 0U;
    std::printf("prefill windows: requested %s, resolved %u rows, %s path, %zu windows, logits on %zu\n",
                pplan.requested ? std::to_string(pplan.requested).c_str() : "engine default", pplan.width,
                pplan.row_path ? "row (diagnostic)" : "forward_commit", pplan.windows.size(), logit_windows);
    std::vector<std::uint32_t> ptop(std::max(E->prefill_rows(), HipEngine::max_rows()));
    std::uint32_t first_top = 0;  // the greedy token after the prompt
    double pf_ms = 0.0;
    for (unsigned pass = 0; pass < repeat; ++pass) {
    if (pass) { E->reset(); dumped.clear(); }
    if (chain) chain->reset();
    const auto t_pf0 = std::chrono::steady_clock::now();
    for (const auto& w : pplan.windows) {
        const auto o = E->forward_commit(uprompt.data() + w.at, w.rows, w.want);
        if (const unsigned m = w.tops()) {
            // A window of more rows than the decode window returns every row's logits but only the last row's argmax
            // (engine: the all-rows head of a wide window); read what it returns.
            const unsigned tops = m > HipEngine::max_rows() ? 1U : m;
            E->read_top(o, tops, ptop.data());
            first_top = ptop[tops - 1U];
            if (!dump.empty()) {
                const auto at = dumped.size();
                dumped.resize(at + std::size_t(m) * V);
                E->read_logits(o, m, dumped.data() + at);
            }
        }
        const bool last = w.at + w.rows == prompt.size();
        if (chain) chain->observe_prefill(o, w.rows, uprompt.data() + w.at, last ? first_top : uprompt[w.at + w.rows]);
    }
    pf_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pf0).count();
    if (repeat > 1U)
        std::printf("prefill %u/%u (%s): %zu tokens %.1f ms = %.1f tok/s\n", pass + 1U, repeat,
                    pass == 0U ? "cold" : pass + 1U == repeat ? "warm" : "mid", prompt.size(), pf_ms, double(prompt.size()) * 1000.0 / pf_ms);
    }
    std::printf("prompt %zu tokens: %.1f ms\n", prompt.size(), pf_ms);
    std::vector<int> generated{int(first_top)};
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
    // TP placement, decided once from measurements by the shared planner (strix/pair/planner.hpp): the one-node pass is
    // measured now on both ranks (each holds the whole pack; timed windows, discarded), agreed max; the stage model's
    // residual over it (launches, host gaps) is priced as fixed work on every rank (split::with_measured_single); the trip
    // is the start-up probe of the route the decode runs (chained gathers, the step's form), agreed by both ranks. A losing
    // plan keeps both ranks decoding one node (the same ids); --tp-force runs the split anyway.
    bool tp_on = false;
    double tp_pred_ms = 0.0;
    if (tp_rank >= 0) {
        namespace psp = strix::pair::split;
        namespace ppl = strix::pair::planner;
        // One decode pass as split stages (strix/pair/split_plan.hpp): weights + recurrent state + KV read once, every
        // projection split by column shares, 4 all-gathers per layer + the argmax pick (exact joins).
        const std::uint32_t layers = E->config().layers;
        const double pass_bytes = double(E->weight_bytes()) + double(E->state_bytes_per_pass()) + double(E->kv_bytes_per_position()) * double(E->committed());
        // Joined row bytes of the 4 gathers per layer (FP16 core|ctx and mlp, FP32 hid twice), averaged per gather.
        const auto& cf = E->config();
        const double gather_row = (8.0 * cf.hidden + 2.0 * (double(cf.heads) * cf.head_dim + cf.intermediate)) / 4.0;
        auto stages = psp::uniform_layers(layers, pass_bytes / double(layers), 1024U, 4U, gather_row, 1U);
        std::vector<double> bpms;
        for (const double g : tp_rates) bpms.push_back(g * 1e6);  // GB/s -> bytes/ms
        const double one_ms = tpg->agree_max(hip_dense_measure_pass(*E, std::uint32_t(generated[0])));
        const auto cal = psp::with_measured_single(stages, psp::Pass{}, *std::min_element(bpms.begin(), bpms.end()), one_ms);
        const auto lc = tpg->measure_link(std::uint64_t(gather_row));
        const auto trip = strix::pair::tp::trip_cost(lc, tp_trip_us);
        std::printf("tp calibration: one node pass %.1f ms measured (timed windows, agreed max), stage model %.1f ms, fixed residual %.1f ms\n",
                    one_ms, cal.model_ms, cal.fixed_ms);
        std::printf("tp link cost: %s; trip %.1f us%s, %s\n", lc.describe().c_str(), trip.latency_ms * 1000.0,
                    tp_trip_us > 0.0 ? " (--tp-trip-us)" : " (measured)", tpg->link().route_label().c_str());
        // The probe is the step's join form (gathers enqueued back to back behind work, no host wait): the in-token class.
        auto set = strix::device::rank_set(bpms, strix::device::FactSource::declared, "tp");
        strix::dist::LinkCost tc = lc;
        tc.latency_us = trip.latency_ms * 1000.0;
        strix::device::set_measured(set, 0U, 1U, strix::device::GapClass::in_token, tc);
        ppl::Costs pc;
        pc.set = &set;
        pc.route = tpg->link().route_for(std::uint64_t(gather_row));  // the route the step's gathers take
        if (tp_gap_us >= 0.0) {
            pc.gap_measured[static_cast<std::size_t>(strix::pair::GatherRoute::staged)] = true;
            pc.segment_gap_us[static_cast<std::size_t>(strix::pair::GatherRoute::staged)] = tp_gap_us;
        }
        ppl::Request rq;
        rq.regime = ppl::Regime::decode_plain;  // the plain pass decides (verify windows scale the same split)
        rq.forced = tp_force;
        rq.force = ppl::Form::tensor;
        const auto tp_plan = ppl::plan(stages, pc, rq);
        std::printf("%s\n", tp_plan.line.c_str());
        for (const auto& l : ppl::refusal_lines(tp_plan)) std::printf("%s\n", l.c_str());
        tp_on = tp_plan.chosen.form == ppl::Form::tensor;
        tp_pred_ms = tp_plan.all[static_cast<std::size_t>(ppl::Form::tensor)].ms;
        std::printf("tp placement: one rank %.1f ms/pass, %zu ranks %.1f ms/pass -> %s\n", tp_plan.single_ms, tp_rates.size(), tp_pred_ms,
                    tp_on ? (tp_force ? "decode on all ranks (forced)" : "decode on all ranks") : "decode stays on one node (both ranks)");
        if (tp_on) {
            E->enable_tp(std::uint32_t(tp_rank), tp_rates, HipEngine::TpGather{&tpg->link(), &strix::xport::tp_gather::HipPairGather::Link::fn}, !dump.empty());
            std::printf("%s\n", E->tp_receipt().c_str());
        }
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
    if (tpg) tpg->finish();  // the channel's last gathers are checked before any result is printed
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
        if (tp_on) {
            const double passes = steps ? double(steps) : double(generated.size() - 1U);
            // link = host time in the exchange call (staged: producer tail + copy + wire; device routes: enqueue only);
            // rest = everything else of the pass wall.
            const auto& tl = tpg->link();
            const double n = double(tl.calls());
            std::printf("tp link: %llu gathers, %.1f per pass, route %s, link %.2f ms/pass (%.1f us/gather), rest %.2f ms/pass\n",
                        (unsigned long long)tl.calls(), n / passes, tl.route_label().c_str(),
                        tl.link_ms() / passes, n > 0.0 ? tl.link_ms() * 1000.0 / n : 0.0, (ms - tl.link_ms()) / passes);
            // the planner's prediction is per plain pass: compared only when every step is one token (no speculation)
            if (!steps) std::printf("plan_check: form=tensor pred_ms=%.1f measured_ms=%.1f err=%+.1f %%\n", tp_pred_ms, per, 100.0 * (tp_pred_ms - per) / per);
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
