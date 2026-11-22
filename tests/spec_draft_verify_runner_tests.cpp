#include "strix/session/draft_verify_runner.hpp"
#include "strix/graph/model_decl.hpp"

#include "executor_cpu_binding.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <optional>

namespace {

namespace ex = strix::exec;
namespace gr = strix::graph;
namespace se = strix::session;
namespace sp = strix::spec;
namespace st = strix::state;

void check(bool pass, const char* message) {
    if (!pass) throw std::runtime_error(message);
}

template <class Error, class Call>
void rejects(Call call, const char* message) {
    try {
        call();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error(message);
}

constexpr std::uint32_t noise = 700;
constexpr std::uint32_t slot = 3;
constexpr std::uint32_t initial_anchor = 11;
constexpr se::RingPlan geometry(4, 3, 8);

struct Image {
    std::array<float, 4> logits;
    std::uint32_t input;
    std::uint32_t carry;
};

struct Row {
    Image state;
    ex::RowOut pick;
};

struct Trace {
    std::map<ex::ProgramKey, unsigned> compiles;
    std::map<ex::ProgramKey, unsigned> verifies;
    unsigned drafts = 0;
    float confidence = 0.95F;
    std::optional<std::uint32_t> correction;
    std::optional<std::uint32_t> stop;
    bool fail_catchup = false;
};

Row serial_row(std::uint32_t token, std::uint32_t position, std::uint32_t previous, const Trace& trace) {
    const auto carry = std::rotl(previous, 5) ^ (token + position);
    const auto score = static_cast<float>(carry & 0xffU) * 0.125F;
    const auto pick = trace.correction == position ? token + 50U : token + 1U;
    const auto status = trace.stop == position ? ex::RowStatus::stop : ex::RowStatus::picked;
    return {{{score, score - 0.5F, -score, -0.0F}, token, carry}, {pick, score, score - 0.5F, status}};
}

using Runner = se::DraftVerifyRunner<Image, std::uint32_t>;
using Ring = Runner::Ring;

struct ScriptArgs {
    Trace* trace;
    const Ring* ring;
    ex::BufferId output;
    ex::ProgramKey key;
};

void script(std::span<const std::byte> raw, const ex::Grid&, const ex::CpuLaunchContext& ctx) {
    ScriptArgs args{};
    check(raw.size() == sizeof(args), "script args size");
    std::memcpy(&args, raw.data(), sizeof(args));
    auto& trace = *args.trace;
    const auto& desc = ctx.step;
    if (args.key.graph_id != 0) {
        if (args.key.phase == ex::Phase::draft) {
            ++trace.drafts;
            check(desc.h.rows == 8 && desc.row[7].token == noise, "P1 block input layout");
            for (std::uint32_t i = 1; i < desc.h.rows; ++i) {
                auto output = ctx.arena_row(args.output, i);
                std::memcpy(output.data(), &trace.confidence, sizeof(trace.confidence));
                ctx.picks[i] = {desc.row[0].token + i, 0.0F, 0.0F, ex::RowStatus::picked};
            }
        } else {
            check(args.key.phase == ex::Phase::state_only, "unexpected draft program phase");
            if (trace.fail_catchup) throw std::runtime_error("injected P1 catch-up failure");
            const auto token = desc.row[0].token;
            std::memcpy(ctx.arena_row(args.output, 0).data(), &token, sizeof(token));
        }
        return;
    }

    ++trace.verifies[args.key];
    const auto frontier = args.ring->committed();
    auto carry = frontier == 0 ? 0U : args.ring->convolution_at(frontier - 1U);
    for (std::uint32_t i = 0; i < desc.h.rows; ++i) {
        if (!ex::step_row_valid(desc, i)) continue;
        const auto& row = desc.row[i];
        check(row.slot == slot && row.position == frontier + i &&
              row.ancestors == (1U << i) - 1U && row.parent == (i ? i - 1U : ex::kStepNoParent) &&
              row.flags == (ex::step_valid | ex::step_pending_state), "X3 verify row topology");
        const auto output = serial_row(row.token, row.position, carry, trace);
        ctx.picks[i] = output.pick;
        std::memcpy(ctx.arena_row(args.output, i).data(), &output.state, sizeof(Image));
        carry = output.state.carry;
    }
}

constexpr gr::StageDecl stages[] = {
    {.id = "script", .op = gr::Op::argmax, .entry = "script", .in = {"token"}, .out = "row"},
};

sp::ChanceWindowTables tables() {
    sp::ChanceWindowTables costs;
    costs.calibration = {{0.0F, 0.0}, {0.5F, 1.0}};
    costs.verify_ms.resize(8);
    for (std::size_t i = 0; i < costs.verify_ms.size(); ++i)
        costs.verify_ms[i].fill(1.0 + static_cast<double>(i) / 10.0);
    costs.verify_ms[7].fill(1.4);
    return costs;
}

st::Layout draft_layout() {
    constexpr st::Decl decls[] = {{.id = "draft", .kind = st::Kind::draft,
                                  .dtype = st::Dtype::u32, .dims = {1}}};
    return st::layout(decls, gr::ModelDecl{}, 32, 1, {.max_pending = 8});
}

struct Fixture {
    Trace trace;
    std::array<Image, geometry.window_cells()> window{};
    std::array<std::uint32_t, geometry.convolution_cells()> convolution{};
    Ring ring{geometry, window, convolution};
    alignas(256) std::array<std::byte, 4096> arena{};
    ex::CpuExecutor executor{arena};
    gr::StageGraph graph{.before = stages};
    ex::Compiler compiler{
        graph, [](const ex::ProgramKey&, const gr::Instance&) {
            return ex::BufferLayout{ex::BufferStorage::arena, sizeof(Image), 0};
        },
        [this](const ex::LaunchBindingRequest& request) {
            ++trace.compiles[request.key];
            const ScriptArgs args{&trace, &ring, request.launch.stages.front().output, request.key};
            const auto raw = std::as_bytes(std::span(&args, 1));
            ex::KernelBinding binding;
            binding.identity = binding.exact_identity = "b1-test-script/v1";
            binding.grid_by_context = request.key.graph_id == 0;
            binding.args.assign(raw.begin(), raw.end());
            return ex::bind_cpu_reference(std::move(binding), script);
        }, {}, [](const gr::StageGraph& decl, const ex::ProgramKey& key) {
            return gr::launch_plan(decl, 0, key.rows, {});
        }
    };
    std::optional<st::CpuState> state;
    std::vector<Image> serial;
    std::uint32_t anchor = initial_anchor;
    bool unmask_padding = false;

    explicit Fixture(bool stateful = false) {
        if (stateful) state.emplace(draft_layout());
    }

    std::span<const std::byte> image(const ex::Program& program, std::uint32_t row) const {
        const auto& buffer = program.buffers[program.launches.front().stages.front().output];
        const auto offset = buffer.arena_offset + row * buffer.layout.row_stride;
        return std::span(arena).subspan(static_cast<std::size_t>(offset), sizeof(Image));
    }

    ex::DraftRunner drafter() {
        ex::DraftRunner::CpuBinding binding{compiler, executor, arena, {.graph_id = 1}};
        binding.confidence = [this](const ex::Program& program, std::span<const std::byte>, std::uint32_t row) {
            float confidence = 0.0F;
            std::memcpy(&confidence, image(program, row).data(), sizeof(confidence));
            return confidence;
        };
        if (state) {
            binding.state = &*state;
            binding.state_rows = [this](const ex::Program& program, st::Row, std::uint32_t row,
                                        std::span<st::Row> out) {
                out[0] = image(program, row).first(sizeof(std::uint32_t));
            };
        }
        return ex::DraftRunner({.id = "block", .kind = sp::DraftKind::block, .graph = "draft.block",
                                .block = 8, .noise_token = noise,
                                .attention = sp::DraftAttention::block_noncausal, .confidence = "p"},
                               std::move(binding));
    }

    Runner runner(ex::DraftRunner& draft) {
        return Runner(ring, anchor, slot, draft, compiler, tables(),
            [this](const ex::Program& program, const ex::StepBlock& desc, const Ring&,
                   std::span<ex::RowOut> picks, std::span<Image> rows, std::span<std::uint32_t> carry) {
                executor.wait(executor.submit(program, {std::span(desc.row).first(desc.h.rows)}), picks);
                check(std::memcmp(executor.descriptor(), &desc, sizeof(desc)) == 0, "CPU did not receive the X3 block");
                for (std::uint32_t i = 0; i < rows.size(); ++i) {
                    std::memcpy(&rows[i], image(program, i).data(), sizeof(Image));
                    carry[i] = rows[i].carry;
                }
                if (unmask_padding) picks.back().status = ex::RowStatus::picked;
            });
    }

    Row next() {
        const auto previous = serial.empty() ? 0 : serial.back().carry;
        const auto row = serial_row(anchor, static_cast<std::uint32_t>(serial.size()), previous, trace);
        serial.push_back(row.state);
        anchor = row.pick.id;
        return row;
    }

    void prime(std::size_t rows) {
        for (std::size_t i = 0; i < rows; ++i) {
            const auto row = next();
            ring.stage({&row.state, 1}, {&row.state.carry, 1});
            ring.commit_staged(1);
        }
    }

    void compare_state() const {
        check(ring.committed() == serial.size() && ring.staged_rows() == 0, "published target frontier");
        const auto keep = std::min(serial.size(), geometry.window());
        for (auto pos = serial.size() - keep; pos < serial.size(); ++pos)
            check(std::memcmp(&serial[pos], &ring.window_at(pos), sizeof(Image)) == 0, "target window bytes");
        const auto taps = std::min(serial.size(), geometry.taps());
        for (auto pos = serial.size() - taps; pos < serial.size(); ++pos)
            check(serial[pos].carry == ring.convolution_at(pos), "target convolution bytes");
        rejects<std::out_of_range>([this] { (void)ring.window_at(serial.size()); },
                                   "uncommitted row was visible");
    }

    void compare(const se::DraftVerifyStep& step, std::uint32_t width, std::uint32_t drafts, std::uint32_t kept) {
        check(step.decision.rows == width && step.decision.k == drafts && step.kept == kept,
              "unexpected ChanceWindow shape or acceptance");
        check(step.decision.padded == (width > drafts + 1U), "unexpected ChanceWindow padding");
        check(step.decision.source == (drafts == 0U ? sp::Source::serial : sp::Source::model), "wrong proposal source");
        for (const auto& pick : step.committed_picks()) {
            const auto expected = next().pick;
            check(pick.id == expected.id && pick.status == expected.status &&
                  std::bit_cast<std::uint32_t>(pick.top1) == std::bit_cast<std::uint32_t>(expected.top1) &&
                  std::bit_cast<std::uint32_t>(pick.top2) == std::bit_cast<std::uint32_t>(expected.top2),
                  "target pick or score differs from serial");
        }
        compare_state();
    }
};

void plan_test() {
    Fixture fixture;
    fixture.prime(2047);
    auto draft = fixture.drafter();
    auto runner = fixture.runner(draft);
    const auto& plan = runner.plan();
    check(std::ranges::equal(plan.widths(), std::array{1U, 2U, 4U, 8U}), "unexpected B1 widths");
    check(plan.keys().size() == 4 * sp::context_bins, "wrong B1 warm set");
    constexpr std::array<std::uint64_t, 5> boundaries{0, 2048, 8192, 32768, 131072};
    for (const auto width : plan.widths()) {
        for (std::size_t i = 0; i < boundaries.size(); ++i) {
            const auto& key = plan.key(width, boundaries[i]);
            check(key.ctx_bucket == i + 1 && key.rows == width && key.phase == ex::Phase::verify &&
                  key.graph_id == 0 && !key.sampled && !key.tree, "invalid bucket key");
            if (i != 0) check(plan.key(width, boundaries[i] - 1).ctx_bucket == i, "bucket changed early");
        }
    }
    fixture.compiler.warm(plan.keys());
    fixture.compare(runner.step(8, 2), 2, 1, 2);
    fixture.compare(runner.step(8, 2), 2, 1, 2);
    fixture.compare(runner.step(8, 2), 2, 1, 2);
    check(fixture.trace.verifies.at(plan.key(2, 2047)) == 1 &&
          fixture.trace.verifies.at(plan.key(2, 2048)) == 2, "verify dispatch did not follow the frontier bucket");
    for (const auto& key : plan.keys()) check(fixture.trace.compiles.at(key) == 1, "verify key was recompiled");

    constexpr auto max = std::numeric_limits<std::uint32_t>::max();
    check(plan.row_limit(max - 1, 8) == 1 && plan.row_limit(max, 8) == 0 &&
          plan.row_limit(0, 7) == 4, "X3 frontier or width clamp");
    check(plan.describe(1, max - 1, slot, initial_anchor, {}).row[0].position == max - 1, "last X3 position");
    rejects<std::invalid_argument>([&] { (void)plan.describe(2, max - 1, slot, initial_anchor, {}); },
                                   "X3 frontier overflow was accepted");
    const se::VerifyWindowPlan narrow(se::RingPlan(4, 3, 3));
    check(std::ranges::equal(narrow.widths(), std::array{1U, 2U}), "ring capacity did not restrict widths");
    std::puts("PASS B1 X3 verify keys and padding");
}

void greedy_test() {
    Fixture fixture;
    auto draft = fixture.drafter();
    auto runner = fixture.runner(draft);
    check(runner.step(0).kept == 0 && runner.step(8, 0).kept == 0 && fixture.trace.drafts == 0,
          "zero budget executed a pass");
    fixture.trace.correction = 0;
    fixture.compare(runner.step(8), 8, 7, 1);
    fixture.trace.correction = 2;
    fixture.compare(runner.step(8), 8, 7, 2);
    fixture.compare(runner.step(8), 8, 7, 8);
    check(fixture.trace.verifies.at(runner.plan().key(8, 0)) == 3, "target replayed after partial acceptance");

    fixture.unmask_padding = true;
    rejects<std::runtime_error>([&] { (void)runner.step(5); }, "unmasked padding was accepted");
    fixture.compare_state();
    fixture.unmask_padding = false;
    const auto first_padding = fixture.ring.committed() + 5U;
    const auto window_before = fixture.window;
    const auto convolution_before = fixture.convolution;
    fixture.compare(runner.step(5), 8, 4, 5);
    for (auto pos = first_padding; pos < first_padding + 3U; ++pos) {
        const auto wi = geometry.window_cell(pos);
        const auto ci = geometry.convolution_cell(pos);
        check(std::memcmp(&fixture.window[wi], &window_before[wi], sizeof(Image)) == 0 &&
              fixture.convolution[ci] == convolution_before[ci], "padding wrote a ring cell");
    }
    fixture.trace.confidence = 0.1F;
    fixture.compare(runner.step(8), 1, 0, 1);
    const auto prior_drafts = fixture.trace.drafts;
    fixture.compare(runner.step(1), 1, 0, 1);
    check(fixture.trace.drafts == prior_drafts, "one remaining token still drafted");
    fixture.trace.confidence = 0.95F;
    fixture.compare(runner.step(8, 3), 2, 1, 2);
    fixture.trace.stop = static_cast<std::uint32_t>(fixture.serial.size()) + 1U;
    const auto stop = runner.step(8, 4);
    fixture.compare(stop, 4, 3, 2);
    check(stop.stopped && runner.step(8).kept == 0, "matching stop did not end the stream");
    std::puts("PASS B1 greedy picks, logits and ring bytes match serial");
}

void retry_test() {
    Fixture fixture(true);
    auto draft = fixture.drafter();
    auto runner = fixture.runner(draft);
    fixture.compare(runner.step(8), 8, 7, 8);
    fixture.trace.fail_catchup = true;
    rejects<std::runtime_error>([&] { (void)runner.step(8); }, "P1 catch-up failure was ignored");
    fixture.compare_state();
    const auto& lane = fixture.state->entry(0, 0);
    check(lane.frontier == 8 && lane.pending == 0, "P1 changed its published frontier on failure");
    fixture.trace.fail_catchup = false;
    fixture.compare(runner.step(8), 8, 7, 8);
    check(lane.frontier == 16 && lane.pending == 0, "P1 frontier did not advance on retry");
    for (std::uint32_t i = 0; i < lane.frontier; ++i) {
        const auto bytes = fixture.state->committed_row(0, 0, i);
        check(bytes.size() == sizeof(std::uint32_t) &&
              std::memcmp(bytes.data(), &fixture.serial[i].input, bytes.size()) == 0, "P1 committed wrong input");
    }
    std::puts("PASS B1 failed catch-up leaves both frontiers ready to retry");
}

}  // namespace

int main() {
    try {
        plan_test();
        greedy_test();
        retry_test();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL B1 draft verify runner: %s\n", error.what());
        return 1;
    }
}
