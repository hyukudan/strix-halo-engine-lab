#include "strix/exec/generic_engine.hpp"
#include "strix/serve/session_runtime.hpp"
#include "strix/weight/registry.hpp"
#include "strix/weight/source.hpp"

#include "executor_cpu_binding.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <map>
#include <utility>

namespace {

namespace ex = strix::exec;
namespace gr = strix::graph;
namespace ss = strix::session;
namespace st = strix::state;
namespace sv = strix::serve;
namespace wt = strix::weight;

constexpr std::uint32_t vocabulary = 64;
constexpr gr::StageDecl stages[] = {
    {.id = "script", .op = gr::Op::argmax, .entry = "script", .in = {"token"}, .out = "history"},
};

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Error = std::invalid_argument, class Call>
void rejects(Call call, const char* message) {
    try {
        call();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error(message);
}

template <class T>
T read(std::span<const std::byte> bytes) {
    check(bytes.size() == sizeof(T), "script image size");
    T value;
    std::memcpy(&value, bytes.data(), sizeof(value));
    return value;
}

class Images final : public wt::ImageSink {
public:
    wt::Status stage(const wt::LoadImage& row, const wt::Shape&, std::span<const wt::Byte> bytes) override {
        check(row.name == "bias", "script image name");
        staged_ = read<float>(bytes);
        return {wt::Error::ok, {}};
    }

    wt::Status commit(std::span<std::uint32_t> checksums) override {
        check(checksums.size() == 1 && staged_ == 3.0F, "W2 staged the scripted bias bytes");
        bias = static_cast<std::uint32_t>(staged_);
        // CRC32C of the four little-endian bytes for 3.0f.
        checksums.front() = 0xf60b907fU;
        return {wt::Error::ok, {}};
    }

    void abort() noexcept override { staged_ = 0; }
    std::uint32_t bias = 0;

private:
    float staged_ = 0;
};

struct Trace {
    const st::CpuState* state = nullptr;
    std::map<ex::ProgramKey, std::uint32_t> compiled;
    std::vector<std::uint32_t> widths;
    bool fail_decode_epilogue = false;
};

struct ScriptArgs {
    Trace* trace;
    ex::BufferId output;
    std::uint32_t bias;
};

void script(std::span<const std::byte> raw, const ex::Grid&, const ex::CpuLaunchContext& ctx) {
    const auto args = read<ScriptArgs>(raw);
    args.trace->widths.push_back(ctx.step.h.rows);
    for (std::uint32_t i = 0; i < ctx.step.h.rows; ++i) {
        if (!ex::step_row_valid(ctx.step, i)) continue;
        const auto& row = ctx.step.row[i];
        std::uint32_t previous = 0;
        if (row.parent != ex::kStepNoParent) {
            check(row.parent < i && ctx.step.row[row.parent].slot == row.slot &&
                  (row.ancestors & (1U << row.parent)) != 0, "same-lane in-step parent");
            previous = read<std::uint32_t>(ctx.arena_row(args.output, row.parent));
        } else if (row.position) {
            previous = read<std::uint32_t>(args.trace->state->committed_row(0, row.slot, row.position - 1));
        }
        const auto next = (previous * 3U + row.token + row.position + args.bias) % vocabulary;
        std::memcpy(ctx.arena_row(args.output, i).data(), &next, sizeof(next));
        if ((row.flags & ex::step_sample) != 0)
            ctx.picks[i] = {(next + args.bias) % vocabulary, static_cast<float>(next), 0.0F, ex::RowStatus::picked};
    }
}

struct Fixture {
    Trace trace;
    std::unique_ptr<ex::GenericEngine> engine;

    explicit Fixture(std::uint32_t context = 8) {
        gr::ModelDecl decl;
        decl.name = "scripted-conformance";
        decl.identity = 0x5855;
        decl.stages.before = stages;
        decl.symbols.value[static_cast<std::size_t>(wt::tensor_map::Sym::vocab)] = vocabulary;

        const std::array bias{3.0F};
        const auto bytes = std::as_bytes(std::span(bias));
        constexpr auto f32 = wt::decoder_format(wt::DecoderFormat::f32).id;
        ex::GenericEngine::CpuBinding binding;
        binding.context = context;
        binding.slots = 3;
        binding.load.sites.push_back({1, "bias", f32, {{1, 0, 0, 0}, 1},
                                      {wt::memory_source(bytes), {0, bytes.size()}}});
        binding.load.images.push_back({"bias", {f32, f32, bytes.size(), "stored.v1"}, {0}});
        binding.images = std::make_unique<Images>();
        binding.states.push_back({.id = "history", .kind = st::Kind::history, .dtype = st::Dtype::u32,
                                  .dims = {1}, .rollback = st::Rollback::rewind});
        binding.plan = [](const gr::StageGraph& graph, const ex::ProgramKey& key) {
            return gr::launch_plan(graph, 0, key.rows, {});
        };
        binding.buffers = [](const ex::ProgramKey&, const gr::Instance&) {
            return ex::BufferLayout{ex::BufferStorage::arena, sizeof(std::uint32_t), 0};
        };
        binding.kernels = [this](const wt::ImageSink& loaded, const st::CpuState& state) {
            trace.state = &state;
            const auto value = static_cast<const Images&>(loaded).bias;
            check(value == 3, "CPU resolver sees published W2 images");
            return [this, value](const ex::LaunchBindingRequest& request) {
                ++trace.compiled[request.key];
                const ScriptArgs args{&trace, request.launch.stages.front().output, value};
                const auto raw = std::as_bytes(std::span(&args, 1));
                ex::KernelBinding result;
                result.identity = result.exact_identity = "scripted-history/v1";
                result.args.assign(raw.begin(), raw.end());
                return ex::bind_cpu_reference(std::move(result), script);
            };
        };
        binding.state_rows = [this](const ex::Program& program, std::span<const std::byte> arena, std::uint32_t row,
                                    std::span<st::Row> outputs) {
            if (trace.fail_decode_epilogue && program.key.phase == ex::Phase::decode)
                throw std::runtime_error("scripted decode epilogue failure");
            const auto id = program.launches.front().stages.front().output;
            const auto& buffer = program.buffers[id];
            const auto offset = buffer.arena_offset + row * buffer.layout.row_stride;
            outputs.front() = arena.subspan(static_cast<std::size_t>(offset), sizeof(std::uint32_t));
        };
        strix::profile::Resolution resolution;
        engine = ex::GenericEngine::open_cpu(std::move(decl), std::move(binding), resolution);
    }

    void history(ss::KvHandle lane, std::span<const std::uint32_t> expected) const {
        check(engine->store().consumed(lane) == expected.size(), "logical consumed rows");
        check(trace.state->entry(0, lane.slot).frontier == expected.size(), "physical consumed rows");
        for (std::uint32_t row = 0; row < expected.size(); ++row) {
            const auto bytes = trace.state->committed_row(0, lane.slot, row);
            const auto want = std::as_bytes(expected.subspan(row, 1));
            check(std::ranges::equal(bytes, want), "bitwise script history row");
        }
    }
};

void prefill_decode_stop() {
    Fixture f;
    auto& e = *f.engine;
    auto lane = e.store().acquire();
    const auto caps = e.caps();
    check(caps.max_context == 8 && caps.max_batch == 3 && caps.mixed_advance && caps.batch_prefill &&
          caps.batch_decode && caps.turn_commit && !caps.temperature && !caps.fork_lanes, "admitted capabilities");
    const auto receipt = e.receipt();
    check(receipt.engine == "scripted-conformance" && !receipt.identity.pack_hash.empty() &&
          !receipt.identity.state_format.empty() && receipt.digests.size() == 3, "resolved receipt");
    e.prefill(std::array{2U, 5U}, lane);
    f.history(lane, std::array{5U, 24U});
    ss::SamplerState s;
    s.max_tokens = 8;
    auto first = e.decode(lane, s);
    check(first.tokens == std::vector{27U} && first.before == ss::Frontier{2, {}} &&
          first.after == ss::Frontier{2, 27U} && first.rows.physical_rows == 0 && first.rng_draws == 0 &&
          first.finish == ss::FinishReason::none, "retained prefill pick publishes an unconsumed anchor");

    const std::array stops{3U, 9U, 12U, 16U, 43U};
    s.stop_tokens = stops;
    s.max_tokens = 1;
    auto stop = e.decode(lane, s);
    check(stop.tokens == std::vector{43U} && stop.finish == ss::FinishReason::stop &&
          stop.before == first.after && stop.after == ss::Frontier{3, 43U} && stop.rows.physical_rows == 1,
          "first stop token wins over output allowance and stays pending");
    rejects([&] { (void)e.decode(lane, s); }, "post-stop decode must require resume");
    rejects([&] { e.prefill(std::array{1U}, lane); }, "prefill cannot bypass stop");
    e.commit_pending(lane);
    const auto launches = f.trace.widths.size();
    e.commit_pending(lane);
    check(launches == f.trace.widths.size() && !e.store().pending(lane), "sealing consumes stop exactly once");
    f.history(lane, std::array{5U, 24U, 40U, 41U});
    e.resume_lane(lane);
    e.append_external(std::array{4U, 1U}, lane);
    f.history(lane, std::array{5U, 24U, 40U, 41U, 6U, 27U});
    s.stop_tokens = {};
    s.max_tokens = 0;
    auto zero = e.decode(lane, s);
    check(zero.tokens.empty() && zero.before == zero.after && zero.finish == ss::FinishReason::length &&
          zero.rows.physical_rows == 0, "zero budget preserves selection boundary");
    e.resume_lane(lane);
    s.max_tokens = 8;
    check(e.decode(lane, s).tokens == std::vector{30U}, "resume uses retained suffix selection");
    auto last = e.decode(lane, s);
    check(last.finish == ss::FinishReason::context && last.after == ss::Frontier{7, 59U},
          "last available context position is pending");
    e.resume_lane(lane);
    auto full = e.decode(lane, s);
    check(full.finish == ss::FinishReason::context && full.tokens.empty() && full.before == full.after,
          "full context never evaluates a post-limit row");
}

void mixed_order_and_padding() {
    Fixture f;
    auto& e = *f.engine;
    auto a = e.store().acquire(), b = e.store().acquire(), c = e.store().acquire();
    const std::array one{2U}, two{7U}, three{9U};
    const ss::PrefillItem prefills[] = {{one, a}, {two, b}, {three, c}};
    e.prefill_batch(prefills);
    ss::SamplerState s;
    s.max_tokens = 8;
    const ss::DecodeItem reversed[] = {{c, s}, {a, s}, {b, s}};
    auto first = e.decode_batch(reversed);
    check(first[0].tokens == std::vector{15U} && first[1].tokens == std::vector{8U} &&
          first[2].tokens == std::vector{13U} && f.trace.widths == std::vector{3U}, "retained picks follow caller order");
    auto batch = e.decode_batch(reversed);
    check(batch[0].tokens == std::vector{58U} && batch[1].tokens == std::vector{30U} &&
          batch[2].tokens == std::vector{50U}, "grouped decode follows caller order");
    check(batch[0].rows.physical_rows == 2 && batch[0].rows.padding_rows == 1 &&
          batch[1].rows.physical_rows == 1 && batch[2].rows.physical_rows == 1, "padded row charged once");

    const std::array suffix{1U, 4U};
    const ss::AdvanceItem items[] = {{b, {}, s, {}}, {a, suffix, s, {}}, {c, {}, s, {}}};
    auto out = e.advance(items);
    check(out[0].output.tokens == std::vector{7U} && out[1].input_tokens == 2 &&
          out[1].output.tokens.empty() && out[2].output.tokens == std::vector{39U}, "mixed round result order");
    check(out[1].output.before == ss::Frontier{2, 30U} && out[1].output.after == ss::Frontier{5, {}} &&
          out[1].rows.physical_rows == 3 && !out[1].output.rows.physical_rows, "suffix consumes pending once");
    check(e.store().tokens(a) == std::vector({2U, 8U, 30U, 1U, 4U}), "mixed committed input ordering");
    f.history(a, std::array{5U, 27U, 52U, 35U, 52U});
    f.history(b, std::array{10U, 47U, 4U});
    f.history(c, std::array{12U, 55U, 36U});
    check(f.trace.widths == std::vector({3U, 4U, 3U, 2U}) && f.trace.compiled.size() == 3,
          "prefill width reused its compiled program across round shapes");
    for (const auto& [key, count] : f.trace.compiled) {
        check(key.tier == strix::profile::Name::standard && count == 1, "one compilation per step key");
    }
}

void full_descriptor_prefill() {
    Fixture f(ex::kStepMaxRows);
    auto& e = *f.engine;
    auto full = e.store().acquire(), other = e.store().acquire();
    const std::array<std::uint32_t, ex::kStepMaxRows> prompt{};
    const std::array short_prompt{7U};
    const ss::PrefillItem group[] = {{prompt, full}, {short_prompt, other}};
    e.prefill_batch(group);
    check(f.trace.widths == std::vector({32U, 1U}), "a full descriptor leaves the next lane in a separate step");
    f.history(full, std::array{3U, 13U, 44U, 10U, 37U, 55U, 46U, 20U, 7U, 33U, 48U, 30U, 41U, 11U, 50U, 40U,
                              11U, 53U, 52U, 50U, 45U, 31U, 54U, 60U, 15U, 9U, 56U, 6U, 49U, 51U, 58U, 16U});
    f.history(other, std::array{10U});
}

void cancel_and_rollback() {
    Fixture f;
    auto& e = *f.engine;
    auto a = e.store().acquire(), b = e.store().acquire();
    e.prefill(std::array{2U}, a);
    e.prefill(std::array{7U}, b);
    ss::SamplerState s;
    s.max_tokens = 8;
    check(e.decode(a, s).tokens == std::vector{8U} && e.decode(b, s).tokens == std::vector{13U}, "two pending anchors");
    const std::array suffix{4U};
    const ss::AdvanceItem mixed[] = {{a, suffix, s, {}}, {b, {}, s, {}}};
    f.trace.fail_decode_epilogue = true;
    rejects<std::runtime_error>([&] { (void)e.advance(mixed); }, "late epilogue fault rolls back the entire round");
    check(e.store().pending(a) == 8U && e.store().pending(b) == 13U &&
          f.trace.state->entry(0, a.slot).pending == 0 && f.trace.state->entry(0, b.slot).pending == 0,
          "fault preserves both anchors and aborts both state windows");
    f.history(a, std::array{5U});
    f.history(b, std::array{10U});
    f.trace.fail_decode_epilogue = false;

    unsigned probes = 0;
    const ss::AdvanceItem cancelled[] = {{a, suffix, s, [&] { return ++probes == 2; }}, {b, {}, s, {}}};
    auto out = e.advance(cancelled);
    check(out[0].output.finish == ss::FinishReason::cancelled && out[0].input_tokens == 0 &&
          out[0].output.before == out[0].output.after && out[0].rows.physical_rows == 0 &&
          out[1].output.tokens == std::vector{50U}, "boundary cancellation preserves peer progress");
    rejects([&] { e.append_external(suffix, a); }, "cancelled lane requires explicit resume");
    e.resume_lane(a);
    std::stop_source stop;
    stop.request_stop();
    auto halted = s;
    halted.cancellation = stop.get_token();
    rejects([&] { (void)e.decode(a, halted); }, "resumed lane refuses stopped source");
    check(e.decode(a, s).tokens == std::vector{30U}, "resume reuses the unchanged pending anchor");
    const ss::DecodeItem batch[] = {{a, halted}, {b, s}};
    auto tokens = e.decode_batch(batch);
    check(tokens[0].finish == ss::FinishReason::cancelled && tokens[0].before == tokens[0].after &&
          tokens[1].tokens == std::vector{7U}, "entry cancellation preserves peer decode");

    e.cancel_lane(b);
    rejects([&] { e.commit_pending(b); }, "cancelled turn cannot seal");
    e.resume_lane(b);
    const auto calls = f.trace.widths.size();
    const std::array invalid{vocabulary};
    const ss::AdvanceItem bad[] = {{b, {}, s, {}}, {a, invalid, s, {}}};
    e.resume_lane(a);
    rejects([&] { (void)e.advance(bad); }, "late invalid token prevalidates the group");
    const ss::DecodeItem dup[] = {{b, s}, {b, s}};
    rejects([&] { (void)e.decode_batch(dup); }, "duplicate handle prevalidates the group");
    auto sampled = s;
    sampled.parameters.temperature = 1;
    rejects([&] { (void)e.decode(b, sampled); }, "unsupported sampling is rejected");
    check(f.trace.widths.size() == calls && e.store().pending(b) == 7U, "prevalidation executes no rows");
    auto stale = a;
    e.store().release(a);
    a = e.store().acquire();
    rejects([&] { e.prefill(suffix, stale); }, "reacquisition rejects old generation");
    f.history(a, {});
}

void owner_transcript() {
    Fixture f;
    sv::GenerationQueue queue;
    sv::SessionHooks hooks;
    hooks.schedule = [](std::span<const sv::SessionView> lanes) {
        sv::SessionWork group{sv::SessionPhase::mixed, {}, {}};
        for (std::size_t i = 0; i < lanes.size(); ++i) group.lanes.push_back(i);
        return std::vector{std::move(group)};
    };
    ss::Policy cache;
    cache.enabled = false;
    sv::SessionRuntime owner(queue, *f.engine, f.engine->store(), cache, std::move(hooks));
    sv::GenerationJob first, second;
    first.prompt = {2, 5};
    first.stop_tokens = {43};
    first.max_tokens = 4;
    second.prompt = {7};
    second.max_tokens = 4;
    auto a = queue.submit(std::move(first));
    auto b = queue.submit(std::move(second));
    check(a && b && owner.step() && owner.step(), "owner admits and begins both requests");
    b->cancel();
    check(owner.step() && owner.live() == 0, "owner finishes and releases both requests");
    bool done_a = false, done_b = false;
    const auto ids_a = a->wait(std::chrono::milliseconds{0}, done_a, false);
    const auto ids_b = b->wait(std::chrono::milliseconds{0}, done_b, false);
    check(done_a && done_b && a->error().empty() && b->error().empty(), "owner sees two successful terminals");
    const auto stats_a = a->stats(), stats_b = b->stats();
    check(ids_a == std::vector({27U, 43U}) && stats_a.finish == "stop" && stats_a.prompt_tokens == 2 &&
          stats_a.generated_tokens == 2, "owner publishes first stop and counts the prompt");
    check(ids_b == std::vector{13U} && stats_b.finish == "cancelled" && stats_b.prompt_tokens == 1 &&
          stats_b.generated_tokens == 1, "owner preserves a cancelled request's previously published token");
}

}  // namespace

int main() {
    try {
        prefill_decode_stop();
        std::cout << "PASS generic prefill/decode/stop and bitwise state\n";
        mixed_order_and_padding();
        full_descriptor_prefill();
        std::cout << "PASS generic mixed request order and grouped row accounting\n";
        cancel_and_rollback();
        std::cout << "PASS generic cancel/resume, prevalidation and rollback\n";
        owner_transcript();
        std::cout << "PASS generic SessionRuntime stop/cancel transcript\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
