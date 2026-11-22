#include "strix/exec/draft_runner.hpp"
#include "strix/graph/model_decl.hpp"

#include "executor_cpu_binding.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>

namespace {

namespace ex = strix::exec;
namespace gr = strix::graph;
namespace sp = strix::spec;
namespace st = strix::state;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Call>
void rejects(Call call, const char* message) {
    try {
        call();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error(message);
}

template <class T>
T load(std::span<const std::byte> bytes) {
    check(bytes.size() == sizeof(T), "wrong scripted image size");
    T value;
    std::memcpy(&value, bytes.data(), sizeof(T));
    return value;
}

constexpr gr::StageDecl script_stages[] = {
    {.id = "script", .op = gr::Op::argmax, .entry = "script", .in = {"token"}, .out = "row"},
};

constexpr std::uint32_t noise = 99;
constexpr std::uint8_t chain_id = 1;
constexpr std::uint8_t block_id = 2;

struct Image {
    std::uint32_t word;
    float confidence;
};

struct Trace {
    std::map<ex::ProgramKey, std::uint32_t> compiles;
    std::map<ex::ProgramKey, std::uint32_t> runs;
    std::optional<ex::Phase> fail_phase;
    std::uint32_t fail_pos = 0;
};

struct ScriptArgs {
    Trace* trace;
    ex::BufferId output;
    ex::ProgramKey key;
};

void script(std::span<const std::byte> raw, const ex::Grid&, const ex::CpuLaunchContext& ctx) {
    const auto args = load<ScriptArgs>(raw);
    ++args.trace->runs[args.key];
    const auto anchor = ctx.step.row[0].token;
    for (std::uint32_t i = 0; i < ctx.step.h.rows; ++i) {
        if (!ex::step_row_valid(ctx.step, i)) continue;
        const auto& row = ctx.step.row[i];
        if (args.trace->fail_phase == args.key.phase && args.trace->fail_pos == row.position)
            throw std::runtime_error("injected launch failure");

        auto token = row.token + 1U;
        Image image{row.token + 101U * row.position, 0.0F};
        if (args.key.graph_id == 0 && row.position == 2) token = 70;
        if (args.key.graph_id == block_id && args.key.phase == ex::Phase::draft) {
            check(ctx.step.h.rows == 4 && ctx.step.row[3].token == noise &&
                  row.ancestors == (0xfU ^ (1U << i)), "block noise and noncausal visibility");
            token = anchor + i;
            image.confidence = 0.5F + static_cast<float>(i) * 0.125F;
        }
        std::memcpy(ctx.arena_row(args.output, i).data(), &image, sizeof(image));
        if ((row.flags & ex::step_sample) != 0) ctx.picks[i] = {token, 0.0F, 0.0F, ex::RowStatus::picked};
    }
}

ex::Compiler compiler(const gr::StageGraph& graph, Trace& trace) {
    return ex::Compiler(graph,
        [](const ex::ProgramKey&, const gr::Instance&) { return ex::BufferLayout{ex::BufferStorage::arena, sizeof(Image), 0}; },
        [&trace](const ex::LaunchBindingRequest& request) {
            ++trace.compiles[request.key];
            const ScriptArgs args{&trace, request.launch.stages.front().output, request.key};
            const auto raw = std::as_bytes(std::span(&args, 1));
            ex::KernelBinding binding;
            binding.identity = binding.exact_identity = "draft-test-script/v1";
            binding.args.assign(raw.begin(), raw.end());
            return ex::bind_cpu_reference(std::move(binding), script);
        }, {}, [](const gr::StageGraph& decl, const ex::ProgramKey& key) {
            return gr::launch_plan(decl, 0, key.rows, {});
        });
}

st::Row row_image(const ex::Program& program, std::span<const std::byte> arena, std::uint32_t row) {
    const auto& buffer = program.buffers[program.launches.front().stages.front().output];
    const auto offset = buffer.arena_offset + row * buffer.layout.row_stride;
    return arena.subspan(static_cast<std::size_t>(offset), sizeof(Image));
}

st::CpuState draft_state() {
    constexpr st::Decl decls[] = {
        {.id = "rows", .kind = st::Kind::draft, .dtype = st::Dtype::u32, .dims = {1}},
        {.id = "carry", .kind = st::Kind::draft, .dtype = st::Dtype::u32, .dims = {1},
         .rollback = st::Rollback::replay, .replay_input_bytes = 4},
    };
    const st::Update updates[] = {{}, [](std::uint32_t pos, st::Row previous, st::Row input, std::span<std::byte> next) {
        const auto word = (std::rotl(load<std::uint32_t>(previous), 7) ^ load<std::uint32_t>(input)) + pos;
        std::memcpy(next.data(), &word, sizeof(word));
    }};
    return st::CpuState(st::layout(decls, gr::ModelDecl{}, 32, 1, {.max_pending = 8}), updates);
}

struct Fixture {
    alignas(256) std::array<std::byte, 4096> workspace{};
    ex::CpuExecutor executor{workspace};
    gr::StageGraph target_graph{.before = script_stages};
    gr::StageGraph chain_graph{.before = script_stages};
    gr::StageGraph block_graph{.before = script_stages};
    Trace trace;
    ex::Compiler target = compiler(target_graph, trace);
    ex::Compiler chain = compiler(chain_graph, trace);
    ex::Compiler block = compiler(block_graph, trace);
    st::CpuState chain_state = draft_state();
    st::CpuState block_state = draft_state();

    ex::DraftRunner::CpuBinding binding(ex::Compiler& compiled, std::uint8_t graph_id, st::CpuState& state) {
        return {compiled, executor, workspace, {.graph_id = graph_id}, &state, 0,
                [](const ex::Program& program, st::Row arena, std::uint32_t row, std::span<st::Row> out) {
                    out[0] = out[1] = row_image(program, arena, row).first(sizeof(std::uint32_t));
                }};
    }

    std::vector<std::uint32_t> verify(std::uint32_t anchor, std::uint32_t pos,
                                      std::span<const std::uint32_t> candidates) {
        std::vector<ex::StepRow> rows;
        auto token = anchor;
        for (std::uint32_t i = 0; i <= candidates.size(); ++i) {
            rows.push_back({token, pos + i, 0, i ? i - 1U : ex::kStepNoParent, (1U << i) - 1U,
                            ex::step_valid | ex::step_sample, 0, 0});
            if (i < candidates.size()) token = candidates[i];
        }
        const auto width = std::bit_ceil(static_cast<std::uint32_t>(rows.size()));
        const auto& program = target.get({.phase = ex::Phase::verify, .rows = width});
        std::vector<ex::RowOut> picks(width);
        executor.wait(executor.submit(program, {rows}), picks);
        std::vector<std::uint32_t> emitted;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            check(picks[i].status == ex::RowStatus::picked, "target omitted a pick");
            emitted.push_back(picks[i].id);
            if (i == candidates.size() || picks[i].id != candidates[i]) break;
        }
        return emitted;
    }
};

void state_equals(const st::CpuState& state, std::span<const std::uint32_t> consumed, std::uint32_t pending = 0) {
    std::uint32_t carry = 0;
    for (std::uint32_t pos = 0; pos < consumed.size(); ++pos) {
        const auto word = consumed[pos] + 101U * pos;
        const auto expected = std::as_bytes(std::span(&word, 1));
        check(std::ranges::equal(state.committed_row(0, 0, pos), expected), "committed draft row bytes");
        carry = (std::rotl(carry, 7) ^ word) + pos;
    }
    const auto count = static_cast<std::uint32_t>(consumed.size());
    check(state.committed_row(0, 0, count).empty(), "unverified draft row is visible");
    check(std::ranges::equal(state.current(1, 0), std::as_bytes(std::span(&carry, 1))), "committed draft carry bytes");
    for (std::uint32_t i = 0; i < 2; ++i) {
        const auto& entry = state.entry(i, 0);
        check(entry.frontier == count && entry.pending == pending, "draft state frontier or pending count");
    }
}

std::vector<std::uint32_t> reference_chain(std::uint32_t next, std::uint32_t k) {
    std::vector<std::uint32_t> expected;
    for (std::uint32_t i = 0; i < k; ++i) expected.push_back(++next);
    return expected;
}

void chain_test(Fixture& f) {
    ex::DraftRunner runner({.id = "chain", .graph = "draft.chain", .max_k = 4}, f.binding(f.chain, chain_id, f.chain_state));
    runner.start(10, 0);
    std::vector<std::uint32_t> proposals;
    runner.draft({}, 10, proposals);
    check(proposals == reference_chain(10, 4), "chain reference or max_k");
    state_equals(f.chain_state, {}, 4);
    auto emitted = f.verify(10, 0, proposals);
    check(emitted == std::vector<std::uint32_t>{11, 12, 70}, "target rejection fixture");
    runner.accept(emitted);
    state_equals(f.chain_state, std::array{10U, 11U, 12U});

    runner.draft({}, 3, proposals);
    check(proposals == reference_chain(70, 3), "chain resumed with wrong anchor");
    emitted = f.verify(70, 3, proposals);
    check(emitted == std::vector<std::uint32_t>{71, 72, 73, 74}, "target full acceptance fixture");
    f.trace.fail_phase = ex::Phase::state_only;
    f.trace.fail_pos = 4;
    rejects([&] { runner.accept(emitted); }, "catch-up failure was swallowed");
    state_equals(f.chain_state, std::array{10U, 11U, 12U});
    f.trace.fail_phase.reset();
    runner.accept(emitted);
    const std::array committed{10U, 11U, 12U, 70U, 71U, 72U, 73U};
    state_equals(f.chain_state, committed);

    runner.draft({}, 2, proposals);
    runner.abort();
    state_equals(f.chain_state, committed);
    f.trace.fail_phase = ex::Phase::draft;
    f.trace.fail_pos = 8;
    rejects([&] { runner.draft({}, 2, proposals); }, "draft launch failure was swallowed");
    check(proposals.empty(), "failed draft leaked proposals");
    state_equals(f.chain_state, committed);
    f.trace.fail_phase.reset();

    const std::array prefix{55U, 56U};
    runner.draft(prefix, 2, proposals);
    check(proposals == reference_chain(prefix.back(), 2), "chain failed to consume a known prefix");
    const std::array window{55U, 56U, proposals.front(), proposals.back()};
    emitted = f.verify(74, 7, window);
    check(emitted == std::vector<std::uint32_t>{75}, "target hybrid prefix rejection fixture");
    runner.accept(emitted);
    state_equals(f.chain_state, std::array{10U, 11U, 12U, 70U, 71U, 72U, 73U, 74U});

    runner.draft(prefix, 2, proposals);
    emitted = f.verify(75, 8, std::array{76U, 77U});
    check(emitted == std::vector<std::uint32_t>{76, 77, 78}, "target alternate window fixture");
    runner.accept(emitted);
    state_equals(f.chain_state, std::array{10U, 11U, 12U, 70U, 71U, 72U, 73U, 74U, 75U, 76U, 77U});
}

void block_test(Fixture& f) {
    auto binding = f.binding(f.block, block_id, f.block_state);
    binding.confidence = [](const ex::Program& program, st::Row arena, std::uint32_t row) {
        return load<Image>(row_image(program, arena, row)).confidence;
    };
    ex::DraftRunner runner({.id = "block", .kind = sp::DraftKind::block, .graph = "draft.block", .max_k = 2,
                            .block = 4, .noise_token = noise, .confidence = "score", .prefix_in_block = true},
                           std::move(binding));
    std::vector<std::uint32_t> proposals;
    std::vector<float> confidence;
    const std::array first{20U, noise, noise, noise};
    sp::BlockForward& forward = runner;
    forward.forward({first, 0, 0, 2}, proposals, confidence);
    check(proposals == reference_chain(20, 2), "block reference proposals");
    check(confidence == std::vector<float>{0.625F, 0.75F}, "block confidence follows proposal rows");
    state_equals(f.block_state, {});
    const auto emitted = f.verify(20, 0, proposals);
    check(emitted == std::vector<std::uint32_t>{21, 22, 70}, "block verify fixture");
    forward.accept(emitted);
    state_equals(f.block_state, std::array{20U, 21U, 22U});

    runner.draft(std::array{71U}, 10, proposals);
    check(proposals == reference_chain(71, 2), "block prefix and max_k");
    runner.accept(std::array{71U, 72U});
    state_equals(f.block_state, std::array{20U, 21U, 22U, 70U, 71U});
}

void lookup_test(Fixture& f) {
    constexpr std::array sources{sp::LookupSource::context, sp::LookupSource::ngram};
    ex::DraftRunner runner({.id = "copy", .kind = sp::DraftKind::lookup, .max_k = 2, .sources = sources},
                           sp::ProposerConfig{{128, 8}, 2});
    runner.accept(std::array{4U, 5U, 6U, 7U, 9U, 4U, 5U});
    const auto before = f.trace.runs;
    std::vector<std::uint32_t> proposals;
    runner.draft(std::array{6U}, 1, proposals);
    check(proposals == std::vector<std::uint32_t>{7}, "lookup continuation after a hypothetical prefix");
    runner.draft({}, 3, proposals);
    check(proposals == std::vector<std::uint32_t>{6, 7}, "lookup reference continuation or max_k");
    check(f.trace.runs == before, "weightless lookup submitted a program");
    const auto emitted = f.verify(5, 6, proposals);
    check(emitted == std::vector<std::uint32_t>{6, 7, 8}, "lookup verification fixture");
    runner.accept(emitted);
    runner.draft({}, 2, proposals);
    check(proposals.empty(), "lookup did not advance to the verified suffix");
}

void check_cache(const Trace& trace) {
    const ex::ProgramKey chain{.phase = ex::Phase::draft, .graph_id = chain_id};
    const ex::ProgramKey block{.phase = ex::Phase::draft, .rows = 4, .graph_id = block_id};
    const ex::ProgramKey chain_catchup{.phase = ex::Phase::state_only, .graph_id = chain_id};
    const ex::ProgramKey block_catchup{.phase = ex::Phase::state_only, .graph_id = block_id};
    std::size_t draft_keys = 0;
    for (const auto& [key, count] : trace.compiles) {
        check(count == 1, "X1 compiled the same key twice");
        if (key.graph_id) ++draft_keys;
    }
    check(draft_keys == 4, "unexpected graph key split");
    for (const auto& key : {chain, block, chain_catchup, block_catchup})
        check(trace.compiles.at(key) == 1 && trace.runs.at(key) > 1, "draft graph key was not reused");
}

}  // namespace

int main() {
    try {
        Fixture fixture;
        chain_test(fixture);
        block_test(fixture);
        lookup_test(fixture);
        check_cache(fixture.trace);
        std::cout << "P1 proposals: chain, block and lookup match reference\n"
                     "P1 S1 state: accepted rows and replay carry match byte for byte\n"
                     "P1 X1 keys: four draft graph keys compiled once each\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
