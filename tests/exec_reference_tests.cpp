#include "executor_cpu_binding.hpp"

#include "attn_core/causal_gqa_reference.hpp"
#include "attn_core/prepared_gqa.hpp"
#include "core/linear_reference.hpp"
#include "core/residual_reference.hpp"
#include "core/short_convolution.hpp"
#include "cpu/core/gdn_reference.hpp"
#include "ops/glu/glu.hpp"
#include "ops/norm/reference.hpp"
#include "ops/rope/rope.hpp"
#include "strix/graph/model_decl.hpp"
#include "strix/head_argmax.hpp"
#include "strix/moe/route_policy.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string_view>

namespace {

namespace ex = strix::exec;
namespace gr = strix::graph;
namespace core = strix::core;
namespace attn = strix::attn_core;
namespace norm = strix::ops::norm;
namespace rope = strix::ops::rope;
namespace glu = strix::ops::glu;
namespace moe = strix::moe;
using strix::attn::group_indexer::bf16_round;

constexpr unsigned hidden = 4, vocab = 8, experts = 4, top_k = 2, taps = 4;
constexpr unsigned positions = 8, slots = 5, decode_steps = 3;
using Row = std::array<float, 6 * hidden>;
using Inputs = std::array<Row, 4>;
using F = std::span<float>;
using C = std::span<const float>;

void check(bool result, std::string_view message) {
    if (!result) throw std::runtime_error(std::string(message));
}

template<class T>
bool same(std::span<const T> left, std::span<const T> right) {
    return left.size() == right.size() &&
           (left.empty() || std::memcmp(left.data(), right.data(), left.size_bytes()) == 0);
}

struct Math {
    static float exp(float x) { return std::exp(x); }
    static float log1p(float x) { return std::log1p(x); }
    static float sqrt(float x) { return std::sqrt(x); }
    static float tanh(float x) { return std::tanh(x); }
};

struct Rms {
    using Input = float;
    using Output = float;
    using Scale = float;
    using ScaleFormat = int;
    using Stream = void*;
    static constexpr float epsilon = 1e-6F;
    static constexpr auto reduction_order = norm::ReductionOrder::lane_strided_fma_lds_halving;
    static constexpr unsigned lanes(unsigned) { return 128; }
    static float load(const float* gain, int, unsigned i) { return gain[i]; }
    static float inv_sqrt(float x) { return 1.0F / std::sqrt(x); }
};

// Fixture-only adapters use narrow FP32 weights, one attention head, and one
// residual block. Each visit has separate state addressed by X3 session slot.
enum class Body {
    embed, rms, residual_rms, message_rms, linear, qkv, prepare_kv, gqa, prepared_gqa, gdn, convolution,
    residual_read, save_block, add, add_bf16, silu, route, gate_up, down, merge, softcap, pick
};

struct Operation {
    Body body;
    bool inkling;
    bool full;
    int layer;
    unsigned width;
    std::vector<float> weights;
};

struct Lane {
    std::array<float, positions * hidden> keys{}, values{};
    std::array<float, (taps - 1) * hidden> key_history{}, value_history{};
    std::array<float, (taps - 1) * 3 * hidden> gdn_history{};
    std::array<float, hidden * hidden> recurrent{};
};

struct Fixture {
    std::vector<Operation> ops;
    std::vector<std::array<Lane, slots>> state;
    std::array<std::array<float, hidden>, slots> residual_bank{};
    std::vector<ex::StepBlock> descriptors;
    std::vector<Row> trace;
    std::vector<Row> logits;
};

void evaluate(const Operation& op, const ex::StepRow& step, const Inputs& in, Lane& lane,
              std::span<float, hidden> bank, Row& result, ex::RowOut& pick) {
    auto output = F(result);
    const C w(op.weights);
    const C a(in[0]), b(in[1]), c(in[2]);
    constexpr auto epsilon = Rms::epsilon;
    switch (op.body) {
    case Body::embed:
        std::ranges::copy(w.subspan(step.token * hidden, hidden), output.begin());
        break;
    case Body::linear:
        core::linear_reference(w.first(op.width * hidden), a.first(hidden), output.first(op.width));
        if (op.inkling) std::ranges::transform(output.first(op.width), output.begin(), bf16_round);
        break;
    case Body::rms:
    case Body::residual_rms:
    case Body::message_rms: {
        std::array<float, hidden> input{};
        for (unsigned i = 0; i < hidden; ++i) {
            input[i] = a[i];
            if (op.body == Body::residual_rms) input[i] += b[i];
            if (op.body == Body::message_rms && op.layer > 0) input[i] = b[i];
        }
        if (op.inkling) {
            const float scale = strix::attn::group_pool::rms_inv(input.data(), hidden, epsilon);
            for (unsigned i = 0; i < hidden; ++i) output[i] = bf16_round((input[i] * scale) * w[i]);
        } else {
            norm::reference<Rms>(input.data(), w.data(), 0, hidden, epsilon, output.data(), 1);
        }
        break;
    }
    case Body::qkv:
        for (unsigned part = 0; part < 6; ++part) {
            const auto source = op.full && part >= 4 ? C(in[part - 3]) : a;
            auto projected = output.subspan(part * hidden, hidden);
            core::linear_reference(w.subspan(part * hidden * hidden, hidden * hidden), source.first(hidden), projected);
            std::ranges::transform(projected, projected.begin(), bf16_round);
        }
        break;
    case Body::prepare_kv:
        for (unsigned part = 0; part < 2; ++part) {
            auto history = part == 0 ? F(lane.key_history) : F(lane.value_history);
            auto prepared = output.subspan(part * hidden, hidden);
            core::causal_short_convolution({hidden, taps}, a.subspan((part + 4) * hidden, hidden),
                w.subspan(part * taps * hidden, taps * hidden), {}, history, prepared);
            if (part == 0) attn::detail::clip_prepared_gqa_head(prepared, {}, 1.0F, epsilon);
            else {
                const float scale = strix::attn::group_pool::rms_inv(prepared.data(), hidden, epsilon);
                for (auto& value : prepared) value = bf16_round(value * scale);
            }
        }
        break;
    case Body::gqa:
    case Body::prepared_gqa: {
        auto q = std::array{a[0], a[1], a[2], a[3]};
        auto key = std::array{b[0], b[1], b[2], b[3]};
        auto value = std::array{c[0], c[1], c[2], c[3]};
        const bool prepared = op.body == Body::prepared_gqa;
        if (prepared) {
            std::ranges::copy(b.subspan(hidden, hidden), value.begin());
            attn::detail::clip_prepared_gqa_head(q, w.first(hidden), 1.0F, epsilon);
        } else {
            // Present one NeoX pair in the shared reference's adjacent-pair layout.
            float cos = 0.0F, sin = 0.0F;
            rope::build_native_table(1e7F, 1, step.position, 1, rope::Round::none, &cos, &sin);
            const auto rotate = [&](std::array<float, hidden>& row) {
                std::array pair{row[0], row[2]};
                constexpr std::uint32_t table_row = 0;
                rope::reference_apply({pair.data(), 1, 1, 2, 2, 0, 1, &cos, &sin, &table_row}, rope::Round::none);
                row[0] = pair[0];
                row[2] = pair[1];
            };
            rotate(q);
            rotate(key);
            std::ranges::transform(key, key.begin(), bf16_round);
            std::ranges::transform(value, value.begin(), bf16_round);
        }
        attn::causal_gqa_reference(
            {1, 1, hidden, op.full ? 0U : 2U, prepared ? attn::ReferenceAttentionArithmetic::bf16_eager
                                                        : attn::ReferenceAttentionArithmetic::fp32},
            {step.position, q, key, value, prepared ? hidden : 0U,
             prepared ? a.subspan(hidden, hidden) : C{}, prepared ? a.subspan(3 * hidden, 1) : C{}},
            {lane.keys, lane.values}, output.first(hidden));
        for (unsigned i = 0; i < hidden; ++i) {
            const float gate = a[(prepared ? 2U : 1U) * hidden + i];
            output[i] = glu::sigmoid_gate<Math, glu::Round::none>(output[i], gate);
            if (prepared) output[i] = bf16_round(output[i]);
        }
        break;
    }
    case Body::gdn: {
        std::array<float, 3 * hidden> raw{}, projected{};
        std::ranges::transform(a.first(raw.size()), raw.begin(), bf16_round);
        core::causal_short_convolution({raw.size(), taps}, raw, w.subspan(hidden, taps * raw.size()), {},
            lane.gdn_history, projected);
        for (float& value : projected)
            value = bf16_round(glu::swiglu<Math, glu::Round::none>(value, 1.0F, glu::no_clamp));
        const auto unit = [](C row, float scale) {
            const float sum = core::dot_f32(row, row);
            std::array<float, hidden> normalized{};
            const float inv = scale / std::sqrt(sum + epsilon);
            for (unsigned i = 0; i < hidden; ++i) normalized[i] = row[i] * inv;
            return normalized;
        };
        const C conv(projected);
        const auto q = unit(conv.first(hidden), 0.5F), k = unit(conv.subspan(hidden, hidden), 1.0F);
        strix::cpu::gdn_reference(q, k, conv.subspan(2 * hidden, hidden), -std::log1p(std::exp(in[3][0])),
            moe::sigmoid_score<Math>(c[0]), lane.recurrent, output.first(hidden));
        std::array<float, hidden> normalized{};
        norm::reference<Rms>(output.data(), w.data(), 0, hidden, epsilon, normalized.data(), 1);
        for (unsigned i = 0; i < hidden; ++i)
            output[i] = glu::swiglu<Math, glu::Round::none>(b[i], normalized[i], glu::no_clamp);
        break;
    }
    case Body::convolution:
        core::causal_short_convolution({hidden, taps, 0, 1.0F, std::sqrt(5.0F)}, a.first(hidden),
            w.first(taps * hidden), {}, lane.key_history, output.first(hidden));
        std::ranges::transform(output.first(hidden), output.begin(), bf16_round);
        break;
    case Body::residual_read: {
        std::array<float, 2 * hidden> states{};
        std::ranges::copy(bank, states.begin());
        std::ranges::copy(a.first(hidden), states.begin() + hidden);
        namespace residual = core::residual_reference;
        const auto mixed = residual::mix(residual::normalize_bank(states, hidden, epsilon), w.first(hidden), hidden);
        std::ranges::copy(mixed, output.begin());
        break;
    }
    case Body::save_block:
        std::ranges::copy(a.first(hidden), bank.begin());
        std::ranges::copy(b.first(hidden), output.begin());
        break;
    case Body::add:
    case Body::add_bf16:
        for (unsigned i = 0; i < hidden; ++i) {
            output[i] = (a[i] + b[i]) + c[i];
            if (op.body == Body::add_bf16) output[i] = bf16_round(output[i]);
        }
        break;
    case Body::silu:
        for (unsigned i = 0; i < hidden; ++i) output[i] = glu::swiglu<Math, glu::Round::none>(a[i], b[i], glu::no_clamp);
        break;
    case Body::route: {
        constexpr moe::RoutePolicy policy{.selection_bias = true, .order = moe::RouteOrder::ascending_id,
                                         .experts = experts, .top_k = top_k, .scale = 8.0F};
        std::array<float, experts> scores{};
        std::array<std::uint32_t, top_k> ids{};
        moe::route_scores_ref<Math>(policy.score, a.first(experts), scores);
        moe::route_select_ref(policy, scores, w.first(experts), ids, output.subspan(top_k, top_k));
        for (unsigned i = 0; i < top_k; ++i) output[i] = static_cast<float>(ids[i]);
        break;
    }
    case Body::gate_up:
    case Body::down:
        for (unsigned i = 0; i < top_k; ++i) {
            const auto expert = static_cast<unsigned>(b[i]);
            auto active = output.subspan(i * hidden, hidden);
            if (op.body == Body::gate_up) {
                std::array<float, 2 * hidden> gate_up{};
                core::linear_reference(w.subspan(expert * gate_up.size() * hidden, gate_up.size() * hidden),
                    a.first(hidden), gate_up);
                constexpr moe::SoftcappedSwiglu<Math> activate{4.0F, 25.0F};
                for (unsigned j = 0; j < hidden; ++j)
                    active[j] = bf16_round(activate(bf16_round(gate_up[j]), bf16_round(gate_up[hidden + j])));
            } else {
                core::linear_reference(w.subspan(expert * hidden * hidden, hidden * hidden), a.subspan(i * hidden, hidden), active);
                std::ranges::transform(active, active.begin(), bf16_round);
            }
        }
        break;
    case Body::merge:
        for (unsigned i = 0; i < top_k; ++i)
            for (unsigned j = 0; j < hidden; ++j) output[j] += a[i * hidden + j] * b[top_k + i];
        std::ranges::transform(output.first(hidden), output.begin(), bf16_round);
        break;
    case Body::softcap:
        for (unsigned i = 0; i < vocab; ++i) output[i] = 30.0F * std::tanh(a[i] / 30.0F);
        break;
    case Body::pick: {
        const auto best = strix::head_argmax::reference2(a.first(vocab));
        check(best.index2 != strix::head_argmax::kNone, "missing finite top two logits");
        pick = {best.first.index, best.first.value, best.value2, ex::RowStatus::picked};
        result = in[0];
        break;
    }
    }
}

Body reference_body(const gr::StageDecl& stage, bool inkling) {
    using gr::Op;
    if (stage.id == "qkv" && inkling) return Body::qkv;
    if (stage.id == "query_input") return Body::message_rms;
    if (stage.id == "post_norm") return Body::residual_rms;
    if (stage.id == "save_block") return Body::save_block;
    if (stage.id == "expert_gate_up") return Body::gate_up;
    if (stage.id == "expert_down") return Body::down;
    if (stage.form == "conv4.clip.kv_append") return Body::prepare_kv;
    if (stage.form == "conv4") return Body::convolution;
    if (stage.form == "weighted_sum") return Body::merge;
    if (stage.form == "add_round_bf16") return Body::add_bf16;
    if (stage.form == "residual_bank") return Body::residual_read;
    if (stage.form == "softcap") return Body::softcap;
    switch (stage.op) {
    case Op::gather: return Body::embed;
    case Op::linear:
    case Op::head: return Body::linear;
    case Op::norm: return Body::rms;
    case Op::attend: return inkling ? Body::prepared_gqa : Body::gqa;
    case Op::mix: if (stage.entry == "gdn_step") return Body::gdn; break;
    case Op::combine: if (stage.form == "add") return Body::add; break;
    case Op::elementwise: if (stage.form == "silu_mul") return Body::silu; break;
    case Op::route: return Body::route;
    case Op::argmax: return Body::pick;
    default: break;
    }
    throw std::runtime_error("unbound tiny reference stage " + std::string(stage.id));
}

Operation operation(const gr::Instance& instance, bool inkling, bool full, std::uint32_t& seed) {
    const auto& stage = *instance.decl;
    Operation op{reference_body(stage, inkling), inkling, full, instance.layer, hidden, {}};
    if (stage.op == gr::Op::head) op.width = vocab;
    if (!inkling && stage.id == "qkv") op.width = 3 * hidden;
    if (!inkling && stage.id == "q") op.width = 2 * hidden;
    if (stage.id == "alpha" || stage.id == "beta") op.width = 1;
    op.weights.resize(experts * 2 * hidden * hidden);
    for (float& weight : op.weights) {
        seed = seed * 1664525U + 1013904223U;
        weight = static_cast<float>(static_cast<int>(seed >> 24U) - 128) * (1.0F / 512.0F);
    }
    if (stage.op == gr::Op::norm || op.body == Body::gdn)
        for (auto& weight : F(op.weights).first(hidden)) weight += 1.0F;
    return op;
}

struct TinyGraph {
    std::vector<gr::StageDecl> before, layer, after;
    gr::StageGraph graph;
};

void project(TinyGraph& out, const gr::StageGraph& source, bool inkling) {
    // Use the first declared cycle so both attention classes execute. The D1
    // legacy graph does not represent the repeat-visit message bank or multiports.
    out.graph.layers.assign(source.layers.begin(), source.layers.begin() + (inkling ? 6 : 4));
    const auto active = [&](const gr::StageDecl& stage) {
        for (std::size_t l = 0; l < out.graph.layers.size(); ++l)
            if (gr::active(source, stage, static_cast<std::int32_t>(l), 0)) return true;
        return false;
    };
    std::ranges::copy(source.before, std::back_inserter(out.before));
    std::ranges::copy_if(source.layer, std::back_inserter(out.layer), active);
    std::ranges::copy(source.after, std::back_inserter(out.after));
    if (inkling) {
        // Bundle the six QKV ports into one fixture row and unpack in its readers.
        for (auto& stage : out.layer) {
            if (stage.id == "prepare_kv") { stage.in[0] = "q"; stage.in[1] = {}; }
            if (stage.id == "attend") { stage.in[1] = "kv_rows"; stage.in[2] = {}; stage.in[3] = {}; }
        }
    } else {
        out.before.push_back({.id = "fixture_embed", .op = gr::Op::gather, .out = "x"});
        out.layer.push_back({.id = "fixture_residual", .op = gr::Op::combine, .form = "add",
                             .in = {"x", "proj", "proj2"}, .out = "x"});
        out.after.push_back({.id = "fixture_head", .op = gr::Op::head, .in = {"x"}, .out = "logits",
                             .operand = gr::weights("fixture_head.weight", vocab, hidden)});
    }
    out.after.push_back({.id = "fixture_pick", .op = gr::Op::argmax, .in = {inkling ? "scored" : "logits"}, .out = "pick"});
    static constexpr gr::FormatRule weights[] = {{"", gr::Format::f32, {}}};
    out.graph.before = out.before;
    out.graph.layer = out.layer;
    out.graph.after = out.after;
    out.graph.formats = weights;
}

struct Args {
    Fixture* fixture;
    std::uint32_t stage;
    std::array<ex::BufferId, 4> inputs;
    ex::BufferId output;
};

void call_reference(std::span<const std::byte> bytes, const ex::Grid& grid, const ex::CpuLaunchContext& context) {
    Args args{};
    check(bytes.size() == sizeof(args) && grid.blocks[0] == context.step.h.rows, "CPU bound arguments or grid");
    std::memcpy(&args, bytes.data(), sizeof(args));
    auto& fixture = *args.fixture;
    if (args.stage == 0) fixture.descriptors.push_back(context.step);
    const auto& op = fixture.ops.at(args.stage);
    for (std::uint32_t i = 0; i < context.step.h.rows; ++i) {
        if (!ex::step_row_valid(context.step, i)) continue;
        const auto& step = ex::step_row(context.step, i);
        Inputs inputs{};
        for (std::size_t port = 0; port < args.inputs.size(); ++port) {
            if (args.inputs[port] == ex::no_buffer) continue;
            const auto cell = context.arena_row(args.inputs[port], i);
            check(cell.size() == sizeof(Row), "CPU arena input row size");
            std::memcpy(inputs[port].data(), cell.data(), cell.size());
        }
        Row result{};
        evaluate(op, step, inputs, fixture.state[args.stage].at(step.slot), fixture.residual_bank.at(step.slot),
                 result, context.picks[i]);
        const auto cell = context.arena_row(args.output, i);
        check(cell.size() == sizeof(Row), "CPU arena output row size");
        std::memcpy(cell.data(), result.data(), cell.size());
        fixture.trace.push_back(result);
        if (op.body == Body::pick) fixture.logits.push_back(result);
    }
}

ex::Program compile(const gr::StageGraph& graph, const gr::LaunchPlan& plan, const ex::ProgramKey& key, Fixture& fixture) {
    const ex::Compiler compiler(graph,
        [](const ex::ProgramKey&, const gr::Instance&) {
            return ex::BufferLayout{ex::BufferStorage::arena, sizeof(Row), sizeof(Row) + 3 * sizeof(float)};
        },
        [&](const ex::LaunchBindingRequest& request) {
            check(request.launch.stages.size() == 1 && request.launch.prologue.empty() && request.launch.epilogue.empty(),
                  "reference fixture requires one material stage per launch");
            const auto& stage = request.launch.stages.front();
            auto inputs = stage.inputs;
            for (auto& id : inputs)
                if (id != ex::no_buffer && request.buffers[id].layout.storage == ex::BufferStorage::external) id = ex::no_buffer;
            const Args args{&fixture, stage.instance, inputs, stage.output};
            const auto bytes = std::as_bytes(std::span(&args, 1));
            ex::KernelBinding result;
            result.identity = "cpu.synthetic-f32/v1/" + std::string(request.launch.entry);
            result.exact_identity = result.identity;
            result.grid.blocks[0] = request.key.rows;
            result.args.assign(bytes.begin(), bytes.end());
            return ex::bind_cpu_reference(std::move(result), call_reference);
        });
    return compiler.compile(key, plan);
}

ex::StepAdvance direct_step(std::span<const gr::Instance> stages, const ex::StepBlock& descriptor,
                            const ex::StepStopTable& stops, Fixture& direct) {
    std::array<std::map<std::string_view, Row>, ex::kStepMaxRows> values;
    std::array<ex::RowOut, ex::kStepMaxRows> picks{};
    direct.descriptors.push_back(descriptor);
    for (std::size_t s = 0; s < stages.size(); ++s) {
        const auto& stage = *stages[s].decl;
        const auto& op = direct.ops.at(s);
        for (std::uint32_t i = 0; i < descriptor.h.rows; ++i) {
            if (!ex::step_row_valid(descriptor, i)) continue;
            const auto& row = ex::step_row(descriptor, i);
            Inputs inputs{};
            for (std::size_t port = 0; port < inputs.size(); ++port)
                if (const auto it = values[i].find(stage.in[port]); it != values[i].end()) inputs[port] = it->second;
            Row result{};
            evaluate(op, row, inputs, direct.state[s].at(row.slot), direct.residual_bank.at(row.slot), result, picks[i]);
            direct.trace.push_back(result);
            if (op.body == Body::pick) direct.logits.push_back(result);
            values[i][stage.out] = result;
        }
    }
    return ex::advance_step(descriptor, std::span(picks).first(descriptor.h.rows), stops);
}

void compare_chain(const gr::StageGraph& graph, const gr::LaunchPlan& plan, const ex::ProgramKey& key,
                   Fixture interpreted, Fixture direct, std::uint32_t active, bool stop_one_or_more) {
    const auto program = compile(graph, plan, key, interpreted);
    check(program.numerical_identity.has_value() && program.arena.total_bytes < program.arena.naive_nonaliased_bytes,
          "expected a bitwise reference program with arena reuse");
    std::vector<std::byte> arena(static_cast<std::size_t>(program.arena.total_bytes), std::byte{0xa5});
    ex::CpuExecutor executor(arena);
    constexpr std::uint32_t slot_order[] = {4, 2, 0, 3};
    ex::StepBlock expected{};
    expected.h = {.rows = key.rows, .valid_mask = (1U << active) - 1U, .step = 11,
                  .feed_mask = (1U << active) - 1U, .sampling_policy = 0,
                  .stop_count = 0, .stop_table_lo = 0, .stop_table_hi = 0};
    for (std::uint32_t i = 0; i < active; ++i)
        expected.row[i] = {i + 1, i, slot_order[i], ex::kStepNoParent, 0, ex::step_valid, i + 5, i + 7};
    ex::StepStopTable stops{};
    if (stop_one_or_more) {
        auto scratch = direct;
        stops.ids[0] = direct_step(plan.instances, expected, stops, scratch).readback[0].id;
        expected.h.stop_count = 1;
        ex::set_step_stop_address(expected.h, reinterpret_cast<std::uintptr_t>(&stops));
    }
    const ex::CpuStepIn input{std::span(expected.row).first(active), expected.h.feed_mask, expected.h.step, 0,
                              std::span(stops.ids).first(expected.h.stop_count)};
    const auto ticket = executor.submit_chain(program, input, decode_steps);
    std::vector<ex::RowOut> actual(key.rows * decode_steps);
    check(executor.poll(ticket, actual), "CPU tickets are complete on submission");
    ex::StepAdvance advanced{};
    for (std::uint32_t step = 0; step < decode_steps; ++step) {
        advanced = direct_step(plan.instances, expected, stops, direct);
        check(same<ex::RowOut>(std::span(actual).subspan(step * key.rows, key.rows),
                               std::span(advanced.readback).first(key.rows)), "readback bits differ");
        expected = advanced.next;
    }
    if (stop_one_or_more) {
        const auto* bound_stops = reinterpret_cast<const ex::StepStopTable*>(
            static_cast<std::uintptr_t>(ex::step_stop_address(executor.descriptor()->h)));
        check(bound_stops && bound_stops->ids[0] == stops.ids[0], "descriptor stop table binding");
        check(actual[0].status == ex::RowStatus::stop && actual[key.rows].status == ex::RowStatus::masked,
              "stop must retire its row for later steps");
        for (auto& descriptor : direct.descriptors)
            ex::set_step_stop_address(descriptor.h, ex::step_stop_address(executor.descriptor()->h));
        ex::set_step_stop_address(expected.h, ex::step_stop_address(executor.descriptor()->h));
    } else {
        check(interpreted.logits.size() == active * decode_steps && expected.h.valid_mask == (1U << active) - 1U,
              "expected three scored steps for each active row");
    }
    check(same<ex::StepBlock>(std::span(executor.descriptor(), 1), std::span(&expected, 1)), "X3 descriptor bits differ");
    check(same<ex::StepBlock>(interpreted.descriptors, direct.descriptors), "per-step X3 descriptor bits differ");
    check(executor.commit_mask() == advanced.commit_mask, "X3 commit mask differs");
    check(same<ex::RowOut>(std::span(executor.output(), key.rows), std::span(actual).last(key.rows)), "latest readback differs");
    check(same<Row>(interpreted.trace, direct.trace), "intermediate bits differ");
    check(same<Row>(interpreted.logits, direct.logits), "logit bits differ");
    check(same<std::array<Lane, slots>>(interpreted.state, direct.state), "state bits differ");
    check(same<std::array<float, hidden>>(interpreted.residual_bank, direct.residual_bank), "residual bank bits differ");

    auto foreign = program;
    foreign.launches.back().fn = [](std::span<const std::byte>, const ex::Grid&, void*) {
        throw std::logic_error("foreign thunk executed");
    };
    const auto trace_size = interpreted.trace.size();
    try {
        (void)executor.submit(foreign, input);
        throw std::logic_error("foreign launch was accepted");
    } catch (const std::invalid_argument&) {
        check(interpreted.trace.size() == trace_size, "validation must precede all launches");
    }
    try {
        executor.wait(ticket, actual);
        throw std::logic_error("consumed ticket was accepted");
    } catch (const std::invalid_argument&) {
        check(interpreted.trace.size() == trace_size, "ticket errors must not run any reference");
    }
    if (active == 1) {
        auto single = direct.descriptors.front();
        single.h.feed_mask = 0;
        Fixture direct_single;
        direct_single.ops = direct.ops;
        direct_single.state.resize(direct.ops.size());
        Fixture single_fixture = direct_single;
        const auto one = direct_step(plan.instances, single, stops, direct_single);
        const auto single_program = compile(graph, plan, key, single_fixture);
        const ex::CpuStepIn single_input{std::span(single.row).first(1), 0, single.h.step};
        const auto profile = executor.profile(single_program, single_input);
        check(profile.launch_us.size() == single_program.launches.size() && profile.total_us >= 0.0F,
              "profile must cover the launch array");
        check(same<ex::RowOut>(std::span(executor.output(), 1), std::span(one.readback).first(1)), "profile changed pick bits");
    }
}

void run(const gr::ModelDecl& decl, bool inkling) {
    TinyGraph tiny;
    project(tiny, decl.stages, inkling);
    for (std::uint32_t active = 1; active <= 4; ++active) {
        const ex::ProgramKey key{.rows = std::bit_ceil(active)};
        const auto plan = gr::launch_plan(tiny.graph, 0, key.rows, gr::FusionPolicy{});
        std::vector<bool> full(tiny.graph.layers.size());
        for (const auto& instance : plan.instances)
            if (inkling ? instance.decl->id == "key_input" : instance.decl->op == gr::Op::attend)
                full.at(static_cast<std::size_t>(instance.layer)) = true;
        Fixture fixture;
        std::uint32_t seed = 29;
        for (const auto& instance : plan.instances) {
            const bool full_attention = instance.layer >= 0 && full.at(static_cast<std::size_t>(instance.layer));
            fixture.ops.push_back(operation(instance, inkling, full_attention, seed));
        }
        fixture.state.resize(fixture.ops.size());
        compare_chain(tiny.graph, plan, key, fixture, fixture, active, false);
        if (active == 4) compare_chain(tiny.graph, plan, key, fixture, fixture, active, true);
    }
}

}  // namespace

int main(int argc, char** argv) try {
    check(argc == 4, "expected model, template directory and declaration fixture directory");
    const std::string_view model(argv[1]);
    const bool inkling = model == "inkling2_mini";
    check(inkling || model == "qwen_27b", "unknown proving model");
    const auto path = std::filesystem::path(argv[3]) / (inkling ? "inkling2-mini.toml" : "qwen-27b.toml");
    const auto desc = strix::model_description::load(path.string(), strix::model_description::TableSyntax::root_array_tables);
    run(gr::inspect_model_decl(desc, {argv[2]}), inkling);
    std::cout << "exec_reference_" << model << ": PASS (3 steps, active rows 1..4, bitwise picks/logits/state and X3 advance)\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
