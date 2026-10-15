#include "../apps/hip_decode_receipt_json.hpp"
#include "../apps/shared_prefetch_receipt.hpp"
#include "strix/compiled_execution.hpp"
#include "strix/compiled_speculative_verify.hpp"
#include "strix/shared_expert_prefetch.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
using namespace strix;
namespace {
int failures = 0;
void expect(bool ok, const char *m) {
    if (!ok) {
        ++failures;
        std::cerr << m << '\n';
    }
}
template <class F> void rejects(F f, const char *message) {
    try {
        f();
        expect(false, "expected invalid_argument");
    } catch (const std::invalid_argument &e) {
        expect(std::string(e.what()).find(message) != std::string::npos, e.what());
    }
}
ModelManifest manifest(const ModelContract &m, bool q8 = true) {
    ModelManifest out;
    const std::array names{"shared_expert.gate_proj.weight", "shared_expert.up_proj.weight",
                           "shared_expert.down_proj.weight"};
    for (std::uint32_t layer = 0; layer < m.hidden_layers; ++layer)
        for (unsigned i = 0; i < 3; ++i) {
            ManifestTensor t;
            t.name = "blk." + std::to_string(layer) + "." + names[i];
            t.layer = layer;
            t.role = TensorRole::dense_projection;
            t.dimensions =
                i == 2
                    ? std::vector<std::uint64_t>{m.shared_expert_intermediate_size, m.hidden_size}
                    : std::vector<std::uint64_t>{m.hidden_size, m.shared_expert_intermediate_size};
            t.ggml_type = q8 ? 8U : 0U;
            t.absolute_offset = 0x1000U;
            t.byte_size =
                q8 ? shared_expert_prefetch::required_weight_bytes(t.dimensions[0], t.dimensions[1])
                   : t.dimensions[0] * t.dimensions[1] * 4U;
            out.tensors.push_back(t);
        }
    return out;
}
kernels::KernelRegistry registry(const ModelContract &m, const ExecutionBindingRequest &r,
                                 bool include_prefetch) {
    kernels::KernelRegistry out;
    std::set<std::string> added;
    for (const auto &step : build_serial_decode_execution_plan(m).steps) {
        const auto family = operator_family_for(step.op);
        if (!family)
            continue;
        const auto shape = execution_shape_for(m, step.op, r);
        const auto id = "host." + std::to_string(static_cast<unsigned>(*family)) + "." +
                        std::to_string(shape.hidden_size) + "." +
                        std::to_string(shape.auxiliary_size);
        if (!added.insert(id).second)
            continue;
        kernels::KernelVariant v;
        v.id = id;
        v.family = *family;
        v.shape = shape;
        v.phase = kernels::ExecutionPhase::decode;
        v.resources.workgroup_threads = 256;
        v.numerics = {"bf16", "fp32", "bf16", true, true};
        v.qualification = kernels::QualificationStatus::experimental;
        v.input_layout = "host";
        v.output_layout = "host";
        v.compiler_identity = "host-only";
        out.add(v);
        if (include_prefetch && step.op == SerialOp::shared_expert) {
            v.id += shared_expert_prefetch::kernel_suffix;
            v.compiler_identity = "host-only-prefetch";
            out.add(v);
        }
    }
    return out;
}
} // namespace
int main() {
    const std::string receipt_name = "blk.0.shared_expert.up_proj.\"\\\n\t";
    expect(strix::decode_receipt::escape_json_string(receipt_name) ==
               "blk.0.shared_expert.up_proj.\\\"\\\\\\u000a\\u0009",
           "receipt preserves name with escaped quotes, backslashes and controls");
    expect(strix::decode_receipt::escape_json_string(std::string("a\0b", 3)) == "a\\u0000b",
           "receipt escapes embedded null without truncating name");

    // The receipt binds the exact runtime owner before work, without latching
    // a zero at construction. Updates below test counter observation only;
    // they do not claim that a GPU projection was launched by this host test.
    auto stats_owner = std::make_shared<shared_expert_prefetch::RuntimeDispatchStats>();
    std::weak_ptr<shared_expert_prefetch::RuntimeDispatchStats> live_owner = stats_owner;
    const decode_detail::SharedPrefetchDispatchReceipt completed_receipt(stats_owner);
    expect(completed_receipt.completed_counts() ==
               "  \"prefetch_q8_shared_expert_dispatched_projections\": 0,\n",
           "pre-execution reader observes actual owner zero");
    stats_owner.reset();
    expect(!live_owner.expired(), "deferred completion reader retains runtime owner");
    auto producer_owner = live_owner.lock();
    producer_owner->dispatched_projections.fetch_add(1, std::memory_order_relaxed);
    expect(completed_receipt.completed_counts() ==
               "  \"prefetch_q8_shared_expert_dispatched_projections\": 1,\n",
           "completion reader observes enqueue update after its construction");
    producer_owner->dispatched_projections.fetch_add(2, std::memory_order_relaxed);
    shared_expert_prefetch::RuntimeDispatchStats unrelated_owner;
    unrelated_owner.dispatched_projections.fetch_add(99, std::memory_order_relaxed);
    expect(completed_receipt.completed_counts() ==
               "  \"prefetch_q8_shared_expert_dispatched_projections\": 3,\n",
           "completion reader follows its actual owner, not another owner or token-derived total");
    const decode_detail::SharedPrefetchDispatchReceipt disabled_receipt(nullptr);
    expect(disabled_receipt.completed_counts().empty(), "disabled mode omits dynamic feature counter");

    const auto m = ModelContract::qwen38_flash_next();
    auto tensors = manifest(m);
    auto p = shared_expert_prefetch::build_plan(m, tensors);
    expect(p.layers.size() == 48 && p.eligible_projections == 144 && p.fallback_projections == 0,
           "canonical complete shared plan");
    const auto execution = build_serial_decode_execution_plan(m);
    auto requirements = build_serial_execution_tensor_requirements(execution);
    std::erase_if(requirements, [](const auto &r) { return r.op != SerialOp::shared_expert; });
    tensors.layout_identity = model_manifest_layout_identity(tensors);
    const auto access = compile_execution_tensor_access_plan(
        execution, tensors, build_model_tensor_index(tensors), requirements);
    shared_expert_prefetch::validate_tensor_access(p, tensors, execution, access);
    auto wrong_access = access;
    wrong_access.accesses.front().tensor_indices.clear();
    rejects(
        [&] {
            shared_expert_prefetch::validate_tensor_access(p, tensors, execution, wrong_access);
        },
        "shared prefetch projection differs from tensor access");
    ManifestTensor narrow;
    narrow.ggml_type = 8;
    narrow.dimensions = {33, 1};
    expect(!shared_expert_prefetch::eligible(narrow), "nonmultiple32 fallback");
    narrow.dimensions = {32, 1};
    expect(shared_expert_prefetch::eligible(narrow), "one Q8 group eligible");
    for (auto &t : tensors.tensors)
        if (t.layer == 0U || t.name == "blk.1.shared_expert.gate_proj.weight") {
            t.ggml_type = 0;
            t.byte_size = t.dimensions[0] * t.dimensions[1] * 4U;
        }
    p = shared_expert_prefetch::build_plan(m, tensors);
    expect(p.eligible_steps == 47 && p.fallback_steps == 1 && p.eligible_projections == 140 &&
               p.fallback_projections == 4,
           "mixed gate/up/down and layer fallbacks counted");
    ManifestTensor scalar;
    scalar.name = "blk.1.shared_expert_gate.weight";
    scalar.role = TensorRole::dense_projection;
    scalar.layer = 1;
    scalar.dimensions = {m.hidden_size};
    tensors.tensors.push_back(scalar);
    expect(shared_expert_prefetch::build_plan(m, tensors).projections.size() == 144,
           "scalar sigmoid gate remains excluded");
    ExecutionBindingRequest off;
    off.minimum_qualification = kernels::QualificationStatus::experimental;
    ExecutionBufferRequest buffers;
    auto reg = registry(m, off, true);
    const auto original = compile_serial_execution(m, reg, off, buffers);
    auto on = off;
    on.prefetch_q8_shared_expert = true;
    on.shared_expert_prefetch_layers = p.layers;
    const auto selected = compile_serial_execution(m, reg, on, buffers);
    expect(original.execution_identity != selected.execution_identity,
           "prefetch changes execution identity");
    unsigned base_count = 0, prefetch_count = 0;
    for (const auto &binding : selected.bindings.bindings)
        if (binding.op == SerialOp::shared_expert) {
            const bool use = binding.kernel_id->ends_with(shared_expert_prefetch::kernel_suffix);
            if (use)
                ++prefetch_count;
            else
                ++base_count;
            expect(use == (*selected.execution.steps[binding.step_id].layer != 0U),
                   "explicit per-layer IDs");
        }
    expect(base_count == 1 && prefetch_count == 47, "mixed base/prefetch bindings");
    auto base_reg = registry(m, off, false);
    expect(compile_serial_execution(m, base_reg, off, buffers).execution_identity ==
               original.execution_identity,
           "OFF identity unaffected by extra experimental registrations");
    rejects([&] { (void)compile_serial_execution(m, base_reg, on, buffers); },
            "compiled serial execution requires every kernel step to bind");
    auto bad = on;
    bad.shared_expert_prefetch_layers.clear();
    rejects([&] { (void)compile_serial_execution(m, reg, bad, buffers); },
            "shared prefetch requires eligible step");
    bad = off;
    bad.shared_expert_prefetch_layers = {0};
    rejects([&] { (void)compile_serial_execution(m, reg, bad, buffers); },
            "shared prefetch layers require opt-in");
    bad = on;
    bad.shared_expert_prefetch_layers = {0, 0};
    rejects([&] { (void)compile_serial_execution(m, reg, bad, buffers); },
            "shared prefetch eligible layers invalid");
    bad = on;
    bad.shared_expert_prefetch_layers = {48};
    rejects([&] { (void)compile_serial_execution(m, reg, bad, buffers); },
            "shared prefetch eligible layers invalid");
    SpeculativeVerifyBindingRequest verify;
    verify.minimum_qualification = kernels::QualificationStatus::experimental;
    verify.prefetch_q8_shared_expert = true;
    rejects([&] { (void)compile_speculative_verify_execution(m, 1, reg, verify, buffers); },
            "shared prefetch runtime does not support speculative verify");
    rejects(
        [&] {
            (void)bind_speculative_verify_plan(m, build_speculative_verify_plan(m, 1), reg, verify);
        },
        "shared prefetch runtime does not support speculative verify");
    rejects([&] { (void)shared_expert_prefetch::build_plan(m, manifest(m, false)); },
            "shared prefetch requires eligible step");
    auto malformed = manifest(m);
    malformed.tensors.pop_back();
    rejects([&] { (void)shared_expert_prefetch::build_plan(m, malformed); },
            "shared prefetch missing projection family");
    malformed = manifest(m);
    malformed.tensors.push_back(malformed.tensors[0]);
    rejects([&] { (void)shared_expert_prefetch::build_plan(m, malformed); },
            "shared prefetch ambiguous projection");
    malformed = manifest(m);
    malformed.tensors[0].absolute_offset = 8;
    rejects([&] { (void)shared_expert_prefetch::build_plan(m, malformed); },
            "shared prefetch weight alignment/extent invalid");
    malformed = manifest(m);
    --malformed.tensors[0].byte_size;
    rejects([&] { (void)shared_expert_prefetch::build_plan(m, malformed); },
            "shared prefetch weight alignment/extent invalid");
    rejects([&] { (void)shared_expert_prefetch::required_weight_bytes(32, UINT64_MAX); },
            "shared prefetch geometry overflow");
    rejects([&] { (void)shared_expert_prefetch::required_weight_bytes(33, 1); },
            "shared prefetch Q8 geometry invalid");
    expect(shared_expert_prefetch::required_weight_bytes(640, 2560) == 1740800ULL,
           "canonical down checked extent");
    std::cout << "Shared expert prefetch host tests failures=" << failures << '\n';
    return failures ? 1 : 0;
}
