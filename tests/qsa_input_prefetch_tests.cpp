#include "strix/compiled_execution.hpp"
#include "strix/compiled_speculative_verify.hpp"
#include "strix/qsa_input_prefetch.hpp"
#include <array>
#include <iostream>
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
template <class F> void rejects(F f) {
    try {
        f();
        expect(false, "expected invalid_argument");
    } catch (const std::invalid_argument &) {
    }
}
ModelManifest manifest(const ModelContract &m, bool q8 = true) {
    ModelManifest result;
    const std::array names{"attn_q.weight", "attn_k.weight", "attn_v.weight",
                           "indexer.q_proj.weight", "indexer.k_proj.weight"};
    const std::array outputs{m.attention_heads * m.head_dim * 2U, m.kv_heads * m.head_dim,
                             m.kv_heads * m.head_dim, m.indexer_n_heads * m.indexer_head_dim,
                             m.indexer_head_dim};
    for (std::uint32_t layer = 0; layer < m.hidden_layers; ++layer)
        if ((layer + 1U) % m.full_attention_interval == 0)
            for (unsigned i = 0; i < 5; ++i) {
                ManifestTensor t;
                t.name = "blk." + std::to_string(layer) + "." + names[i];
                t.layer = layer;
                t.dimensions = {m.hidden_size, outputs[i]};
                t.ggml_type = q8 ? 8U : 0U;
                t.absolute_offset = 0x1000;
                t.byte_size =
                    q8 ? qsa_input_prefetch::required_weight_bytes(m.hidden_size, outputs[i])
                       : m.hidden_size * outputs[i] * 4U;
                result.tensors.push_back(t);
            }
    return result;
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
        if (include_prefetch && step.op == SerialOp::indexed_attention) {
            v.id += qsa_input_prefetch::kernel_suffix;
            v.compiler_identity = "host-only-prefetch";
            out.add(v);
        }
    }
    return out;
}
} // namespace
int main() {
    const auto m = ModelContract::qwen38_flash_next();
    auto tensors = manifest(m);
    auto p = qsa_input_prefetch::build_plan(m, tensors);
    expect(p.layers.size() == 12 && p.eligible_projections == 60 && p.fallback_projections == 0,
           "canonical complete plan");
    expect(!qsa_input_prefetch::eligible(ManifestTensor{}), "empty not eligible");
    ManifestTensor narrow;
    narrow.ggml_type = 8;
    narrow.dimensions = {33, 1};
    expect(!qsa_input_prefetch::eligible(narrow), "nonmultiple32 fallback");
    narrow.dimensions = {32, 1};
    expect(qsa_input_prefetch::eligible(narrow), "one Q8 group eligible");
    for (auto &t : tensors.tensors)
        if (t.layer == 3 || t.name == "blk.7.attn_q.weight") {
            t.ggml_type = 0;
            t.byte_size = t.dimensions[0] * t.dimensions[1] * 4U;
        }
    p = qsa_input_prefetch::build_plan(m, tensors);
    expect(p.eligible_steps == 11 && p.fallback_steps == 1 && p.eligible_projections == 54 &&
               p.fallback_projections == 6,
           "mixed layer and tensor fallbacks counted");
    ExecutionBindingRequest off;
    off.minimum_qualification = kernels::QualificationStatus::experimental;
    ExecutionBufferRequest buffers;
    auto reg = registry(m, off, true);
    const auto original = compile_serial_execution(m, reg, off, buffers);
    auto on = off;
    on.prefetch_q8_qsa_input = true;
    on.qsa_input_prefetch_layers = p.layers;
    const auto selected = compile_serial_execution(m, reg, on, buffers);
    expect(original.execution_identity != selected.execution_identity,
           "prefetch changes execution identity");
    unsigned base_count = 0, prefetch_count = 0;
    for (const auto &b : selected.bindings.bindings)
        if (b.op == SerialOp::indexed_attention) {
            const bool use = b.kernel_id->ends_with(qsa_input_prefetch::kernel_suffix);
            if (use)
                ++prefetch_count;
            else
                ++base_count;
            expect(use == (*selected.execution.steps[b.step_id].layer != 3U),
                   "explicit per-layer IDs");
        }
    expect(base_count == 1 && prefetch_count == 11, "mixed base/prefetch bindings");
    auto base_reg = registry(m, off, false);
    const auto original_only = compile_serial_execution(m, base_reg, off, buffers);
    expect(original_only.execution_identity == original.execution_identity,
           "OFF identity unaffected by extra experimental registrations");
    rejects([&] { (void)compile_serial_execution(m, base_reg, on, buffers); });
    auto bad = on;
    bad.qsa_input_prefetch_layers.clear();
    rejects([&] { (void)compile_serial_execution(m, reg, bad, buffers); });
    bad = off;
    bad.qsa_input_prefetch_layers = {3};
    rejects([&] { (void)compile_serial_execution(m, reg, bad, buffers); });
    bad = on;
    bad.qsa_input_prefetch_layers = {3, 3};
    rejects([&] { (void)compile_serial_execution(m, reg, bad, buffers); });
    bad = on;
    bad.qsa_input_prefetch_layers = {0};
    rejects([&] { (void)compile_serial_execution(m, reg, bad, buffers); });
    SpeculativeVerifyBindingRequest verify;
    verify.prefetch_q8_qsa_input = true;
    rejects([&] { (void)compile_speculative_verify_execution(m, 1, reg, verify, buffers); });
    rejects([&] { (void)qsa_input_prefetch::build_plan(m, manifest(m, false)); });
    auto malformed = manifest(m);
    malformed.tensors.pop_back();
    rejects([&] { (void)qsa_input_prefetch::build_plan(m, malformed); });
    malformed = manifest(m);
    malformed.tensors.push_back(malformed.tensors[0]);
    rejects([&] { (void)qsa_input_prefetch::build_plan(m, malformed); });
    malformed = manifest(m);
    malformed.tensors[0].absolute_offset = 8;
    rejects([&] { (void)qsa_input_prefetch::build_plan(m, malformed); });
    malformed = manifest(m);
    --malformed.tensors[0].byte_size;
    rejects([&] { (void)qsa_input_prefetch::build_plan(m, malformed); });
    rejects([&] { (void)qsa_input_prefetch::required_weight_bytes(32, UINT64_MAX); });
    rejects([&] { (void)qsa_input_prefetch::required_weight_bytes(33, 1); });
    expect(qsa_input_prefetch::required_weight_bytes(2560, 12288) == 33423360ULL,
           "canonical checked extent");
    std::cout << "QSA input prefetch host tests failures=" << failures << '\n';
    return failures ? 1 : 0;
}
