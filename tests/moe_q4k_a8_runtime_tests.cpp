#include "strix/moe_q4k_a8_runtime.hpp"
#include "strix/execution_buffers.hpp"
#include "strix/raw_baseline_recording.hpp"
#include "strix/integer_projection_runtime.hpp"
#include <iostream>
#include <stdexcept>

namespace {
int failures = 0;
void check(bool v, const char* message) { if (!v) { ++failures; std::cerr << message << '\n'; } }
template<class F> void rejects(F fn, const char* message) {
    try { fn(); check(false, message); } catch (const std::invalid_argument&) {}
}
strix::ModelManifest manifest(const strix::ModelContract& model) {
    strix::ModelManifest m;
    for (std::uint32_t layer = 0; layer < model.hidden_layers; ++layer) {
        for (unsigned slot = 0; slot < 3; ++slot) {
            strix::ManifestTensor t;
            t.layer = layer; t.role = strix::TensorRole::moe_expert;
            t.name = "blk." + std::to_string(layer) + (slot == 0 ? ".ffn_gate_exps.weight" :
                slot == 1 ? ".ffn_up_exps.weight" : ".ffn_down_exps.weight");
            const bool fallback = layer == 2;
            t.ggml_type = slot < 2 ? (fallback ? 13U : 12U) : (fallback ? 8U : 7U);
            t.dimensions = slot < 2 ? std::vector<std::uint64_t>{2560,640,512} :
                std::vector<std::uint64_t>{640,2560,512};
            t.byte_size = slot < 2 ? (fallback ? 576716800ULL : 471859200ULL) :
                (fallback ? 891289600ULL : 629145600ULL);
            m.tensors.push_back(t);
        }
    }
    return m;
}
}
int main() {
    using namespace strix::moe_q4k_a8;
    const auto l = build_runtime_workspace(2560,640,512,10);
    check(l.gate_up_offset == 0 && l.gate_up_bytes == 51200 && l.accumulation_offset == 51200 &&
        l.accumulation_bytes == 10240 && l.status_offset == 61440 && l.codes_offset == 61696 &&
        l.deltas_offset == 64256 && l.sums_offset == 64768 && l.total_bytes == 65280,
        "canonical typed regions");
    for (unsigned h = 256; h <= 2560; h += 256) for (unsigned i = 32; i <= 640; i += 32)
        for (unsigned k = 1; k <= 10; ++k) {
            const auto q = build_runtime_workspace(h,i,512,k);
            check(q.accumulation_offset >= q.gate_up_bytes &&
                q.status_offset == q.accumulation_offset + q.accumulation_bytes &&
                q.codes_offset >= q.status_offset + 4 && q.deltas_offset >= q.codes_offset + q.codes_bytes &&
                q.sums_offset >= q.deltas_offset + q.deltas_bytes && q.total_bytes >= q.sums_offset + q.sums_bytes &&
                (q.accumulation_offset | q.codes_offset | q.deltas_offset | q.sums_offset | q.total_bytes) % 256 == 0,
                "all admitted regions bounded, disjoint and aligned");
            const auto off = build_runtime_workspace(h,i,512,k,false);
            const auto on = build_runtime_workspace(h,i,512,k,true);
            check(off.total_bytes == q.total_bytes && off.rawdots_offset == 0 && off.rawdots_bytes == 0 &&
                off.gate_up_bytes == q.gate_up_bytes && off.accumulation_offset == q.accumulation_offset &&
                off.status_offset == q.status_offset && off.codes_offset == q.codes_offset &&
                off.deltas_offset == q.deltas_offset && off.sums_offset == q.sums_offset,
                "old four-argument layout equals explicit-off across admitted geometry");
            check(on.rawdots_offset == q.total_bytes && on.rawdots_bytes == static_cast<std::uint64_t>(k)*h*4U &&
                on.total_bytes == on.rawdots_offset + on.rawdots_bytes &&
                on.gate_up_bytes == q.gate_up_bytes && on.accumulation_offset == q.accumulation_offset &&
                on.status_offset == q.status_offset && on.codes_offset == q.codes_offset &&
                on.deltas_offset == q.deltas_offset && on.sums_offset == q.sums_offset,
                "raw dots append without moving any original internal region");
        }
    for (auto h : {0ULL,128ULL,257ULL,2816ULL,~0ULL})
        rejects([&]{(void)build_runtime_workspace(h,640,512,10);}, "invalid H");
    rejects([]{(void)build_runtime_workspace(2560,641,512,10);}, "invalid I");
    rejects([]{(void)build_runtime_workspace(2560,640,513,10);}, "invalid E");
    rejects([]{(void)build_runtime_workspace(2560,640,5,10);}, "K exceeds E");
    rejects([]{(void)build_runtime_workspace(2560,640,512,11);}, "invalid K");
    const auto model = strix::ModelContract::qwen38_flash_next();
    const auto m = manifest(model);
    const auto c = classify_runtime_manifest(model,m);
    check(c.quantized_layers.size() == 47 && c.legacy_q5k_layers == std::vector<std::uint32_t>{2},
        "synthetic canonical manifest classification 47 Q4K / layer2 Q5K");
    check(c.q4k_with_q51_down_layers.size()==47,"synthetic manifest classifies Q4K/Q5_1 targets");
    auto bad = m;
    bad.tensors.pop_back(); rejects([&]{(void)classify_runtime_manifest(model,bad);}, "missing down");
    bad=m; bad.tensors.push_back(m.tensors[0]); rejects([&]{(void)classify_runtime_manifest(model,bad);}, "duplicate gate");
    bad=m; bad.tensors[0].name="ffn_gate_up_exps.weight"; rejects([&]{(void)classify_runtime_manifest(model,bad);}, "fused gateup");
    bad=m; bad.tensors[0].ggml_type=13; rejects([&]{(void)classify_runtime_manifest(model,bad);}, "mixed gateup types");
    bad=m; bad.tensors[0].dimensions[0]=2559; rejects([&]{(void)classify_runtime_manifest(model,bad);}, "bad shape");
    bad=m; --bad.tensors[0].byte_size; rejects([&]{(void)classify_runtime_manifest(model,bad);}, "bad byte size");
    bad=m; bad.tensors[0].absolute_offset=1; rejects([&]{(void)classify_runtime_manifest(model,bad);}, "misaligned weight");
    bad=m; bad.tensors[0].layer.reset(); rejects([&]{(void)classify_runtime_manifest(model,bad);}, "missing layer");
    bad=m; bad.tensors[0].layer=48; rejects([&]{(void)classify_runtime_manifest(model,bad);}, "invalid layer");
    bad=m; bad.tensors[2].ggml_type=99; rejects([&]{(void)classify_runtime_manifest(model,bad);}, "unsupported down");
    bad=m; for (auto& t : bad.tensors) if (t.ggml_type==12) {t.ggml_type=13;t.byte_size=576716800;}
    rejects([&]{(void)classify_runtime_manifest(model,bad);}, "no quantized layer");
    strix::ExecutionBufferRequest request;
    const auto old = strix::build_serial_execution_buffer_plan(model,request);
    check(old.mlp_intermediate_bytes == 15364, "default workspace golden unchanged");
    request.q4k_a8g32_gate_up=true;
    check(strix::build_serial_execution_buffer_plan(model,request).mlp_intermediate_bytes >= l.total_bytes,
        "admitted workspace covers all typed regions");
    request.grouped_exact_q51_down=true;
    check(strix::build_serial_execution_buffer_plan(model,request).mlp_intermediate_bytes==l.total_bytes,
        "grouped exact Q5_1 down reuses existing typed workspace");
    request.q4k_a8g32_gate_up=false;
    rejects([&]{(void)strix::build_serial_execution_buffer_plan(model,request);},"grouped Q5_1 needs A8 layout");
    request.q4k_a8g32_gate_up=true;
    request.logical_rows=2; rejects([&]{(void)strix::build_serial_execution_buffer_plan(model,request);}, "mode rejects rows2");
    request.logical_rows=1; request.control_element_bytes=2;
    rejects([&]{(void)strix::build_serial_execution_buffer_plan(model,request);}, "mode rejects FP16 controls");
    const auto combined=[&](std::string prefix) {
        prefix += ";" + std::string(runtime_numerical_identity) + ";" + std::string(grouped_q51_raw_identity);
        check(prefix.size()<=512,"combined RAW numerical identity remains inside512 byte contract");
    };
    combined(strix::raw_floating_lm_head_numerical_identity(strix::RawBaselineActivationStorage::bf16));
    combined(strix::raw_floating_lm_head_numerical_identity(strix::RawBaselineActivationStorage::fp32));
    for(const auto weight:{strix::integer_projection::WeightEncoding::q4_0,strix::integer_projection::WeightEncoding::iq4_xs})
        for(const auto activation:{strix::low_precision::ActivationEncoding::symmetric_a8,strix::low_precision::ActivationEncoding::symmetric_a4})
            for(const auto instruction:{strix::integer_projection::Instruction::iu8,strix::integer_projection::Instruction::iu4}) {
                if(instruction==strix::integer_projection::Instruction::iu4 &&
                   (weight!=strix::integer_projection::WeightEncoding::q4_0 || activation!=strix::low_precision::ActivationEncoding::symmetric_a4)) continue;
                for(const auto input:{strix::integer_projection::RuntimeInput::bf16,strix::integer_projection::RuntimeInput::fp32}) {
                    strix::integer_projection::RuntimeConfig config;
                    config.weights=weight; config.activations=activation; config.instruction=instruction; config.input=input;
                    combined(strix::integer_projection::runtime_numerical_identity(config));
                }
            }
    return failures ? 1 : 0;
}
