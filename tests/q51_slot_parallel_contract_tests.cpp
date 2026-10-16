#include "strix/q51_slot_parallel_contract.hpp"
#include "strix/moe_q4k_a8_runtime.hpp"
#include "strix/moe_fork_join_arena.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
namespace slot = strix::q51_slot_parallel;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class Function> void reject(Function&& function, const char* message) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}
void view_contract() {
    slot::Request request{true, slot::numerical_identity, {67,288,3,3}};
    const auto sizes = strix::q51_grouped_down::required_bytes(request.shape);
    slot::Bindings bindings{{0x100000,sizes.intermediates},{0x200000,sizes.weights},
        {0x300000,sizes.routes},{0x400000,sizes.route_weights},{0x500000,4},
        {0x600000,sizes.accumulation},{0x700000,3U*67U*4U}};
    slot::validate(request, bindings);
    require(slot::required_rawdots_bytes({2560,640,512,10}) == 102400,
        "canonical raw-dot allocation is ten full FP32 output rows");
    auto invalid_request = request;
    invalid_request.enabled = false;
    reject([&]{slot::validate(invalid_request,bindings);}, "explicit enable required");
    invalid_request = request; invalid_request.approved_numerical_identity = "other";
    reject([&]{slot::validate(invalid_request,bindings);}, "explicit numerical contract required");
    for (const auto shape : {slot::Shape{2561,640,512,10}, slot::Shape{2560,641,512,10},
                            slot::Shape{2560,640,513,10}, slot::Shape{2560,640,512,11}})
        reject([&]{(void)slot::required_rawdots_bytes(shape);}, "unsupported geometry rejected before products");
    for (unsigned fault = 0; fault < 6; ++fault) {
        auto bad = bindings;
        if (fault == 0) bad.rawdots.bytes -= 4;
        if (fault == 1) bad.rawdots.bytes += 4;
        if (fault == 2) bad.rawdots.address += 4;
        if (fault == 3) bad.rawdots.address -= 4;
        if (fault == 4) bad.rawdots.address = 0;
        if (fault == 5) bad.rawdots.address = std::numeric_limits<std::uintptr_t>::max() & ~std::uintptr_t{255};
        reject([&]{slot::validate(request,bad);}, "raw-dot extent, alignment or end overflow rejected");
    }
    const std::array owners{bindings.intermediates,bindings.weights,bindings.routes,
                            bindings.route_weights,bindings.status,bindings.accumulation};
    for (const auto owner : owners) {
        auto bad = bindings; bad.rawdots.address = owner.address;
        reject([&]{slot::validate(request,bad);}, "raw dots cannot alias any grouped owner");
    }
    auto bad = bindings; bad.rawdots.address = bindings.accumulation.address - 256;
    reject([&]{slot::validate(request,bad);}, "partial raw-dot overlap rejected");
    bad = bindings; --bad.intermediates.bytes;
    reject([&]{slot::validate(request,bad);}, "original grouped view contract remains enforced");
    auto readonly_alias = bindings; readonly_alias.routes.address = readonly_alias.route_weights.address;
    slot::validate(request,readonly_alias);
}

strix::CompiledSerialExecution source_plan(bool enabled) {
    const auto model = strix::ModelContract::qwen38_flash_next();
    strix::CompiledSerialExecution source;
    source.execution = strix::build_serial_decode_execution_plan(model);
    source.io = strix::build_serial_execution_io_plan(source.execution, true);
    strix::ExecutionBufferRequest buffers;
    buffers.parallel_moe_router = true; buffers.q4k_a8g32_gate_up = true;
    buffers.grouped_exact_q51_down = true; buffers.q51_slot_parallel_down = enabled;
    source.arena = strix::build_serial_execution_arena_plan(model,buffers,source.execution,source.io);
    // A deliberately equal synthetic identity proves the stale-layout rejection
    // compares actual metadata as well. This is no compiled-launcher evidence.
    source.execution_identity = 0x5135314152454e41ULL;
    return source;
}

void arena_contract() {
    const auto model = strix::ModelContract::qwen38_flash_next();
    strix::ExecutionBufferRequest buffers;
    buffers.q4k_a8g32_gate_up = true; buffers.grouped_exact_q51_down = true;
    const auto old_buffers = strix::build_serial_execution_buffer_plan(model,buffers);
    buffers.q51_slot_parallel_down = true;
    const auto new_buffers = strix::build_serial_execution_buffer_plan(model,buffers);
    require(old_buffers.mlp_intermediate_bytes == 65280 && new_buffers.mlp_intermediate_bytes == 167680,
        "buffer plan charges the full flag-aware MLP temporary class");
    const auto before = source_plan(false), after = source_plan(true);
    for (const auto& allocation : after.arena.allocations)
        if (allocation.use.buffer.buffer_class == strix::ExecutionBufferClass::mlp_intermediate)
            require(allocation.bytes == new_buffers.mlp_intermediate_bytes,
                "every shared and routed temporary grows, including fallback layers");
    const auto old_join = strix::prepare_moe_fork_join_arena(before,std::numeric_limits<std::uint64_t>::max());
    const auto new_join = strix::prepare_moe_fork_join_arena(after,std::numeric_limits<std::uint64_t>::max());
    require(new_join.shared_region_bytes() == old_join.shared_region_bytes() + 102400 &&
            new_join.arena().total_bytes + before.arena.total_bytes ==
                old_join.arena().total_bytes + after.arena.total_bytes + 102400,
        "fork-join charges the larger shared tail in addition to the serial arena change");
    reject([&]{(void)old_join.rebind_execution(after);}, "old sealed fork-join plan rejects larger source arena");
    reject([&]{(void)new_join.rebind_execution(before);}, "new sealed fork-join plan rejects incumbent source arena");
    bool insufficient_budget = false;
    try { (void)strix::prepare_moe_fork_join_arena(after,new_join.arena().total_bytes-1U); }
    catch (const std::length_error&) { insufficient_budget = true; }
    require(insufficient_budget, "full enlarged fork-join budget required");
    buffers.grouped_exact_q51_down = false;
    reject([&]{(void)strix::build_serial_execution_buffer_plan(model,buffers);}, "slot-parallel requires grouped mode");
    buffers.grouped_exact_q51_down = true; buffers.q4k_a8g32_gate_up = false;
    reject([&]{(void)strix::build_serial_execution_buffer_plan(model,buffers);}, "slot-parallel requires A8 layout");
    strix::moe_q4k_a8::RuntimeLayerClassification classified;
    for (std::uint32_t layer = 0; layer < 48U; ++layer)
        if (layer != 2U && layer != 4U && layer != 30U && layer != 46U && layer != 47U)
            classified.q4k_with_q51_down_layers.push_back(layer);
    strix::moe_q4k_a8::validate_slot_parallel_manifest(model,classified);
    auto missing = classified; missing.q4k_with_q51_down_layers.pop_back();
    reject([&]{strix::moe_q4k_a8::validate_slot_parallel_manifest(model,missing);}, "late eligible layer missing");
    auto extra = classified; extra.q4k_with_q51_down_layers.push_back(47U);
    reject([&]{strix::moe_q4k_a8::validate_slot_parallel_manifest(model,extra);}, "fallback layer cannot enter parallel set");
    auto fewer = model; fewer.hidden_layers=44; fewer.gdn_layers=33; fewer.full_attention_layers=11;
    require(fewer.validate().empty(), "noncanonical layer-count negative is structurally valid");
    reject([&]{strix::moe_q4k_a8::validate_slot_parallel_manifest(fewer,classified);}, "canonical full model required");
    std::cout << "{\"q51_slot_parallel_workspace_report\":{\"derived_from_host_planner\":true,"
        "\"mlp_off\":" << old_buffers.mlp_intermediate_bytes << ",\"mlp_on\":" << new_buffers.mlp_intermediate_bytes
        << ",\"serial_off\":" << before.arena.total_bytes << ",\"serial_on\":" << after.arena.total_bytes
        << ",\"fork_join_off\":" << old_join.arena().total_bytes << ",\"fork_join_on\":"
        << new_join.arena().total_bytes << "}}\n";
}
} // namespace
int main() try { view_contract(); arena_contract(); return 0; }
catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
