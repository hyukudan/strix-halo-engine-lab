#include "strix/compiled_execution.hpp"
#include "strix/execution_memory.hpp"

#include <set>
#include <string>
#include <utility>

#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::kernels::KernelVariant make_variant(
    std::string id,
    strix::kernels::OperatorFamily family,
    const strix::kernels::ShapeClass& shape) {
    strix::kernels::KernelVariant variant;
    variant.id = std::move(id);
    variant.family = family;
    variant.phase =
        strix::kernels::ExecutionPhase::decode;
    variant.shape = shape;
    variant.wave =
        strix::kernels::WaveSize::wave32;
    variant.resources.workgroup_threads = 256;
    variant.numerics.input_precision = "bf16";
    variant.numerics.accumulator_precision = "fp32";
    variant.numerics.output_precision = "bf16";
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        strix::kernels::QualificationStatus::oracle_matched;
    variant.graph_safe = true;
    variant.input_layout = "test";
    variant.output_layout = "test";
    variant.compiler_identity = "test";
    return variant;
}

strix::kernels::KernelRegistry make_registry(
    const strix::ModelContract& model,
    const strix::ExecutionBindingRequest& request) {
    strix::kernels::KernelRegistry registry;
    const auto execution =
        strix::build_serial_decode_execution_plan(model);
    std::set<std::string> added;

    for (const auto& step : execution.steps) {
        const auto family =
            strix::operator_family_for(step.op);
        if (!family.has_value()) {
            continue;
        }
        const auto shape =
            strix::execution_shape_for(
                model, step.op, request);
        const auto key =
            std::to_string(
                static_cast<unsigned>(*family)) +
            "." +
            std::to_string(shape.auxiliary_size);
        if (!added.insert(key).second) {
            continue;
        }
        registry.add(
            make_variant(
                "memory." + key,
                *family,
                shape));
    }
    return registry;
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    strix::ExecutionBindingRequest binding_request;
    strix::ExecutionBufferRequest buffer_request;

    const auto registry =
        make_registry(model, binding_request);
    const auto compiled =
        strix::compile_serial_execution(
            model,
            registry,
            binding_request,
            buffer_request,
            256);

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity =
        compiled.execution_identity;
    memory.arena_base =
        static_cast<std::uintptr_t>(
            0x10000000ULL);
    memory.arena_bytes =
        compiled.arena.total_bytes;
    memory.allocation_generation = 7;

    strix::validate_execution_memory_binding(
        compiled.execution_identity,
        compiled.arena,
        memory);

    const auto& lm_io =
        compiled.io.steps.back();
    const auto input =
        strix::resolve_execution_buffer(
            compiled.execution_identity,
            compiled.arena,
            memory,
            lm_io.reads[0]);
    const auto output =
        strix::resolve_execution_buffer(
            compiled.execution_identity,
            compiled.arena,
            memory,
            lm_io.writes[0]);

    expect(input.address ==
               memory.arena_base +
                   strix::find_arena_allocation(
                       compiled.arena,
                       lm_io.reads[0])->offset,
           "resolved input address uses compiled arena offset");
    expect(output.allocation_generation == 7,
           "resolved buffer carries physical allocation generation");
    expect(input.address != output.address,
           "simultaneously live LM input/output resolve to different addresses");

    auto wrong_identity = memory;
    ++wrong_identity.execution_identity;
    bool identity_rejected = false;
    try {
        strix::validate_execution_memory_binding(
            compiled.execution_identity,
            compiled.arena,
            wrong_identity);
    } catch (const std::invalid_argument&) {
        identity_rejected = true;
    }
    expect(identity_rejected,
           "memory binding rejects compiled-plan identity mismatch");

    auto undersized = memory;
    --undersized.arena_bytes;
    bool size_rejected = false;
    try {
        strix::validate_execution_memory_binding(
            compiled.execution_identity,
            compiled.arena,
            undersized);
    } catch (const std::invalid_argument&) {
        size_rejected = true;
    }
    expect(size_rejected,
           "memory binding rejects undersized physical arena");

    auto null_base = memory;
    null_base.arena_base = 0;
    bool null_rejected = false;
    try {
        strix::validate_execution_memory_binding(
            compiled.execution_identity,
            compiled.arena,
            null_base);
    } catch (const std::invalid_argument&) {
        null_rejected = true;
    }
    expect(null_rejected,
           "memory binding rejects null base address");

    auto overflow = memory;
    overflow.arena_base =
        std::numeric_limits<std::uintptr_t>::max() -
        static_cast<std::uintptr_t>(
            compiled.arena.total_bytes / 2U);
    bool overflow_rejected = false;
    try {
        strix::validate_execution_memory_binding(
            compiled.execution_identity,
            compiled.arena,
            overflow);
    } catch (const std::overflow_error&) {
        overflow_rejected = true;
    }
    expect(overflow_rejected,
           "memory binding rejects address-range overflow");

    auto declared_overflow = memory;
    declared_overflow.arena_bytes = std::numeric_limits<std::uint64_t>::max();
    bool declared_rejected = false;
    try { strix::validate_execution_memory_binding(compiled.execution_identity, compiled.arena, declared_overflow); }
    catch (const std::overflow_error&) { declared_rejected = true; }
    expect(declared_rejected, "entire declared physical view must fit the address space");
    for (const bool oversized_physical : {false, true}) {
        auto forged = compiled.arena;
        for (auto& allocation : forged.allocations) if (allocation.use == lm_io.reads[0]) {
            allocation.offset = memory.arena_bytes + 1U;
            allocation.bytes = 4;
        }
        auto physical = memory;
        if (oversized_physical) physical.arena_bytes += 1024;
        bool rejected = false;
        try { static_cast<void>(strix::resolve_execution_buffer(compiled.execution_identity, forged, physical, lm_io.reads[0])); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "forged offsets cannot underflow bounds or enter excess physical capacity");
    }

    if (failures == 0) {
        std::cout << "execution memory tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
