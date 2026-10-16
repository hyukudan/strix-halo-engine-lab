#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_moe_experts_launcher.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
#include "q4k_a8_oracle.hpp"
#endif

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <array>
#include <bit>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(
    bool condition,
    const char* message) {
    if (!condition) {
        std::cerr
            << "FAIL: "
            << message
            << '\n';
        ++failures;
    }
}

void check(
    hipError_t error,
    const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            hipGetErrorString(error));
    }
}

struct TestWeights {
    std::vector<std::byte> packed;
    std::vector<float> values;
    std::uint32_t type = 0;
    const void* data() const { return packed.data(); }
    std::size_t size() const { return packed.size(); }
};

const strix::ExecutionStep&
first_expert_step(
    const strix::SerialExecutionPlan& plan) {
    for (const auto& step :
         plan.steps) {
        if (step.op ==
            strix::SerialOp::moe_experts) {
            return step;
        }
    }
    throw std::runtime_error(
        "serial plan has no routed-expert step");
}

// Expected values are generated before encoding. The oracle never decodes
// packed bytes or calls a production quantization helper.
TestWeights make_weights(std::uint32_t experts, std::uint32_t output_width,
                         std::uint32_t input_width, std::uint32_t salt,
                         std::uint32_t type) {
    TestWeights result;
    result.type = type;
    const std::uint32_t elements = type == 7U ? 32U : 256U;
    const std::uint32_t bytes = type == 7U ? 24U : (type == 12U ? 144U : 176U);
    const auto total = static_cast<std::size_t>(experts) * output_width * input_width;
    result.packed.resize(total / elements * bytes);
    result.values.resize(total);
    auto set_bit = [](std::byte* target, unsigned int bit, unsigned int value) {
        target[bit / 8U] |= static_cast<std::byte>((value & 1U) << (bit % 8U));
    };
    for (std::size_t block = 0; block < total / elements; ++block) {
        auto* out = result.packed.data() + block * bytes;
        const float d = std::ldexp(1.0F, -15 + static_cast<int>((block + salt) % 3U));
        const float m = type == 7U ? -0.000244140625F : d * 0.5F;
        const auto dh = __float2half(d), mh = __float2half(m);
        std::memcpy(out, &dh, 2); std::memcpy(out + 2, &mh, 2);
        for (unsigned int group = 0; group < elements / 32U; ++group) {
            const auto scale = static_cast<unsigned int>((block + salt + group * 17U) % 64U);
            const auto minimum = static_cast<unsigned int>((block * 3U + salt + group * 29U) % 64U);
            if (type != 7U) {
                // Encode each independently generated six-bit factor by its
                // specified field location, including the split upper bits.
                for (unsigned int bit = 0; bit < 6U; ++bit) {
                    const auto scale_bit = group < 4U ? group * 8U + bit :
                        (bit < 4U ? (group + 4U) * 8U + bit : (group - 4U) * 8U + bit + 2U);
                    const auto min_bit = group < 4U ? (group + 4U) * 8U + bit :
                        (bit < 4U ? (group + 4U) * 8U + bit + 4U : group * 8U + bit + 2U);
                    set_bit(out + 4, scale_bit, scale >> bit);
                    set_bit(out + 4, min_bit, minimum >> bit);
                }
            }
            for (unsigned int lane = 0; lane < 32U; ++lane) {
                const auto code = static_cast<unsigned int>((block * 7U + salt + group * 3U + lane) %
                                                           (type == 12U ? 16U : 32U));
                const auto element = group * 32U + lane;
                result.values[block * elements + element] = type == 7U ?
                    d * static_cast<float>(code) + m :
                    d * static_cast<float>(scale) * static_cast<float>(code) - m * static_cast<float>(minimum);
                const auto q_offset = type == 7U ? 8U : (type == 12U ? 16U : 48U);
                const auto nibble_bit = type == 7U ? (lane % 16U) * 8U + (lane / 16U) * 4U :
                    ((group / 2U) * 32U + lane) * 8U + (group % 2U) * 4U;
                for (unsigned int bit = 0; bit < 4U; ++bit)
                    set_bit(out + q_offset, nibble_bit + bit, code >> bit);
                if (type == 7U) set_bit(out + 4, lane, code >> 4U);
                if (type == 13U) set_bit(out + 16, lane * 8U + group, code >> 4U);
            }
        }
    }
    return result;
}

float weight_value(const TestWeights& weights, std::uint32_t expert,
                   std::uint32_t output, std::uint32_t output_width,
                   std::uint32_t input_width, std::uint32_t feature) {
    return weights.values[(static_cast<std::size_t>(expert) * output_width + output) * input_width + feature];
}

float silu(float value) {
    return value /
           (1.0F +
            std::exp(-value));
}

float round_bf16(float value) {
    if (value == 0.0F) return value;
    int exponent = 0;
    const double scaled = std::frexp(std::fabs(static_cast<double>(value)), &exponent) * 256.0;
    auto integer = static_cast<unsigned int>(std::floor(scaled));
    const auto fraction = scaled - integer;
    if (fraction > 0.5 || (fraction == 0.5 && (integer & 1U))) ++integer;
    return static_cast<float>(std::copysign(std::ldexp(static_cast<double>(integer), exponent - 8), value));
}

}  // namespace

int main(int argc, char** argv) try {
    const unsigned int requested_mode = argc > 1 ? static_cast<unsigned int>(std::stoul(argv[1])) : 0U;
    if (requested_mode > 7U) throw std::invalid_argument("invalid K-quant test mode");
    const bool slot_parallel_q51=argc>2 && (std::string(argv[2])=="q4k-a8-q51-slot-parallel" ||
        std::string(argv[2])=="q4k-a8-q51-slot-parallel-duplicate");
    const bool grouped_q51=slot_parallel_q51 || (argc>2 && (std::string(argv[2])=="q4k-a8-q51-grouped" || std::string(argv[2])=="q4k-a8-q51-grouped-duplicate"));
    const bool q4k_a8 = grouped_q51 || (argc > 2 && (std::string(argv[2]) == "q4k-a8" || std::string(argv[2]) == "q4k-a8-duplicate"));
    const bool duplicate_routes = argc > 2 && (std::string(argv[2]) == "q4k-a8-duplicate" || std::string(argv[2])=="q4k-a8-q51-grouped-duplicate" ||
        std::string(argv[2])=="q4k-a8-q51-slot-parallel-duplicate");
#ifndef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    if(q4k_a8) throw std::invalid_argument("Q4K A8 test requires runtime capability");
#endif
    const bool bf16 = requested_mode >= 4U;
    const auto mode = requested_mode % 4U;
    constexpr std::array<std::uint32_t, 3> types{12U, 13U, 7U};
    const auto gate_type = mode < 3U ? types[mode] : 12U;
    const auto up_type = mode < 3U ? types[mode] : 13U;
    const auto down_type = grouped_q51 && mode==0U ? 7U : (mode < 3U ? types[mode] : 7U);
    if(q4k_a8 && mode>1U) throw std::invalid_argument("Q4K A8 fixture requires paired Q4K/Q5K");
    constexpr std::uint32_t hidden = 768;
    constexpr std::uint32_t intermediate = 512;
    constexpr std::uint32_t experts = 4;
    constexpr std::uint32_t top_k = 2;

    const auto model =
        strix::ModelContract::qwen38_flash_next();
    {
        strix::HipMoeExpertsLauncherConfig golden;
        golden.hidden_size=2560; golden.intermediate_size=640; golden.expert_count=512; golden.top_k=10;
        strix::ExecutionBindingRequest request;
        request.minimum_qualification=strix::kernels::QualificationStatus::experimental;
        strix::kernels::KernelRegistry registry;
        strix::register_hip_moe_experts_kernel_variant(model,request,golden,registry);
        const std::string id="hip.moe_experts.bf16.h2560.m640.e512.k10.r1.c1";
        expect(registry.find(id)!=nullptr,"default MoE kernel ID matches pre-edit golden");
        const auto launcher_golden=strix::make_hip_moe_experts_launcher(id,&golden);
        expect(launcher_golden.implementation_identity==5740034692284928882ULL,
            "default MoE implementation identity matches pre-edit FNV golden");
        if(slot_parallel_q51) {
            golden.q4k_a8g32_gate_up=true; golden.grouped_exact_q51_down=true; golden.q51_slot_parallel_down=true;
            request.q4k_a8g32_gate_up=true; request.grouped_exact_q51_down=true; request.q51_slot_parallel_down=true;
            strix::register_hip_moe_experts_kernel_variant(model,request,golden,registry);
            const auto candidate_id=id+".q51-slot-parallel-v1"+std::string(strix::moe_q4k_a8::grouped_q51_kernel_ending);
            const auto* variant=registry.find(candidate_id);
            expect(variant&&!variant->graph_safe&&variant->resources.scratch_bytes==
                strix::moe_q4k_a8::build_runtime_workspace(2560,640,512,10,true).total_bytes,
                "registered slot variant has new ID, eager contract and flag-aware scratch resources");
            auto candidate=strix::make_hip_moe_experts_launcher(candidate_id,&golden);
            expect(candidate.implementation_identity!=launcher_golden.implementation_identity,
                "slot candidate has distinct implementation identity");
            const auto reject=[&](auto&& operation,const char* gate) {
                bool rejected=false; try {operation();}
                catch(const std::invalid_argument& error) {rejected=std::string(error.what()).find(gate)!=std::string::npos;}
                expect(rejected,"slot admission reaches its intended metadata gate");
            };
            auto wrong_config=golden; wrong_config.q51_slot_parallel_down=false;
            reject([&] {static_cast<void>(strix::make_hip_moe_experts_launcher(candidate_id,&wrong_config));},"slot-parallel Q5_1 id");
            reject([&] {static_cast<void>(strix::make_hip_moe_experts_launcher(id+std::string(strix::moe_q4k_a8::grouped_q51_kernel_ending),&golden));},"slot-parallel Q5_1 id");
            reject([&] {static_cast<void>(strix::make_hip_moe_experts_launcher(id+std::string(strix::moe_q4k_a8::grouped_q51_kernel_ending)+".q51-slot-parallel-v1",&golden));},"marker placement");
            for(bool require_safe:{false,true}) {
                auto graph_request=request; graph_request.device_graph_inputs=!require_safe; graph_request.require_graph_safe=require_safe;
                strix::kernels::KernelRegistry graph_registry;
                reject([&] {strix::register_hip_moe_experts_kernel_variant(model,graph_request,golden,graph_registry);},"slot-parallel Q5_1 variant admits eager");
            }
        }
    }

    const auto plan =
        strix::build_serial_decode_execution_plan(
            model);
    const auto io =
        strix::build_serial_execution_io_plan(
            plan);
    const auto& step =
        first_expert_step(plan);
    const auto& step_io =
        io.steps[step.id];

    strix::ExecutionBufferRequest buffer_request;
    buffer_request.logical_rows = 1;
    buffer_request.q4k_a8g32_gate_up = q4k_a8;
    buffer_request.grouped_exact_q51_down=grouped_q51;
    buffer_request.q51_slot_parallel_down=slot_parallel_q51;
    buffer_request.activation_element_bytes = bf16 ? 2U : 4U;
    buffer_request.mlp_intermediate_element_bytes = 4;
    const auto arena =
        strix::build_serial_execution_arena_plan(
            model,
            buffer_request,
            plan,
            io,
            256);

    void* arena_pointer = nullptr;
    check(
        hipMalloc(
            &arena_pointer,
            static_cast<std::size_t>(
                arena.total_bytes)),
        "hipMalloc routed-expert arena");
    if(slot_parallel_q51)
        check(hipMemset(arena_pointer,0x5a,static_cast<std::size_t>(arena.total_bytes)),"initialize complete slot-parallel fixture arena");

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity =
        0x4d4f455445535401ULL;
    memory.arena_base =
        reinterpret_cast<std::uintptr_t>(
            arena_pointer);
    memory.arena_bytes =
        arena.total_bytes;
    memory.allocation_generation = 1;

    const auto hidden_input =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.reads[0]);
    const auto route_indices =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.reads[1]);
    const auto route_weights =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.reads[2]);
    const auto output =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.writes[0]);

    std::vector<float> hidden_values(
        hidden);
    for (std::size_t index = 0;
         index < hidden_values.size();
         ++index) {
        hidden_values[index] =
            static_cast<float>(
                static_cast<int>(
                    index % 19U) -
                9) /
            23.0F;
    }

    const std::vector<std::uint32_t>
        selected{1U, duplicate_routes ? 1U : 3U};
    const std::vector<float>
        selected_weights{0.7F, 0.3F};

    std::vector<std::uint16_t> hidden_bf16(hidden);
    if (bf16) {
        for (std::size_t index = 0; index < hidden; ++index) {
            hidden_values[index] = round_bf16(hidden_values[index]);
            hidden_bf16[index] = static_cast<std::uint16_t>(
                std::bit_cast<std::uint32_t>(hidden_values[index]) >> 16U);
        }
    }

    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                hidden_input.address),
            bf16 ? static_cast<const void*>(hidden_bf16.data()) : hidden_values.data(),
            hidden_values.size() * (bf16 ? sizeof(std::uint16_t) : sizeof(float)),
            hipMemcpyHostToDevice),
        "copy routed-expert hidden");
    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                route_indices.address),
            selected.data(),
            selected.size() *
                sizeof(std::uint32_t),
            hipMemcpyHostToDevice),
        "copy routed-expert indices");
    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                route_weights.address),
            selected_weights.data(),
            selected_weights.size() *
                sizeof(float),
            hipMemcpyHostToDevice),
        "copy routed-expert weights");

    const auto gate =
        make_weights(
            experts,
            intermediate,
            hidden,
            3U, gate_type);
    const auto up =
        make_weights(
            experts,
            intermediate,
            hidden,
            7U, up_type);
    const auto down =
        make_weights(
            experts,
            hidden,
            intermediate,
            11U, down_type);

    std::vector<float> expected_projections(2ULL*intermediate*top_k);
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    if(q4k_a8 && gate_type==12U) {
        const auto activation=q4k_a8_oracle::quantize(hidden_values);
        const auto row_bytes=static_cast<std::size_t>(hidden/256U)*144U;
        for(unsigned slot=0;slot<top_k;++slot) for(unsigned which=0;which<2;++which)
            for(unsigned row=0;row<intermediate;++row) {
                const auto& packed=which ? up.packed : gate.packed;
                const auto offset=(static_cast<std::size_t>(selected[slot])*intermediate+row)*row_bytes;
                expected_projections[(slot*2U+which)*intermediate+row]=q4k_a8_oracle::row(
                    std::span<const std::byte>(packed.data()+offset,row_bytes),activation);
            }
    }
#endif
    std::vector<float> expected(
        hidden,
        0.0F);
    for (std::uint32_t slot = 0;
         slot < top_k;
         ++slot) {
        const auto expert =
            selected[slot];
        std::vector<float> temp(
            intermediate);

        for (std::uint32_t inner = 0;
             inner < intermediate;
             ++inner) {
            double gate_sum = 0.0;
            double up_sum = 0.0;
            for (std::uint32_t feature = 0;
                 feature < hidden;
                 ++feature) {
                gate_sum +=
                    static_cast<double>(
                        hidden_values[
                            feature]) *
                    weight_value(
                        gate,
                        expert,
                        inner,
                        intermediate,
                        hidden,
                        feature);
                up_sum +=
                    static_cast<double>(
                        hidden_values[
                            feature]) *
                    weight_value(
                        up,
                        expert,
                        inner,
                        intermediate,
                        hidden,
                        feature);
            }
            if(q4k_a8 && gate_type==12U) {
                gate_sum=expected_projections[slot*2U*intermediate+inner];
                up_sum=expected_projections[(slot*2U+1U)*intermediate+inner];
            }
            temp[inner] =
                silu(
                    static_cast<float>(
                        gate_sum)) *
                static_cast<float>(
                    up_sum);
        }

        for (std::uint32_t feature = 0;
             feature < hidden;
             ++feature) {
            double value = 0.0;
            for (std::uint32_t inner = 0;
                 inner < intermediate;
                 ++inner) {
                value +=
                    static_cast<double>(
                        temp[inner]) *
                    weight_value(
                        down,
                        expert,
                        feature,
                        hidden,
                        intermediate,
                        inner);
            }
            expected[feature] +=
                static_cast<float>(
                    value) *
                selected_weights[slot];
        }
    }

    if (bf16) for (auto& value : expected) value = round_bf16(value);

    strix::ModelManifest manifest;
    manifest.architecture =
        "qwen4_exp";
    auto add_tensor =
        [&](std::string name,
            std::vector<std::uint64_t> dims,
            std::uint64_t bytes, std::uint32_t type) {
            strix::ManifestTensor tensor;
            tensor.name =
                std::move(name);
            tensor.ggml_type = type;
            tensor.dimensions =
                std::move(dims);
            tensor.byte_size =
                bytes;
            tensor.role =
                strix::TensorRole::
                    moe_expert;
            tensor.layer =
                step.layer;
            manifest.total_tensor_bytes +=
                bytes;
            manifest.tensors.push_back(
                std::move(tensor));
        };

    add_tensor(
        "blk.0.ffn_gate_exps.weight",
        {hidden, intermediate, experts},
        gate.size(), gate.type);
    add_tensor(
        "blk.0.ffn_up_exps.weight",
        {hidden, intermediate, experts},
        up.size(), up.type);
    add_tensor(
        "blk.0.ffn_down_exps.weight",
        {intermediate, hidden, experts},
        down.size(), down.type);

    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    std::vector<void*> device_weights(
        3,
        nullptr);
    const auto upload =
        [&](std::size_t index,
            const void* source,
            std::size_t bytes) {
            check(
                hipMalloc(
                    &device_weights[index],
                    bytes),
                "hipMalloc routed-expert weights");
            check(
                hipMemcpy(
                    device_weights[index],
                    source,
                    bytes,
                    hipMemcpyHostToDevice),
                "copy routed-expert weights");
        };
    upload(
        0,
        gate.data(),
        gate.size());
    upload(
        1,
        up.data(),
        up.size());
    upload(
        2,
        down.data(),
        down.size());

    strix::ModelRuntimeBinding runtime;
    runtime.manifest_layout_identity =
        manifest.layout_identity;
    runtime.model_content_identity.bytes[0] =
        1;
    for (std::size_t index = 0;
         index < 3;
         ++index) {
        runtime.tensors.push_back(
            {
                static_cast<std::uint32_t>(
                    index),
                strix::RuntimeTensorStorage::
                    resident,
                reinterpret_cast<std::uintptr_t>(
                    device_weights[index]),
                manifest.tensors[index].
                    byte_size,
                1,
                0,
            });
    }

    strix::ExecutionTensorAccessPlan access;
    access.manifest_layout_identity =
        manifest.layout_identity;
    access.plan_identity =
        0x4d4f455445535402ULL;
    access.accesses.push_back(
        {
            step.id,
            step.op,
            0,
            {0, 1, 2},
        });

    strix::ExecutionRuntimeResources resources;
    resources.scratch =
        &memory;
    resources.model_manifest =
        &manifest;
    resources.model_weights =
        &runtime;
    resources.tensor_access =
        &access;

    strix::ExecutionDispatchContext context;
    context.step_id =
        step.id;
    context.op =
        step.op;
    context.layer =
        step.layer;
    context.reads =
        step_io.reads;
    context.writes =
        step_io.writes;
    context.temporaries =
        step_io.temporaries;
    context.arena =
        &arena;
    context.memory =
        &memory;
    context.resources =
        &resources;
    context.logical_rows = 1;
    context.request_concurrency = 1;

    strix::HipMoeExpertsLauncherConfig config;
    config.hidden_size =
        hidden;
    config.intermediate_size =
        intermediate;
    config.expert_count =
        experts;
    config.top_k =
        top_k;
    config.activation = bf16 ? strix::HipActivationStorage::bf16 : strix::HipActivationStorage::fp32;
    config.q4k_a8g32_gate_up = q4k_a8;
    config.grouped_exact_q51_down=grouped_q51;
    if(grouped_q51) config.q4k_a8_stats=std::make_shared<strix::moe_q4k_a8::RuntimeDispatchStats>();

    auto launcher =
        strix::make_hip_moe_experts_launcher(
            "test.moe_experts" + (grouped_q51 ? std::string(".grouped-exact-q51-down-v1") : "") +
                (q4k_a8 ? std::string(strix::moe_q4k_a8::runtime_kernel_suffix) : ""),
            &config);
    launcher.function(
        context,
        launcher.user_data);
    check(
        hipDeviceSynchronize(),
        "routed-expert synchronize");

    std::vector<float> actual(
        hidden);
    std::vector<std::uint16_t> actual_bf16(hidden);
    check(
        hipMemcpy(
            bf16 ? static_cast<void*>(actual_bf16.data()) : actual.data(),
            reinterpret_cast<const void*>(
                output.address),
            actual.size() * (bf16 ? sizeof(std::uint16_t) : sizeof(float)),
            hipMemcpyDeviceToHost),
        "copy routed-expert output");
    if (bf16) {
        for (std::size_t index = 0; index < hidden; ++index)
            actual[index] = std::bit_cast<float>(static_cast<std::uint32_t>(actual_bf16[index]) << 16U);
    }

    float maximum_error = 0.0F;
    for (std::size_t index = 0;
         index < actual.size();
         ++index) {
        maximum_error =
            std::max(
                maximum_error,
                std::abs(
                    actual[index] -
                    expected[index]));
        if (bf16) expect(std::abs(actual[index] - expected[index]) <=
                         1e-6F + std::abs(expected[index]) / 128.0F,
                         "BF16 routed output within one storage rounding step");
    }
    expect(
        bf16 || maximum_error <= 1e-5F,
        "K-quant routed experts match independently generated values");

    const auto output_bytes = hidden * (bf16 ? sizeof(std::uint16_t) : sizeof(float));
    std::vector<std::byte> before(output_bytes), after(output_bytes);
    check(hipMemcpy(before.data(), reinterpret_cast<void*>(output.address), output_bytes,
                    hipMemcpyDeviceToHost), "snapshot committed routed output");
    if(slot_parallel_q51) {
        strix::HipSetupQuiescenceGuard guard;
        const auto temporary=strix::resolve_execution_buffer(memory.execution_identity,arena,memory,step_io.temporaries[0]);
        const auto layout=strix::moe_q4k_a8::build_runtime_workspace(hidden,intermediate,experts,top_k,true);
        expect(layout.total_bytes<=temporary.bytes,"production candidate has flag-aware temporary capacity");
        std::vector<std::byte> initial_arena(static_cast<std::size_t>(arena.total_bytes));
        check(hipMemcpy(initial_arena.data(),arena_pointer,initial_arena.size(),hipMemcpyDeviceToHost),"retain complete fixture arena seed");
        auto candidate_config=config;
        candidate_config.q51_slot_parallel_down=true;
        candidate_config.q4k_a8_stats=std::make_shared<strix::moe_q4k_a8::RuntimeDispatchStats>();
        auto candidate=strix::make_hip_moe_experts_launcher("test.q51-slot-parallel-v1"+
            std::string(strix::moe_q4k_a8::grouped_q51_kernel_ending),&candidate_config);
        const auto raw_begin=static_cast<std::size_t>(temporary.address-memory.arena_base+layout.rawdots_offset);
        const auto raw_end=raw_begin+static_cast<std::size_t>(layout.rawdots_bytes);
        const auto raw_poison=[](const std::vector<std::byte>& bytes,std::size_t begin,std::size_t end) {
            return std::all_of(bytes.begin()+static_cast<std::ptrdiff_t>(begin),bytes.begin()+static_cast<std::ptrdiff_t>(end),
                [](auto value) {return value==std::byte{0x5a};});
        };
        struct Result { std::vector<std::byte> arena; unsigned status=0; bool rejected=false; };
        const auto execute=[&](bool parallel) {
            Result result; result.arena.resize(initial_arena.size());
            guard.run([&] {
                check(hipMemcpy(arena_pointer,initial_arena.data(),initial_arena.size(),hipMemcpyHostToDevice),"restore identical complete arena before production call");
                auto& selected_launcher=parallel?candidate:launcher;
                try { selected_launcher.function(context,selected_launcher.user_data); }
                catch(const std::invalid_argument&) {result.rejected=true;}
                check(hipMemcpy(result.arena.data(),arena_pointer,result.arena.size(),hipMemcpyDeviceToHost),"read complete production arena");
                if(gate_type==12U)
                    check(hipMemcpy(&result.status,reinterpret_cast<void*>(temporary.address+layout.status_offset),4,hipMemcpyDeviceToHost),"read precise production local status");
                else
                    check(hipMemcpy(&result.status,reinterpret_cast<void*>(temporary.address+(2ULL*intermediate+hidden)*4U),4,hipMemcpyDeviceToHost),"read precise fallback local status");
            });
            return result;
        };
        const auto pair=[&](unsigned expected_status,bool raw_untouched) {
            const auto reference=execute(false),actual_candidate=execute(true);
            expect(reference.rejected==(expected_status!=0U)&&actual_candidate.rejected==reference.rejected&&
                reference.status==expected_status&&actual_candidate.status==expected_status,
                "production original/candidate retain exact rejection and status bits");
            expect(std::equal(reference.arena.begin(),reference.arena.begin()+static_cast<std::ptrdiff_t>(raw_begin),actual_candidate.arena.begin())&&
                std::equal(reference.arena.begin()+static_cast<std::ptrdiff_t>(raw_end),reference.arena.end(),
                    actual_candidate.arena.begin()+static_cast<std::ptrdiff_t>(raw_end)),
                "production candidate preserves every arena byte outside dedicated raw dots");
            if(expected_status!=0U) {
                const auto out_begin=static_cast<std::size_t>(output.address-memory.arena_base);
                expect(std::equal(initial_arena.begin()+static_cast<std::ptrdiff_t>(out_begin),
                    initial_arena.begin()+static_cast<std::ptrdiff_t>(out_begin+output_bytes),
                    actual_candidate.arena.begin()+static_cast<std::ptrdiff_t>(out_begin)),
                    "production rejection preserves complete semantic output");
            }
            if(raw_untouched) expect(raw_poison(actual_candidate.arena,raw_begin,raw_end),"rejected/fallback production call changed raw dots");
            return reference;
        };
        for(unsigned repeat=0;repeat<3U;++repeat) static_cast<void>(pair(0U,gate_type!=12U));
        expect(candidate_config.q4k_a8_stats->slot_parallel_q51_submissions.load()==(gate_type==12U?3U:0U),
            "production candidate counter proves actual selected branch and zero fallback submissions");
        auto bad_arena=initial_arena;
        const auto route_offset=static_cast<std::size_t>(route_indices.address-memory.arena_base);
        const auto mix_offset=static_cast<std::size_t>(route_weights.address-memory.arena_base);
        const auto input_offset=static_cast<std::size_t>(hidden_input.address-memory.arena_base);
        const auto restore_arena=initial_arena;
        const unsigned invalid_route=experts;
        std::memcpy(initial_arena.data()+route_offset+4U,&invalid_route,4U); static_cast<void>(pair(1U,true));
        initial_arena=restore_arena;
        const unsigned nan_weight=0x7fc12345U;
        std::memcpy(initial_arena.data()+mix_offset,&nan_weight,4U); static_cast<void>(pair(1U,true));
        initial_arena=restore_arena;
        if(gate_type==12U) {
            if(bf16) { const std::uint16_t nan=0x7fc1U; std::memcpy(initial_arena.data()+input_offset,&nan,2U); }
            else { const unsigned nan=0x7fc12345U; std::memcpy(initial_arena.data()+input_offset,&nan,4U); }
            static_cast<void>(pair(2U,true)); initial_arena=restore_arena;
            // Corrupt the selected down coefficient after finite gate/up and SiLU.
            auto corrupt=down.packed; const std::uint16_t nan=0x7e11U;
            const auto row=static_cast<std::size_t>(selected.front())*hidden*(intermediate/32U)*24U;
            std::memcpy(corrupt.data()+row+2U,&nan,2U);
            check(hipMemcpy(device_weights[2],corrupt.data(),corrupt.size(),hipMemcpyHostToDevice),"upload down-only nonfinite coefficient");
            static_cast<void>(pair(8U,false));
            check(hipMemcpy(device_weights[2],down.data(),down.size(),hipMemcpyHostToDevice),"restore finite down coefficient");
            // Exact BF16 input 2^26 calibrates only a finite route weight. This
            // reaches the actual FP32 finite/BF16-overflow boundary in both paths.
            const float large=67108864.0F; const std::uint16_t large_bf16=0x4c80U;
            for(unsigned index=0;index<hidden;++index)
                std::memcpy(initial_arena.data()+input_offset+index*(bf16?2U:4U),bf16?static_cast<const void*>(&large_bf16):&large,bf16?2U:4U);
            std::array<float,2> calibration{1.0F,0.0F};
            std::memcpy(initial_arena.data()+mix_offset,calibration.data(),sizeof(calibration));
            const auto finite=pair(0U,false);
            float maximum=0.0F;
            const auto accumulator_offset=static_cast<std::size_t>(temporary.address-memory.arena_base+layout.accumulation_offset);
            for(unsigned index=0;index<hidden;++index) { float value=0.0F;
                std::memcpy(&value,finite.arena.data()+accumulator_offset+index*4U,4U);
                expect(std::isfinite(value),"slot threshold calibration is finite"); maximum=std::max(maximum,std::abs(value)); }
            if(!(maximum>1.0F)) throw std::runtime_error("slot threshold calibration has no finite scale");
            calibration[0]=std::bit_cast<float>(0x7f7fc000U)/maximum;
            expect(std::isfinite(calibration[0]),"slot threshold route weight is finite");
            std::memcpy(initial_arena.data()+mix_offset,calibration.data(),sizeof(calibration));
            const auto band=pair(bf16?8U:0U,false);
            bool reached=false;
            for(unsigned index=0;index<hidden;++index) { unsigned bits=0;
                std::memcpy(&bits,band.arena.data()+accumulator_offset+index*4U,4U); bits&=0x7fffffffU;
                expect(bits<0x7f800000U,"slot threshold accumulator remains finite FP32"); reached|=bits>=0x7f7f8000U; }
            expect(reached,"slot threshold fixture reaches actual BF16 rounding-overflow band");
            initial_arena=restore_arena;
        }
        // Capture rejection must precede even the accumulation/status clear.
        hipStream_t capture_stream=nullptr; hipGraph_t rejected_graph=nullptr;
        check(hipMemcpy(arena_pointer,initial_arena.data(),initial_arena.size(),hipMemcpyHostToDevice),"restore pre-capture complete arena");
        check(hipStreamCreateWithFlags(&capture_stream,hipStreamNonBlocking),"create production admission stream");
        try {
            check(hipStreamBeginCapture(capture_stream,hipStreamCaptureModeThreadLocal),"begin production rejection capture");
            bool rejected=false; strix::HipLaunchContext launch_context; launch_context.stream=capture_stream;
            { strix::HipLaunchScope scope(launch_context);
                try {candidate.function(context,candidate.user_data);}
                catch(const std::invalid_argument& error) {rejected=std::string(error.what()).find("eager execution only")!=std::string::npos;} }
            check(hipStreamEndCapture(capture_stream,&rejected_graph),"end production rejection capture");
            std::size_t nodes=0; check(hipGraphGetNodes(rejected_graph,nullptr,&nodes),"inspect rejected production graph");
            check(hipMemcpy(bad_arena.data(),arena_pointer,bad_arena.size(),hipMemcpyDeviceToHost),"read untouched production rejection arena");
            expect(rejected&&nodes==0U&&bad_arena==initial_arena,"production capture rejects before any clear, kernel or byte mutation");
            check(hipGraphDestroy(rejected_graph),"destroy production rejection graph"); rejected_graph=nullptr;
            check(hipStreamDestroy(capture_stream),"destroy production admission stream"); capture_stream=nullptr;
        } catch(...) {guard.recover_or_fail_stop(); if(rejected_graph)(void)hipGraphDestroy(rejected_graph); if(capture_stream)(void)hipStreamDestroy(capture_stream); throw;}
        for(auto* pointer:device_weights) check(hipFree(pointer),"free slot production weight");
        check(hipFree(arena_pointer),"free slot production arena");
        std::cout<<"Q5_1 slot-parallel production eager comparisons complete; source fixture is not model/performance qualification\n";
        return failures?1:0;
    }
    if(grouped_q51 && gate_type==12U) {
        expect(config.q4k_a8_stats->grouped_q51_submissions.load()==1U,"grouped Q5_1 native host branch actually submitted");
        const auto grouped_temporary=strix::resolve_execution_buffer(memory.execution_identity,arena,memory,step_io.temporaries[0]);
        const auto grouped_layout=strix::moe_q4k_a8::build_runtime_workspace(hidden,intermediate,experts,top_k);
        std::vector<std::uint32_t> grouped_accumulator(hidden),reference_accumulator(hidden);
        check(hipMemcpy(grouped_accumulator.data(),reinterpret_cast<void*>(grouped_temporary.address+grouped_layout.accumulation_offset),hidden*4,hipMemcpyDeviceToHost),"snapshot grouped FP32 accumulation");
        auto reference_config=config; reference_config.grouped_exact_q51_down=false;
        auto reference=strix::make_hip_moe_experts_launcher(
            "test.q4-reference"+std::string(strix::moe_q4k_a8::runtime_kernel_suffix),&reference_config);
        reference.function(context,reference.user_data);
        check(hipMemcpy(after.data(),reinterpret_cast<void*>(output.address),output_bytes,hipMemcpyDeviceToHost),"read A8-only down reference");
        expect(before==after,"grouped exact Q5_1 native output equals A8-only original down bits");
        check(hipMemcpy(reference_accumulator.data(),reinterpret_cast<void*>(grouped_temporary.address+grouped_layout.accumulation_offset),hidden*4,hipMemcpyDeviceToHost),"read original FP32 accumulation");
        expect(grouped_accumulator==reference_accumulator,"grouped exact Q5_1 native FP32 accumulation equals original down bits before BF16 rounding");
    }
    if(q4k_a8 && gate_type==12U) {
        const auto temporary=strix::resolve_execution_buffer(memory.execution_identity,arena,memory,step_io.temporaries[0]);
        const auto layout=strix::moe_q4k_a8::build_runtime_workspace(hidden,intermediate,experts,top_k);
        std::vector<float> staged(expected_projections.size());
        check(hipMemcpy(staged.data(),reinterpret_cast<void*>(temporary.address+layout.gate_up_offset),
            staged.size()*sizeof(float),hipMemcpyDeviceToHost),"read quantized gate/up slot workspace");
        for(unsigned slot=0;slot<top_k;++slot) for(unsigned row=0;row<intermediate;++row) {
            const auto gate_index=slot*2U*intermediate+row;
            const auto up_index=gate_index+intermediate;
            expect(std::bit_cast<unsigned>(staged[up_index])==std::bit_cast<unsigned>(expected_projections[up_index]),
                "untouched up half exactly matches independent integer/FP32 oracle");
            expect(std::abs(staged[gate_index]-silu(expected_projections[gate_index])*expected_projections[up_index])<=1e-6F,
                "activated gate half matches independent slot composition");
        }
    }
    if(q4k_a8 && gate_type==12U) {
        const auto temporary=strix::resolve_execution_buffer(memory.execution_identity,arena,memory,step_io.temporaries[0]);
        std::vector<std::byte> scratch_after(static_cast<std::size_t>(temporary.bytes));
        const auto malformed_binding = [&](unsigned tensor, std::uintptr_t address, std::uint64_t bytes) {
            check(hipMemset(reinterpret_cast<void*>(temporary.address),0x5a,static_cast<std::size_t>(temporary.bytes)),"poison pre-admission scratch");
            const auto saved=runtime.tensors[tensor];
            runtime.tensors[tensor].address=address; runtime.tensors[tensor].bytes=bytes;
            bool reject=false;
            try { launcher.function(context,launcher.user_data); }
            catch(const std::invalid_argument&) { reject=true; }
            runtime.tensors[tensor]=saved;
            check(hipMemcpy(scratch_after.data(),reinterpret_cast<void*>(temporary.address),scratch_after.size(),hipMemcpyDeviceToHost),"read pre-admission scratch");
            check(hipMemcpy(after.data(),reinterpret_cast<void*>(output.address),output_bytes,hipMemcpyDeviceToHost),"read pre-admission destination");
            expect(reject && before==after && std::all_of(scratch_after.begin(),scratch_after.end(),
                [](std::byte v){return v==std::byte{0x5a};}),"malformed binding rejects before clear, scratch mutation or semantic publication");
        };
        malformed_binding(0,temporary.address,runtime.tensors[0].bytes);
        malformed_binding(1,runtime.tensors[0].address,runtime.tensors[1].bytes);
        malformed_binding(2,output.address,runtime.tensors[2].bytes);
        malformed_binding(0,runtime.tensors[0].address,runtime.tensors[0].bytes-1);
    }
    if(q4k_a8 && gate_type==13U) {
        if(grouped_q51) expect(config.q4k_a8_stats->grouped_q51_submissions.load()==0U,"Q5K gate fallback does not submit grouped Q5_1 down");
        auto legacy_config=config; legacy_config.q4k_a8g32_gate_up=false; legacy_config.grouped_exact_q51_down=false;
        auto legacy=strix::make_hip_moe_experts_launcher("test.legacy.moe",&legacy_config);
        legacy.function(context,legacy.user_data);
        check(hipMemcpy(after.data(),reinterpret_cast<void*>(output.address),output_bytes,hipMemcpyDeviceToHost),
            "read Q5K legacy fallback");
        expect(before==after,"Q5K mode fallback output is bitwise identical to original launcher");
    }
    auto bad_routes = selected;
    bad_routes[1] = experts;
    check(hipMemcpy(reinterpret_cast<void*>(route_indices.address), bad_routes.data(),
                    bad_routes.size() * sizeof(std::uint32_t), hipMemcpyHostToDevice), "upload invalid expert ID");
    bool rejected = false;
    try { launcher.function(context, launcher.user_data); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "invalid expert ID rejects eager execution");
    check(hipMemcpy(after.data(), reinterpret_cast<void*>(output.address), output_bytes,
                    hipMemcpyDeviceToHost), "read output after invalid route");
    expect(before == after, "invalid route cannot publish partial expert output");
    check(hipMemcpy(reinterpret_cast<void*>(route_indices.address), selected.data(),
                    selected.size() * sizeof(std::uint32_t), hipMemcpyHostToDevice), "restore valid routes");
    auto bad_weights = selected_weights;
    bad_weights[0] = std::numeric_limits<float>::quiet_NaN();
    check(hipMemcpy(reinterpret_cast<void*>(route_weights.address), bad_weights.data(),
                    bad_weights.size() * sizeof(float), hipMemcpyHostToDevice), "upload nonfinite route weight");
    rejected = false;
    try { launcher.function(context, launcher.user_data); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "nonfinite route weight rejects eager execution");
    check(hipMemcpy(after.data(), reinterpret_cast<void*>(output.address), output_bytes,
                    hipMemcpyDeviceToHost), "read output after nonfinite weight");
    expect(before == after, "nonfinite weight cannot publish partial expert output");

    // The separate eager record retains precise local bits until the token's
    // explicit completion boundary. All D2H storage survives a checked drain.
    const auto check_deferred_error = [&](unsigned expected_bits,bool later_valid=false) {
        strix::HipSetupQuiescenceGuard completion_guard;
        hipStream_t stream=nullptr;void* record=nullptr;
        strix::DeferredMoeFirstError receipt;
        // A bounded host callback blocks only the fixture stream. It never calls
        // HIP. Even an accidental layer drain is released after five seconds.
        struct StreamGate {std::mutex mutex;std::condition_variable changed;
            bool entered=false,released=false,timed_out=false;} gate;
        const auto release_gate=[&]{std::lock_guard lock(gate.mutex);
            gate.released=true;gate.changed.notify_all();};
        try {
            check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"create deferred fixture stream");
            check(hipMalloc(&record,sizeof(receipt)),"allocate deferred fixture record");
            strix::HipLaunchContext scope_context;scope_context.stream=stream;
            scope_context.deferred_moe_status={memory.execution_identity,reinterpret_cast<std::uintptr_t>(record),sizeof(receipt),1};
            auto deferred_config=config;deferred_config.deferred_moe_status=true;
            auto deferred=strix::make_hip_moe_experts_launcher("test.deferred-status-v1"+
                (grouped_q51 ? std::string(".grouped-exact-q51-down-v1") : "")+
                (q4k_a8 ? std::string(strix::moe_q4k_a8::runtime_kernel_suffix) : ""),&deferred_config);
            const auto temporary=strix::resolve_execution_buffer(memory.execution_identity,arena,memory,step_io.temporaries[0]);
            for(unsigned fault=0;fault<3;++fault){
                check(hipMemset(reinterpret_cast<void*>(temporary.address),0x5a,temporary.bytes),"poison deferred preflight scratch");
                std::vector<std::byte> scratch_after(temporary.bytes);
                auto bad_scope=scope_context;
                if(fault==2)bad_scope.deferred_moe_status.arena_base=temporary.address;
                hipGraph_t rejected_graph=nullptr;
                if(fault)check(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"capture deferred preflight rejection");
                bool rejected=false;
                try{
                    if(fault==0)deferred.function(context,deferred.user_data);
                    else{strix::HipLaunchScope scope(bad_scope);
                        auto& candidate=fault==1?launcher:deferred;
                        candidate.function(context,candidate.user_data);}
                }catch(const std::invalid_argument& error){rejected=std::string(error.what())==(fault==2?
                    "deferred MoE record aliases execution storage":
                    "deferred MoE launcher requires matching owned eager completion scope");}
                catch(...){if(fault)(void)hipStreamEndCapture(stream,&rejected_graph);
                    if(rejected_graph)(void)hipGraphDestroy(rejected_graph);throw;}
                if(fault){check(hipStreamEndCapture(stream,&rejected_graph),"finish deferred rejected capture");
                    std::size_t nodes=1;check(hipGraphGetNodes(rejected_graph,nullptr,&nodes),"inspect rejected deferred nodes");
                    expect(nodes==0,"deferred mismatch/alias rejects with zero queued nodes");
                    check(hipGraphDestroy(rejected_graph),"destroy rejected deferred graph");}
                check(hipMemcpy(scratch_after.data(),reinterpret_cast<void*>(temporary.address),scratch_after.size(),hipMemcpyDeviceToHost),"read rejected deferred scratch");
                check(hipMemcpy(after.data(),reinterpret_cast<void*>(output.address),output_bytes,hipMemcpyDeviceToHost),"read rejected deferred output");
                expect(rejected && before==after && std::all_of(scratch_after.begin(),scratch_after.end(),
                    [](std::byte value){return value==std::byte{0x5a};}),"deferred scope/alias admission preserves all scratch and semantic output");
            }
            check(hipMemsetAsync(record,0,sizeof(receipt),stream),"clear deferred fixture record");
            {strix::HipLaunchScope scope(scope_context);deferred.function(context,deferred.user_data);}
            check(hipMemcpyAsync(&receipt,record,sizeof(receipt),hipMemcpyDeviceToHost,stream),"read deferred failure");
            check(hipStreamSynchronize(stream),"drain deferred failure");
            const unsigned domain=q4k_a8&&gate_type==12U ? 2U : 1U;
            expect(receipt.domain==domain&&receipt.bits==expected_bits,"deferred failure preserves domain and complete local bits");
            check(hipMemcpy(after.data(),reinterpret_cast<void*>(output.address),output_bytes,hipMemcpyDeviceToHost),"read deferred rejected destination");
            expect(before==after,"deferred invalid layer preserves semantic output");
            bool deferred_rejected=false;
            try{strix::check_deferred_moe_status(receipt);}catch(const std::invalid_argument&){deferred_rejected=true;}
            expect(deferred_rejected,"deferred completion reports original error class");
            if(later_valid) {
                check(hipMemcpyAsync(reinterpret_cast<void*>(route_indices.address),selected.data(),selected.size()*4,hipMemcpyHostToDevice,stream),"restore deferred valid route");
                check(hipMemcpyAsync(reinterpret_cast<void*>(route_weights.address),selected_weights.data(),selected_weights.size()*4,hipMemcpyHostToDevice,stream),"restore deferred valid weights");
                {strix::HipLaunchScope scope(scope_context);deferred.function(context,deferred.user_data);}
                check(hipMemcpyAsync(&receipt,record,sizeof(receipt),hipMemcpyDeviceToHost,stream),"read retained first error");
                check(hipStreamSynchronize(stream),"drain later valid layer");
                expect(receipt.domain==domain&&receipt.bits==expected_bits,"later local reset and successful layer preserve first error");
                check(hipLaunchHostFunc(stream,[](void* pointer){auto& blocked=*static_cast<StreamGate*>(pointer);
                    std::unique_lock lock(blocked.mutex);blocked.entered=true;blocked.changed.notify_all();
                    if(!blocked.changed.wait_for(lock,std::chrono::seconds(5),[&]{return blocked.released;}))
                        blocked.timed_out=true;
                },&gate),"queue bounded deferred stream gate");
                bool entered=false;
                {std::unique_lock lock(gate.mutex);entered=gate.changed.wait_for(lock,std::chrono::seconds(5),[&]{return gate.entered;});}
                {strix::HipLaunchScope scope(scope_context);deferred.function(context,deferred.user_data);}
                const auto pending=hipStreamQuery(stream);
                bool timed_out=false;{std::lock_guard lock(gate.mutex);timed_out=gate.timed_out;}
                release_gate();
                check(hipStreamSynchronize(stream),"drain released deferred stream gate");
                expect(entered && !timed_out && pending==hipErrorNotReady,
                    "deferred launcher returns while owned stream is blocked, without a layer drain");

            }
            check(hipFree(record),"free deferred fixture record");record=nullptr;
            check(hipStreamDestroy(stream),"destroy deferred fixture stream");stream=nullptr;
        } catch(...) {
            release_gate();
            completion_guard.recover_or_fail_stop();
            if(record)(void)hipFree(record);
            if(stream)(void)hipStreamDestroy(stream);
            throw;
        }
    };
    check_deferred_error(1U,true);
    check(hipMemcpy(reinterpret_cast<void*>(route_indices.address),bad_routes.data(),bad_routes.size()*4,hipMemcpyHostToDevice),"set deferred invalid route");
    check_deferred_error(1U,true);
    check(hipMemcpy(reinterpret_cast<void*>(route_weights.address),bad_weights.data(),bad_weights.size()*4,hipMemcpyHostToDevice),"restore original invalid graph weights");

    // Captured execution must report the same failure through the existing
    // device graph status, without host reads or synchronization during capture.
    const auto graph_layout = strix::graph_input_layout(1);
    void* graph_arguments = nullptr;
    void* graph_status = nullptr;
    hipStream_t graph_stream = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    check(hipMalloc(&graph_arguments, static_cast<std::size_t>(graph_layout.bytes)), "allocate graph arguments");
    check(hipMalloc(&graph_status, sizeof(std::uint32_t)), "allocate graph status");
    check(hipStreamCreateWithFlags(&graph_stream, hipStreamNonBlocking), "create graph stream");
    strix::HipLaunchContext graph_context;
    graph_context.stream = graph_stream;
    graph_context.graph_capture = true;
    graph_context.arguments = strix::bind_graph_inputs(graph_layout,
        {memory.execution_identity, reinterpret_cast<std::uintptr_t>(graph_arguments), graph_layout.bytes, 1});
    graph_context.status = {memory.execution_identity, reinterpret_cast<std::uintptr_t>(graph_status), sizeof(std::uint32_t), 1};
    graph_context.context_tokens = 64;
    graph_context.vocabulary_size = static_cast<std::uint32_t>(model.vocab_size);
    check(hipStreamBeginCapture(graph_stream, hipStreamCaptureModeThreadLocal), "begin routed graph capture");
    { strix::HipLaunchScope scope(graph_context); launcher.function(context, launcher.user_data); }
    check(hipStreamEndCapture(graph_stream, &graph), "end routed graph capture");
    check(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), "instantiate routed graph");
    if(grouped_q51 && gate_type==12U) {
        std::size_t count=0;check(hipGraphGetNodes(graph,nullptr,&count),"count native Q5_1 graph nodes");
        std::vector<hipGraphNode_t> nodes(count);check(hipGraphGetNodes(graph,nodes.data(),&count),"read native Q5_1 graph nodes");
        unsigned grouped_nodes=0,original_down_nodes=0;
        for(const auto node:nodes) {
            hipGraphNodeType type;check(hipGraphNodeGetType(node,&type),"inspect native Q5_1 node type");
            if(type!=hipGraphNodeTypeKernel) continue;
            hipKernelNodeParams params{};check(hipGraphKernelNodeGetParams(node,&params),"inspect native Q5_1 launch");
            if(params.gridDim.x==(hidden+7U)/8U && params.gridDim.y==1U && params.blockDim.x==256U) ++grouped_nodes;
            if(params.gridDim.x==hidden && params.gridDim.y==1U && params.blockDim.x==256U) ++original_down_nodes;
        }
        expect(grouped_nodes==1U && original_down_nodes==0U,"native capture uses one grouped Q5_1 down and no serial original down nodes");
    }

    if(q4k_a8) {
        check(hipMemcpy(reinterpret_cast<void*>(route_indices.address),selected.data(),selected.size()*sizeof(unsigned),hipMemcpyHostToDevice),"restore graph valid routes");
        check(hipMemcpy(reinterpret_cast<void*>(route_weights.address),selected_weights.data(),selected_weights.size()*sizeof(float),hipMemcpyHostToDevice),"restore graph valid weights");
        for(unsigned replay=0;replay<3;++replay) {
            const unsigned preset=0x200U;
            check(hipMemcpyAsync(graph_status,&preset,4,hipMemcpyHostToDevice,graph_stream),"preset graph aggregate");
            check(hipMemsetAsync(reinterpret_cast<void*>(output.address),0xa5,output_bytes,graph_stream),"stream ordered poison before successful replay");
            check(hipGraphLaunch(graph_exec,graph_stream),"launch valid quantized/fallback graph");
            unsigned status=0;
            check(hipMemcpyAsync(&status,graph_status,4,hipMemcpyDeviceToHost,graph_stream),"read preserved aggregate");
            check(hipMemcpyAsync(after.data(),reinterpret_cast<void*>(output.address),output_bytes,hipMemcpyDeviceToHost,graph_stream),"read successful replay");
            check(hipStreamSynchronize(graph_stream),"drain successful replay");
            expect(status==preset && before==after,"graph replay preserves preset aggregate and reproduces eager bits");
        }
    }
    for (unsigned int invalid = 0; invalid < 2U; ++invalid) {
        check(hipMemcpy(reinterpret_cast<void*>(route_indices.address),
            invalid == 0U ? selected.data() : bad_routes.data(), selected.size() * sizeof(std::uint32_t),
            hipMemcpyHostToDevice), "set graph routes");
        check(hipMemcpy(reinterpret_cast<void*>(route_weights.address),
            invalid == 0U ? bad_weights.data() : selected_weights.data(), selected_weights.size() * sizeof(float),
            hipMemcpyHostToDevice), "set graph route weights");
        const unsigned preset=q4k_a8 ? 0x200U : 0U;
        check(hipMemcpyAsync(graph_status,&preset,sizeof(preset),hipMemcpyHostToDevice,graph_stream), "preset aggregate graph status");
        check(hipGraphLaunch(graph_exec, graph_stream), "launch invalid routed graph");
        std::uint32_t failure = 0;
        check(hipMemcpyAsync(&failure, graph_status, sizeof(failure), hipMemcpyDeviceToHost, graph_stream), "read graph validation status");
        check(hipStreamSynchronize(graph_stream), "drain invalid routed graph");
        expect((failure & 16U) != 0U, "graph reports invalid route or weight at completion");
        if(q4k_a8) expect(failure==0x210U,"invalid graph preserves unrelated aggregate bit");
        check(hipMemcpy(after.data(), reinterpret_cast<void*>(output.address), output_bytes,
                        hipMemcpyDeviceToHost), "read output after invalid graph replay");
        expect(before == after, "invalid graph routes cannot publish expert output");
    }
    if(q4k_a8 && gate_type==12U) {
        const auto temporary=strix::resolve_execution_buffer(memory.execution_identity,arena,memory,step_io.temporaries[0]);
        const auto layout=strix::moe_q4k_a8::build_runtime_workspace(hidden,intermediate,experts,top_k);
        const auto restore = [&] {
            check(hipMemcpy(reinterpret_cast<void*>(hidden_input.address),bf16 ? static_cast<const void*>(hidden_bf16.data()) : hidden_values.data(),hidden*(bf16?2U:4U),hipMemcpyHostToDevice),"restore quantized input");
            check(hipMemcpy(reinterpret_cast<void*>(device_weights[0]),gate.data(),gate.size(),hipMemcpyHostToDevice),"restore gate payload");
            check(hipMemcpy(reinterpret_cast<void*>(device_weights[2]),down.data(),down.size(),hipMemcpyHostToDevice),"restore down payload");
            check(hipMemcpy(reinterpret_cast<void*>(route_indices.address),selected.data(),selected.size()*4,hipMemcpyHostToDevice),"restore route IDs");
            check(hipMemcpy(reinterpret_cast<void*>(route_weights.address),selected_weights.data(),selected_weights.size()*4,hipMemcpyHostToDevice),"restore route weights");
        };
        const auto upload_input = [&](const std::vector<float>& input) {
            std::vector<std::uint16_t> storage(hidden);
            if(bf16) for(unsigned i=0;i<hidden;++i) storage[i]=static_cast<std::uint16_t>(
                std::bit_cast<unsigned>(std::isfinite(input[i]) ? round_bf16(input[i]) : input[i])>>16U);
            check(hipMemcpy(reinterpret_cast<void*>(hidden_input.address),bf16 ? static_cast<const void*>(storage.data()) : input.data(),hidden*(bf16?2U:4U),hipMemcpyHostToDevice),"upload adversarial input");
        };
        const auto error_case = [&](unsigned bit, bool down_phase) {
            bool error=false;
            try { launcher.function(context,launcher.user_data); }
            catch(const std::invalid_argument&) { error=true; }
            unsigned local=0;
            check(hipMemcpy(&local,reinterpret_cast<void*>(temporary.address+layout.status_offset),4,hipMemcpyDeviceToHost),"read eager pipeline status");
            check(hipMemcpy(after.data(),reinterpret_cast<void*>(output.address),output_bytes,hipMemcpyDeviceToHost),"read rejected pipeline output");
            expect(error && local==bit && before==after,"eager failure has precise sticky bit and preserves semantic destination");
            check_deferred_error(bit);
            if(down_phase) {
                std::vector<float> activated(static_cast<std::size_t>(2U*intermediate*top_k));
                check(hipMemcpy(activated.data(),reinterpret_cast<void*>(temporary.address),activated.size()*4,hipMemcpyDeviceToHost),"read pre-down finite activation");
                expect(std::all_of(activated.begin(),activated.end(),[](float value){return std::isfinite(value);}),
                    "down-overflow adversary reached finite post-SiLU workspace");
            }
            const unsigned preset=0x200U;
            check(hipMemcpyAsync(graph_status,&preset,4,hipMemcpyHostToDevice,graph_stream),"preset pipeline-error aggregate");
            check(hipGraphLaunch(graph_exec,graph_stream),"replay pipeline error");
            unsigned aggregate=0;
            check(hipMemcpyAsync(&aggregate,graph_status,4,hipMemcpyDeviceToHost,graph_stream),"read pipeline error aggregate");
            check(hipMemcpyAsync(&local,reinterpret_cast<void*>(temporary.address+layout.status_offset),4,hipMemcpyDeviceToHost,graph_stream),"read captured pipeline status");
            check(hipMemcpyAsync(after.data(),reinterpret_cast<void*>(output.address),output_bytes,hipMemcpyDeviceToHost,graph_stream),"read captured rejected destination");
            check(hipStreamSynchronize(graph_stream),"drain pipeline error graph");
            expect(aggregate==0x210U && local==bit && before==after,
                "captured failure preserves output, precise local bit and unrelated aggregate");
        };
        for(const auto bits : {0x7fc00000U,0x7f800000U,0xff800000U}) {
            restore(); auto bad=hidden_values; bad[0]=std::bit_cast<float>(bits); upload_input(bad); error_case(2U,false);
        }
        if(!bf16) { restore(); upload_input(std::vector<float>(hidden,std::bit_cast<float>(63U))); error_case(4U,false); }
        restore();
        auto corrupt_gate=gate.packed;
        const auto header=(static_cast<std::size_t>(selected[0])*intermediate)*(hidden/256U)*144U;
        const std::uint16_t nan_half=0x7e00U;
        std::memcpy(corrupt_gate.data()+header,&nan_half,2);
        check(hipMemcpy(device_weights[0],corrupt_gate.data(),corrupt_gate.size(),hipMemcpyHostToDevice),"upload nonfinite gate header");
        error_case(8U,false);
        restore(); upload_input(std::vector<float>(hidden,1e25F)); error_case(8U,false);
        restore(); upload_input(std::vector<float>(hidden,1e18F));
        auto huge_down=down.packed;
        for(std::size_t offset=0;offset<huge_down.size();offset+=(down_type==7U ? 24U : 144U)) {
            const std::uint16_t d=0x7bffU,m=0;
            std::memcpy(huge_down.data()+offset,&d,2); std::memcpy(huge_down.data()+offset+2,&m,2);
        }
        check(hipMemcpy(device_weights[2],huge_down.data(),huge_down.size(),hipMemcpyHostToDevice),"upload finite down-overflow payload");
        error_case(8U,true);
        // Use identical exactly representable BF16/FP32 input and isolate slot0.
        // Calibrate from actual finite FP32 accumulation, then scale only a finite
        // route weight. The largest output targets the middle of the BF16-overflow
        // band without overflowing FP32; no rounded BF16 output is used as a reference.
        restore(); upload_input(std::vector<float>(hidden,67108864.0F));
        std::array<float,2> calibration_weights{1.0F,0.0F};
        check(hipMemcpy(reinterpret_cast<void*>(route_weights.address),calibration_weights.data(),sizeof(calibration_weights),hipMemcpyHostToDevice),"upload first-slot calibration weights");
        launcher.function(context,launcher.user_data);
        std::vector<float> accumulation(hidden);
        check(hipMemcpy(accumulation.data(),reinterpret_cast<void*>(temporary.address+layout.accumulation_offset),hidden*4,hipMemcpyDeviceToHost),"read finite calibration accumulation");
        float maximum=0.0F;
        for(const auto value:accumulation) {
            if(!std::isfinite(value)) throw std::runtime_error("nonfinite rounding-band calibration");
            maximum=std::max(maximum,std::abs(value));
        }
        if(!(maximum>1.0F)) throw std::runtime_error("rounding-band calibration too small");
        calibration_weights[0]=std::bit_cast<float>(0x7f7fc000U)/maximum;
        if(!std::isfinite(calibration_weights[0])) throw std::runtime_error("rounding-band route scale is nonfinite");
        check(hipMemcpy(reinterpret_cast<void*>(route_weights.address),calibration_weights.data(),sizeof(calibration_weights),hipMemcpyHostToDevice),"upload finite rounding-band route scale");
        check(hipMemcpy(reinterpret_cast<void*>(output.address),before.data(),output_bytes,hipMemcpyHostToDevice),"restore semantic destination before rounding-band test");
        if(bf16) error_case(8U,false);
        else {
            launcher.function(context,launcher.user_data);
            std::vector<std::byte> scaled_before(output_bytes);
            check(hipMemcpy(scaled_before.data(),reinterpret_cast<void*>(output.address),output_bytes,hipMemcpyDeviceToHost),"read finite FP32 band commit");
            const unsigned preset=0x200U;
            check(hipMemcpyAsync(graph_status,&preset,4,hipMemcpyHostToDevice,graph_stream),"preset finite FP32 band aggregate");
            check(hipMemsetAsync(reinterpret_cast<void*>(output.address),0xa5,output_bytes,graph_stream),"poison FP32 rounding-band replay output");
            check(hipGraphLaunch(graph_exec,graph_stream),"replay finite FP32 rounding band");
            unsigned aggregate=0,local=0;
            check(hipMemcpyAsync(&aggregate,graph_status,4,hipMemcpyDeviceToHost,graph_stream),"read FP32 band aggregate");
            check(hipMemcpyAsync(&local,reinterpret_cast<void*>(temporary.address+layout.status_offset),4,hipMemcpyDeviceToHost,graph_stream),"read FP32 band local status");
            check(hipMemcpyAsync(after.data(),reinterpret_cast<void*>(output.address),output_bytes,hipMemcpyDeviceToHost,graph_stream),"read FP32 band replay output");
            check(hipStreamSynchronize(graph_stream),"drain FP32 rounding-band graph");
            expect(aggregate==preset && local==0 && scaled_before==after,"FP32 threshold admits finite BF16-overflow band in eager and graph");
        }
        check(hipMemcpy(accumulation.data(),reinterpret_cast<void*>(temporary.address+layout.accumulation_offset),hidden*4,hipMemcpyDeviceToHost),"read actual rounding-band accumulation");
        bool band=false; unsigned maximum_bits=0;
        for(const auto value:accumulation) {
            const auto bits=std::bit_cast<unsigned>(value)&0x7fffffffU;
            expect(bits<0x7f800000U,"rounding-band accumulation remains finite FP32");
            band=band || (bits>=0x7f7f8000U && bits<0x7f800000U);
            maximum_bits=std::max(maximum_bits,bits);
        }
        expect(band,"actual finite accumulator reaches BF16 rounding-overflow band");
        std::vector<float> finite_stages(2ULL*intermediate*top_k);
        check(hipMemcpy(finite_stages.data(),reinterpret_cast<void*>(temporary.address),finite_stages.size()*4,hipMemcpyDeviceToHost),"read finite pre-down rounding-band stages");
        expect(std::all_of(finite_stages.begin(),finite_stages.end(),[](float value){return std::isfinite(value);}),"rounding-band error originates after finite projection and SiLU");
        if(!bf16) {
            std::vector<float> committed(hidden);
            check(hipMemcpy(committed.data(),reinterpret_cast<void*>(output.address),hidden*4,hipMemcpyDeviceToHost),"read FP32 admitted band values");
            for(unsigned i=0;i<hidden;++i) expect(std::bit_cast<unsigned>(committed[i])==std::bit_cast<unsigned>(accumulation[i]),"FP32 band commit preserves every finite accumulator bit");
        }
        std::cout << "rounding-band maxabs-bits=" << maximum_bits << " finite-route-scale-bits="
            << std::bit_cast<unsigned>(calibration_weights[0]) << " bf16-reject=" << bf16 << '\n';
        restore();
        check(hipMemcpy(reinterpret_cast<void*>(output.address),before.data(),output_bytes,hipMemcpyHostToDevice),"restore output after rounding-band fixture");
        std::cout << "Q4K A8 native independent up comparisons=" << top_k*intermediate
            << " successful graph replays=3 input/header/activation/down failure gates passed\n";
    }
    check(hipGraphExecDestroy(graph_exec), "destroy routed graph executable");
    check(hipGraphDestroy(graph), "destroy routed graph");
    check(hipStreamDestroy(graph_stream), "destroy routed graph stream");
    check(hipFree(graph_status), "free graph status");
    check(hipFree(graph_arguments), "free graph arguments");

    for (auto* pointer :
         device_weights) {
        check(
            hipFree(pointer),
            "hipFree routed-expert weight");
    }
    check(
        hipFree(arena_pointer),
        "hipFree routed-expert arena");

    if (failures == 0) {
        std::cout
            << "HIP routed-expert launcher tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP routed-expert launcher tests: "
        << error.what() << '\n';
    return 2;
}
