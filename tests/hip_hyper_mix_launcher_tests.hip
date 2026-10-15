#include "strix/execution_arena.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_hyper_mix_launcher.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <limits>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            hipGetErrorString(error));
    }
}

struct Q8Block {
    __half scale;
    std::int8_t values[32];
};

struct IQ4XSBlock {
    __half d;
    std::uint16_t scales_h;
    std::uint8_t scales_l[4];
    std::uint8_t qs[128];
};

static_assert(sizeof(Q8Block) == 34);
static_assert(sizeof(IQ4XSBlock) == 136);

void set_iq4_scale(
    IQ4XSBlock& block,
    std::uint32_t subblock,
    std::uint8_t value) {
    const auto low =
        static_cast<std::uint8_t>(
            value & 0x0fU);
    const auto high =
        static_cast<std::uint16_t>(
            (value >> 4U) & 0x03U);

    auto& byte =
        block.scales_l[
            subblock / 2U];
    if ((subblock % 2U) == 0U) {
        byte =
            static_cast<std::uint8_t>(
                (byte & 0xf0U) |
                low);
    } else {
        byte =
            static_cast<std::uint8_t>(
                (byte & 0x0fU) |
                static_cast<std::uint8_t>(
                    low << 4U));
    }

    const auto shift =
        2U * subblock;
    block.scales_h =
        static_cast<std::uint16_t>(
            block.scales_h &
            static_cast<std::uint16_t>(
                ~(0x03U << shift)));
    block.scales_h =
        static_cast<std::uint16_t>(
            block.scales_h |
            static_cast<std::uint16_t>(
                high << shift));
}

std::vector<IQ4XSBlock> make_iq4_rows(
    std::uint32_t rows,
    float base_scale) {
    std::vector<IQ4XSBlock> values(rows);
    for (std::uint32_t row = 0;
         row < rows;
         ++row) {
        auto& block = values[row];
        block.d =
            __float2half(
                base_scale *
                static_cast<float>(
                    row + 1U));
        block.scales_h = 0;
        std::fill(
            std::begin(block.scales_l),
            std::end(block.scales_l),
            std::uint8_t{0});

        for (std::uint32_t subblock = 0;
             subblock < 8U;
             ++subblock) {
            set_iq4_scale(
                block,
                subblock,
                33U);
            for (std::uint32_t byte = 0;
                 byte < 16U;
                 ++byte) {
                // code 8 maps to +1; both nibbles are identical.
                block.qs[
                    subblock * 16U +
                    byte] =
                    0x88U;
            }
        }
    }
    return values;
}

std::vector<Q8Block> make_q8_rows(
    std::uint32_t rows,
    float base_scale) {
    std::vector<Q8Block> values(rows);
    for (std::uint32_t row = 0;
         row < rows;
         ++row) {
        values[row].scale =
            __float2half(
                base_scale *
                static_cast<float>(
                    row % 17U + 1U));
        std::fill(
            std::begin(values[row].values),
            std::end(values[row].values),
            std::int8_t{1});
    }
    return values;
}

template <typename T>
void upload(
    const std::vector<T>& values,
    void** device) {
    check(
        hipMalloc(
            device,
            values.size() *
                sizeof(T)),
        "hipMalloc hyper-mix tensor");
    check(
        hipMemcpy(
            *device,
            values.data(),
            values.size() *
                sizeof(T),
            hipMemcpyHostToDevice),
        "copy hyper-mix tensor");
}

const strix::ExecutionStep& first_step(
    const strix::SerialExecutionPlan& plan,
    strix::SerialOp op) {
    for (const auto& step : plan.steps) {
        if (step.op == op) {
            return step;
        }
    }
    throw std::runtime_error(
        "requested hyper-mix step not found");
}

float sigmoid(float value) {
    return
        1.0F /
        (1.0F +
         std::exp(-value));
}

struct ReferenceResult {
    std::vector<float> mixed;
    std::vector<float> injection;
};

ReferenceResult reference(
    const std::vector<float>& input,
    const std::vector<float>& norm,
    const std::vector<IQ4XSBlock>& down,
    const std::vector<Q8Block>& up,
    const std::vector<IQ4XSBlock>& inject,
    std::uint32_t hidden,
    std::uint32_t hc,
    std::uint32_t lowrank,
    float epsilon,
    bool with_injection) {
    const auto expanded =
        hidden * hc;
    std::vector<float> normalized(
        expanded);
    for (std::uint32_t stream = 0;
         stream < hc;
         ++stream) {
        float sum = 0.0F;
        for (std::uint32_t feature = 0;
             feature < hidden;
             ++feature) {
            const auto index =
                stream * hidden +
                feature;
            sum +=
                input[index] *
                input[index];
        }
        const float inv =
            1.0F /
            std::sqrt(
                sum /
                    static_cast<float>(
                        hidden) +
                epsilon);
        for (std::uint32_t feature = 0;
             feature < hidden;
             ++feature) {
            const auto index =
                stream * hidden +
                feature;
            normalized[index] =
                input[index] *
                inv *
                (1.0F +
                 norm[index]);
        }
    }

    std::vector<float> low(
        lowrank);
    for (std::uint32_t out = 0;
         out < lowrank;
         ++out) {
        const float weight =
            __half2float(
                down[out].d);
        float sum = 0.0F;
        for (const auto value :
             normalized) {
            sum +=
                value *
                weight;
        }
        const float scaled =
            sum /
            static_cast<float>(hc);
        low[out] =
            scaled *
            sigmoid(scaled);
    }

    ReferenceResult result;
    result.mixed.resize(hidden);
    for (std::uint32_t feature = 0;
         feature < hidden;
         ++feature) {
        float mixed = 0.0F;
        for (std::uint32_t stream = 0;
             stream < hc;
             ++stream) {
            const auto output_index =
                stream * hidden +
                feature;
            const float weight =
                __half2float(
                    up[output_index].scale);
            float projected = 0.0F;
            for (const auto value :
                 low) {
                projected +=
                    value *
                    weight;
            }
            mixed +=
                sigmoid(projected) *
                normalized[
                    output_index];
        }
        result.mixed[feature] =
            mixed /
            static_cast<float>(hc);
    }

    if (with_injection) {
        result.injection.resize(hc);
        for (std::uint32_t stream = 0;
             stream < hc;
             ++stream) {
            const float weight =
                __half2float(
                    inject[stream].d);
            float sum = 0.0F;
            for (const auto value :
                 normalized) {
                sum +=
                    value *
                    weight;
            }
            result.injection[stream] =
                2.0F *
                sigmoid(
                    sum /
                    static_cast<float>(
                        hc));
        }
    }
    return result;
}

struct Fixture {
    static constexpr std::uint32_t down_type = 23;
    static constexpr std::uint32_t up_type = 8;
    static constexpr std::uint32_t inject_type = 23;
    static constexpr std::uint32_t hidden = 64;
    static constexpr std::uint32_t hc = 4;
    static constexpr std::uint32_t lowrank = 32;
    static constexpr std::uint32_t expanded =
        hidden * hc;

    std::vector<float> input;
    std::vector<float> norm;
    std::vector<IQ4XSBlock> down;
    std::vector<Q8Block> up;
    std::vector<IQ4XSBlock> inject;

    Fixture()
        : input(expanded),
          norm(expanded),
          down(make_iq4_rows(
              lowrank,
              0.00005F)),
          up(make_q8_rows(
              expanded,
              0.00007F)),
          inject(make_iq4_rows(
              hc,
              0.00003F)) {
        for (std::size_t index = 0;
             index < input.size();
             ++index) {
            input[index] =
                static_cast<float>(
                    static_cast<int>(
                        index % 23U) -
                    11) /
                17.0F;
            norm[index] =
                static_cast<float>(
                    static_cast<int>(
                        index % 5U) -
                    2) *
                0.01F;
        }
        norm[0] = -1.0F;
        norm[1] = -1.25F;
        norm[2] = 0.375F;
    }
};

// Dense geometry deliberately crosses both feature-tile and output-tile
// boundaries without quantized block alignment masking the tail conditions.
struct DenseTailFixture {
    static constexpr std::uint32_t hidden = 67;
    static constexpr std::uint32_t hc = 3;
    static constexpr std::uint32_t lowrank = 139;
    static constexpr std::uint32_t expanded = hidden * hc;
    static constexpr std::uint32_t down_type = 0;
    static constexpr std::uint32_t up_type = 0;
    static constexpr std::uint32_t inject_type = 0;
    std::vector<float> input, norm, down, up, inject;
    DenseTailFixture() : input(expanded), norm(expanded), down(expanded * lowrank),
        up(lowrank * expanded), inject(expanded * hc) {
        for (std::size_t index = 0; index < input.size(); ++index) {
            input[index] = static_cast<float>(static_cast<int>(index % 29U) - 14) / 19.0F;
            norm[index] = static_cast<float>(static_cast<int>(index % 13U) - 6) / 23.0F;
        }
        for (auto* weights : {&down, &up, &inject})
            for (std::size_t index = 0; index < weights->size(); ++index)
                (*weights)[index] = static_cast<float>(static_cast<int>((index * 7U + index / 11U) % 31U) - 15) / 997.0F;
    }
};

ReferenceResult reference(const std::vector<float>& input, const std::vector<float>& norm,
    const std::vector<float>& down, const std::vector<float>& up, const std::vector<float>& inject,
    std::uint32_t hidden, std::uint32_t hc, std::uint32_t lowrank, float epsilon, bool with_injection) {
    const auto width = hidden * hc;
    std::vector<float> normalized(width), low(lowrank);
    for (std::uint32_t stream = 0; stream < hc; ++stream) {
        double square_sum = 0.0;
        for (std::uint32_t feature = 0; feature < hidden; ++feature) {
            const double value = input[stream * hidden + feature];
            square_sum += value * value;
        }
        const double inverse = 1.0 / std::sqrt(square_sum / hidden + epsilon);
        for (std::uint32_t feature = 0; feature < hidden; ++feature) {
            const auto index = stream * hidden + feature;
            normalized[index] = static_cast<float>(input[index] * inverse * (1.0 + norm[index]));
        }
    }
    for (std::uint32_t out = 0; out < lowrank; ++out) {
        double sum = 0.0;
        for (std::uint32_t feature = 0; feature < width; ++feature)
            sum += static_cast<double>(normalized[feature]) * down[out * width + feature];
        const float scaled = static_cast<float>(sum / hc);
        low[out] = scaled * sigmoid(scaled);
    }
    ReferenceResult result;
    result.mixed.resize(hidden);
    for (std::uint32_t feature = 0; feature < hidden; ++feature) {
        double mixed = 0.0;
        for (std::uint32_t stream = 0; stream < hc; ++stream) {
            const auto index = stream * hidden + feature;
            double gate = 0.0;
            for (std::uint32_t inner = 0; inner < lowrank; ++inner)
                gate += static_cast<double>(low[inner]) * up[index * lowrank + inner];
            mixed += normalized[index] * sigmoid(static_cast<float>(gate));
        }
        result.mixed[feature] = static_cast<float>(mixed / hc);
    }
    if (with_injection) {
        result.injection.resize(hc);
        for (std::uint32_t stream = 0; stream < hc; ++stream) {
            double sum = 0.0;
            for (std::uint32_t feature = 0; feature < width; ++feature)
                sum += static_cast<double>(normalized[feature]) * inject[stream * width + feature];
            result.injection[stream] = 2.0F * sigmoid(static_cast<float>(sum / hc));
        }
    }
    return result;
}

template <std::uint32_t Width, std::uint32_t Streams = 4U, std::uint32_t Lowrank = 320U,
          bool DenseUp = false, bool PrefetchValues = false>
struct TypedProjectionFixture {
    static constexpr std::uint32_t hidden = Width / Streams, hc = Streams, lowrank = Lowrank, expanded = Width;
    static constexpr std::uint32_t down_type = 8U, up_type = DenseUp ? 0U : 8U, inject_type = 0U;
    using UpElement = std::conditional_t<DenseUp, float, Q8Block>;
    std::vector<float> input, norm, inject;
    std::vector<Q8Block> down;
    std::vector<UpElement> up;
    TypedProjectionFixture() : input(Width), norm(Width), inject(Width * hc),
        down(lowrank * (Width / 32U)), up(Width * (DenseUp ? lowrank : lowrank / 32U)) {
        static_assert(Width % 32U == 0U && Width % hc == 0U && (DenseUp || lowrank % 32U == 0U));
        constexpr std::array<std::uint16_t, 10> scales{0U, 0x8000U, 1U, 0x8001U, 0x1000U,
            0x1400U, 0x2000U, 0x2400U, 0x3000U, 0x7bffU};
        constexpr std::array<int, 8> codes{-128, -127, -17, -1, 0, 1, 63, 127};
        for (std::size_t i = 0; i < Width; ++i) {
            input[i] = static_cast<float>(static_cast<int>(i % 23U) - 11) / 17.0F;
            norm[i] = static_cast<float>(static_cast<int>(i % 13U) - 6) / 29.0F;
        }
        for (std::size_t b = 0; b < down.size(); ++b) {
            const auto bits = scales[b % scales.size()];
            std::memcpy(&down[b].scale, &bits, sizeof(bits));
            for (std::size_t j = 0; j < 32; ++j)
                down[b].values[j] = static_cast<std::int8_t>(bits == 0x7bffU ? 0 : codes[(j + b * 3U) % codes.size()]);
        }
        if constexpr (DenseUp) {
            for (std::size_t i = 0; i < up.size(); ++i)
                up[i] = static_cast<float>(static_cast<int>((i * 5U + i / lowrank) % 17U) - 8) / 191.0F;
        } else {
            for (std::size_t b = 0; b < up.size(); ++b) {
                const auto bits = static_cast<std::uint16_t>(0x0800U + (b % 7U) * 0x0100U);
                std::memcpy(&up[b].scale, &bits, sizeof(bits));
                for (std::size_t j = 0; j < 32; ++j)
                    up[b].values[j] = static_cast<std::int8_t>(static_cast<int>((b + j * 3U) % 15U) - 7);
            }
        }
        // A visible, isolated gate makes the last-block mutation meaningful
        // even after final BF16 storage. Other rows retain varied coefficients.
        for (std::size_t b = 0; b < Width / 32U; ++b)
            std::fill(std::begin(down[b].values), std::end(down[b].values), static_cast<std::int8_t>(0));
        const std::uint16_t unit_scale = 0x3c00U;
        std::memcpy(&down[Width / 32U - 1U].scale, &unit_scale, sizeof(unit_scale));
        down[Width / 32U - 1U].values[30] = 1;
        if constexpr (DenseUp) {
            std::fill(up.begin(), up.begin() + lowrank, 0.0F);
            up[0] = 16.0F;
        } else {
            for (std::size_t b = 0; b < lowrank / 32U; ++b)
                std::fill(std::begin(up[b].values), std::end(up[b].values), static_cast<std::int8_t>(0));
            std::memcpy(&up[0].scale, &unit_scale, sizeof(unit_scale));
            up[0].values[0] = 16;
        }
        for (std::size_t i = 0; i < inject.size(); ++i) {
            const float ordinary = static_cast<float>(static_cast<int>((i * 7U + i / 11U) % 31U) - 15) / 997.0F;
            inject[i] = std::nextafter(ordinary, i % 2U ? -INFINITY : INFINITY);
        }
        if constexpr (PrefetchValues) {
            input.back() = 0.375F;  // The last-feature mutation must be observable.
            inject[0] = 0.0F;
            inject[1] = -0.0F;
            inject[2] = std::numeric_limits<float>::denorm_min();
            inject[3] = -std::numeric_limits<float>::denorm_min();
            if constexpr (lowrank > 1U) {
                auto& extreme = down[2U * (Width / 32U) - 1U];
                const std::uint16_t max_scale = 0x7bffU;
                std::memcpy(&extreme.scale, &max_scale, sizeof(max_scale));
                std::fill(std::begin(extreme.values), std::end(extreme.values), static_cast<std::int8_t>(0));
                extreme.values[0] = 1;
                extreme.values[1] = -1;
            }
        }
    }
};

template <std::uint32_t Width, std::uint32_t Streams, std::uint32_t Lowrank>
struct OutputPrefetchFixture : TypedProjectionFixture<Width, Streams, Lowrank, false, true> {
    OutputPrefetchFixture() {
        constexpr std::array<std::uint16_t,12> scales{
            0U,0x8000U,1U,0x8001U,0x03ffU,0x83ffU,0x3c00U,0xbc00U,0x7bffU,0xfbffU,0x2400U,0xa400U};
        constexpr std::array<int,8> codes{-128,-127,-17,-1,0,1,63,127};
        // Retain isolated up row zero for the existing independent tail/no-op
        // control. Other rows vary signed finite scales and both code extremes.
        for (std::size_t b = Lowrank / 32U; b < this->up.size(); ++b) {
            const auto row = b / (Lowrank / 32U), column = b % (Lowrank / 32U);
            const auto bits = scales[(row * 7U + column * 5U + (row >> 2U)) % scales.size()];
            std::memcpy(&this->up[b].scale, &bits, sizeof(bits));
            for (std::size_t j = 0; j < 32U; ++j)
                this->up[b].values[j] = static_cast<std::int8_t>(codes[(j * 3U + row * 5U + column * 7U) % codes.size()]);
        }
    }
};

ReferenceResult reference(const std::vector<float>& input, const std::vector<float>& norm,
    const std::vector<Q8Block>& down, const std::vector<Q8Block>& up, const std::vector<float>& inject,
    std::uint32_t hidden, std::uint32_t hc, std::uint32_t lowrank, float epsilon, bool with_injection) {
    const auto unpack = [](const std::vector<Q8Block>& blocks) {
        std::vector<float> values(blocks.size() * 32U);
        for (std::size_t b = 0; b < blocks.size(); ++b)
            for (std::size_t j = 0; j < 32; ++j)
                values[b * 32U + j] = __half2float(blocks[b].scale) * static_cast<float>(blocks[b].values[j]);
        return values;
    };
    return reference(input, norm, unpack(down), unpack(up), inject, hidden, hc, lowrank, epsilon, with_injection);
}

ReferenceResult reference(const std::vector<float>& input, const std::vector<float>& norm,
    const std::vector<Q8Block>& down, const std::vector<float>& up, const std::vector<float>& inject,
    std::uint32_t hidden, std::uint32_t hc, std::uint32_t lowrank, float epsilon, bool with_injection) {
    std::vector<float> unpacked(down.size() * 32U);
    for (std::size_t b = 0; b < down.size(); ++b)
        for (std::size_t j = 0; j < 32U; ++j)
            unpacked[b * 32U + j] = __half2float(down[b].scale) * static_cast<float>(down[b].values[j]);
    return reference(input, norm, unpacked, up, inject, hidden, hc, lowrank, epsilon, with_injection);
}

template <typename FixtureType = Fixture>
void run_numeric_fixture(
    strix::SerialOp op,
    bool with_injection, bool bf16 = false, bool enable_prefetch = false,
    bool enable_output_prefetch = false) {
    using Fixture = FixtureType;
    auto model =
        strix::ModelContract::
            qwen38_flash_next();
    if (enable_prefetch || enable_output_prefetch) {
        // Allocate the exact fixture geometry before deriving any buffer or IO
        // plan: the checkpoint default has only four injection output lanes.
        model.hidden_size = Fixture::hidden;
        model.hc_count = Fixture::hc;
        model.hc_lowrank = Fixture::lowrank;
    }
    const auto plan =
        strix::build_serial_decode_execution_plan(
            model);
    const auto io =
        strix::build_serial_execution_io_plan(
            plan);
    const auto& step =
        first_step(plan, op);
    const auto& step_io =
        io.steps[step.id];

    strix::ExecutionBufferRequest request;
    request.logical_rows = 1;
    request.activation_element_bytes = bf16 ? 2U : 4U;
    request.control_element_bytes = 4;
    const auto arena =
        strix::build_serial_execution_arena_plan(
            model,
            request,
            plan,
            io,
            256);

    void* arena_pointer = nullptr;
    check(
        hipMalloc(
            &arena_pointer,
            static_cast<std::size_t>(
                arena.total_bytes)),
        "hipMalloc hyper-mix arena");

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity =
        0x45450000ULL +
        static_cast<std::uint64_t>(
            op);
    memory.arena_base =
        reinterpret_cast<
            std::uintptr_t>(
                arena_pointer);
    memory.arena_bytes =
        arena.total_bytes;
    memory.allocation_generation = 1;

    const auto input_address =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.reads[0]);
    const auto mixed_address =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.writes[0]);

    std::optional<
        strix::ExecutionBufferAddress>
        injection_address;
    if (with_injection) {
        injection_address =
            strix::resolve_execution_buffer(
                memory.execution_identity,
                arena,
                memory,
                step_io.writes[1]);
    }

    Fixture fixture;
    std::vector<std::uint16_t> input_bf16(fixture.input.size());
    if (bf16) {
        for (std::size_t index = 0; index < fixture.input.size(); ++index) {
            int exponent = 0;
            const double value = fixture.input[index];
            const double scaled = std::frexp(std::fabs(value), &exponent) * 256.0;
            auto integer = static_cast<unsigned int>(std::floor(scaled));
            const double fraction = scaled - integer;
            if (fraction > 0.5 || (fraction == 0.5 && (integer & 1U))) ++integer;
            fixture.input[index] = static_cast<float>(std::copysign(std::ldexp(static_cast<double>(integer), exponent - 8), value));
            input_bf16[index] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(fixture.input[index]) >> 16U);
        }
    }
    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                input_address.address),
            bf16 ? static_cast<const void*>(input_bf16.data()) : fixture.input.data(),
            fixture.input.size() * (bf16 ? sizeof(std::uint16_t) : sizeof(float)),
            hipMemcpyHostToDevice),
        "copy hyper-mix input");

    strix::ModelManifest manifest;
    manifest.architecture =
        "qwen4_exp";

    const auto add_manifest =
        [&](std::string name,
            std::uint32_t ggml_type,
            std::vector<std::uint64_t> dimensions,
            strix::TensorRole role,
            std::uint64_t bytes) {
            strix::ManifestTensor tensor;
            tensor.name =
                std::move(name);
            tensor.ggml_type =
                ggml_type;
            tensor.dimensions =
                std::move(dimensions);
            tensor.role =
                role;
            tensor.byte_size =
                bytes;
            if (op !=
                strix::SerialOp::
                    final_hyper_mix) {
                tensor.layer =
                    step.layer;
            }
            manifest.total_tensor_bytes +=
                bytes;
            manifest.tensors.push_back(
                std::move(tensor));
        };

    add_manifest(
        with_injection
            ? "blk.0.hc_attn_norm.weight"
            : "output_hc_norm.weight",
        0,
        {Fixture::expanded},
        strix::TensorRole::normalization,
        fixture.norm.size() *
            sizeof(float));
    add_manifest(
        with_injection
            ? "blk.0.hc_attn_down.weight"
            : "output_hc_down.weight",
        Fixture::down_type,
        {
            Fixture::expanded,
            Fixture::lowrank,
        },
        strix::TensorRole::dense_projection,
        fixture.down.size() *
            sizeof(typename decltype(fixture.down)::value_type));
    add_manifest(
        with_injection
            ? "blk.0.hc_attn_up.weight"
            : "output_hc_up.weight",
        Fixture::up_type,
        {
            Fixture::lowrank,
            Fixture::expanded,
        },
        strix::TensorRole::dense_projection,
        fixture.up.size() *
            sizeof(typename decltype(fixture.up)::value_type));
    if (with_injection) {
        add_manifest(
            "blk.0.hc_attn_inject.weight",
            Fixture::inject_type,
            {
                Fixture::expanded,
                Fixture::hc,
            },
            strix::TensorRole::dense_projection,
            fixture.inject.size() *
                sizeof(typename decltype(fixture.inject)::value_type));
    }

    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    std::vector<void*> device_weights(
        manifest.tensors.size(),
        nullptr);
    // The independent oracle consumes source zero-centered parameters;
    // GGUF serializes their already-folded multiplicative scales.
    auto gguf_norm = fixture.norm;
    for (auto& value : gguf_norm) value += 1.0F;
    upload(gguf_norm, &device_weights[0]);
    upload(
        fixture.down,
        &device_weights[1]);
    upload(
        fixture.up,
        &device_weights[2]);
    if (with_injection) {
        upload(
            fixture.inject,
            &device_weights[3]);
    }

    strix::ModelRuntimeBinding runtime;
    runtime.manifest_layout_identity =
        manifest.layout_identity;
    runtime.model_content_identity.bytes[0] =
        1;
    for (std::size_t index = 0;
         index <
            manifest.tensors.size();
         ++index) {
        runtime.tensors.push_back(
            {
                static_cast<
                    std::uint32_t>(
                    index),
                strix::RuntimeTensorStorage::
                    resident,
                reinterpret_cast<
                    std::uintptr_t>(
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
        0x56560000ULL +
        static_cast<std::uint64_t>(
            op);
    for (std::uint32_t slot = 0;
         slot <
            manifest.tensors.size();
         ++slot) {
        access.accesses.push_back(
            {
                step.id,
                step.op,
                slot,
                {slot},
            });
    }

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
        op;
    context.layer =
        step.layer;
    context.reads =
        step_io.reads;
    context.writes =
        step_io.writes;
    context.temporaries = step_io.temporaries;
    context.arena =
        &arena;
    context.memory =
        &memory;
    context.resources =
        &resources;
    context.logical_rows = 1;
    context.request_concurrency = 1;

    strix::HipHyperMixLauncherConfig config;
    config.hidden_size =
        Fixture::hidden;
    config.hc_count =
        Fixture::hc;
    config.lowrank =
        Fixture::lowrank;
    config.rms_epsilon =
        1e-6F;
    config.activation = bf16 ? strix::HipActivationStorage::bf16 : strix::HipActivationStorage::fp32;

    auto launcher =
        strix::make_hip_hyper_mix_launcher(
            "test.hyper_mix",
            &config);
    launcher.function(
        context,
        launcher.user_data);
    check(
        hipDeviceSynchronize(),
        "hyper-mix synchronize");

    std::vector<float> actual_mixed(
        Fixture::hidden);
    std::vector<std::uint16_t> actual_bf16(Fixture::hidden);
    check(
        hipMemcpy(
            bf16 ? static_cast<void*>(actual_bf16.data()) : actual_mixed.data(),
            reinterpret_cast<
                const void*>(
                mixed_address.address),
            actual_mixed.size() * (bf16 ? sizeof(std::uint16_t) : sizeof(float)),
            hipMemcpyDeviceToHost),
        "copy hyper-mix output");
    if (bf16) for (std::size_t index = 0; index < actual_mixed.size(); ++index)
        actual_mixed[index] = std::bit_cast<float>(static_cast<std::uint32_t>(actual_bf16[index]) << 16U);

    std::vector<float> actual_injection;
    if (with_injection) {
        actual_injection.resize(
            Fixture::hc);
        check(
            hipMemcpy(
                actual_injection.data(),
                reinterpret_cast<
                    const void*>(
                    injection_address->
                        address),
                actual_injection.size() *
                    sizeof(float),
                hipMemcpyDeviceToHost),
            "copy hyper-mix injection");
    }

    const auto expected =
        reference(
            fixture.input,
            fixture.norm,
            fixture.down,
            fixture.up,
            fixture.inject,
            Fixture::hidden,
            Fixture::hc,
            Fixture::lowrank,
            1e-6F,
            with_injection);

    float mixed_error = 0.0F;
    for (std::size_t index = 0;
         index <
            actual_mixed.size();
         ++index) {
        // The finite extreme-output cases can exceed one: BF16 rounding alone
        // can then exceed the old absolute CPU tolerance. Account only for the
        // storage half-ULP bound here; every legacy/candidate bit check is exact.
        const float storage_slack = bf16 && enable_output_prefetch ?
            std::ldexp(std::abs(expected.mixed[index]), -8) : 0.0F;
        if (enable_output_prefetch)
            expect(std::isfinite(expected.mixed[index]) && std::isfinite(actual_mixed[index]),
                   "finite output-prefetch data has a finite CPU/GPU reference");
        mixed_error = std::max(mixed_error,
            std::abs(actual_mixed[index] - expected.mixed[index]) - storage_slack);
    }
    expect(
        mixed_error <= 2e-3F,
        "fused hyper-mix output matches CPU reference");

    if (with_injection) {
        float injection_error = 0.0F;
        for (std::size_t index = 0;
             index <
                actual_injection.size();
             ++index) {
            injection_error =
                std::max(
                    injection_error,
                    std::abs(
                        actual_injection[index] -
                        expected.injection[index]));
        }
        expect(
            injection_error <= 2e-3F,
            "fused hyper-mix injection weights match CPU reference");
    }

    auto parallel_config = config;
    parallel_config.parallel = true;
    auto parallel_launcher = strix::make_hip_hyper_mix_launcher("test.hyper_mix.parallel", &parallel_config);
    expect(parallel_launcher.implementation_identity != launcher.implementation_identity,
           "parallel hyper-mix has a distinct numerical implementation identity");
    parallel_launcher.function(context, parallel_launcher.user_data);
    check(hipDeviceSynchronize(), "parallel hyper-mix synchronize");
    std::vector<float> parallel_mixed(actual_mixed.size());
    check(hipMemcpy(bf16 ? static_cast<void*>(actual_bf16.data()) : parallel_mixed.data(), reinterpret_cast<void*>(mixed_address.address),
                    parallel_mixed.size() * (bf16 ? sizeof(std::uint16_t) : sizeof(float)), hipMemcpyDeviceToHost), "read parallel hyper-mix output");
    if (bf16) for (std::size_t index = 0; index < parallel_mixed.size(); ++index)
        parallel_mixed[index] = std::bit_cast<float>(static_cast<std::uint32_t>(actual_bf16[index]) << 16U);
    expect(parallel_mixed == actual_mixed, "parallel projections preserve legacy FP32 accumulation results");
    if (with_injection) {
        std::vector<float> parallel_injection(actual_injection.size());
        check(hipMemcpy(parallel_injection.data(), reinterpret_cast<void*>(injection_address->address),
                        parallel_injection.size() * sizeof(float), hipMemcpyDeviceToHost), "read parallel injection");
        expect(parallel_injection == actual_injection, "parallel injection preserves legacy accumulation results");
    }
    auto staged_config = parallel_config;
    staged_config.staged_projections = true;
    auto staged_launcher = strix::make_hip_hyper_mix_launcher("test.hyper_mix.staged", &staged_config);
    expect(staged_launcher.implementation_identity != parallel_launcher.implementation_identity,
           "staged projections have a distinct implementation identity");
    // Preserve immutable baseline bits independently of all later device writes.
    const auto expected_bf16_bits = actual_bf16;
    const auto workspace_address = strix::resolve_dispatch_temporary(context, 0);
    const auto poison_outputs = [&](hipStream_t stream) {
        check(hipMemsetAsync(reinterpret_cast<void*>(workspace_address.address), 0xff,
                             static_cast<std::size_t>(workspace_address.bytes), stream),
              "poison complete admitted hyper workspace");
        check(hipMemsetAsync(reinterpret_cast<void*>(mixed_address.address), 0xff,
                             parallel_mixed.size() * (bf16 ? sizeof(std::uint16_t) : sizeof(float)), stream),
              "poison hyper-mix output");
        if (with_injection)
            check(hipMemsetAsync(reinterpret_cast<void*>(injection_address->address), 0xff,
                                 actual_injection.size() * sizeof(float), stream), "poison hyper injection");
    };
    const auto outputs_match_expected_bits = [&] {
        const auto mixed_bytes = parallel_mixed.size() * (bf16 ? sizeof(std::uint16_t) : sizeof(float));
        std::vector<std::uint8_t> observed(mixed_bytes);
        check(hipMemcpy(observed.data(), reinterpret_cast<void*>(mixed_address.address),
                        mixed_bytes, hipMemcpyDeviceToHost), "read hyper-mix output bits");
        const void* expected = bf16 ? static_cast<const void*>(expected_bf16_bits.data()) : parallel_mixed.data();
        bool matches = std::memcmp(observed.data(), expected, mixed_bytes) == 0;
        if (with_injection) {
            const auto injection_bytes = actual_injection.size() * sizeof(float);
            std::vector<std::uint8_t> injection(injection_bytes);
            check(hipMemcpy(injection.data(), reinterpret_cast<void*>(injection_address->address),
                            injection_bytes, hipMemcpyDeviceToHost), "read hyper injection bits");
            matches &= std::memcmp(injection.data(), actual_injection.data(), injection_bytes) == 0;
        }
        return matches;
    };
    // Every candidate starts from NaN poison in its entire admitted scratch and
    // outputs. No-op or partial writes cannot inherit prior correct values.
    hipStream_t staged_stream = nullptr;
    hipGraph_t staged_graph = nullptr;
    hipGraphExec_t staged_executable = nullptr;
    check(hipStreamCreateWithFlags(&staged_stream, hipStreamNonBlocking), "create staged stream");
    strix::HipLaunchContext staged_scope;
    staged_scope.stream = staged_stream;
    {
        strix::HipLaunchScope scope(staged_scope);
        poison_outputs(staged_stream);
        check(hipStreamSynchronize(staged_stream), "synchronize staged no-op control");
        expect(!outputs_match_expected_bits(), "poisoned no-op fails staged bit comparison");
        staged_launcher.function(context, staged_launcher.user_data);
        check(hipStreamSynchronize(staged_stream), "synchronize eager staged projection");
        expect(outputs_match_expected_bits(), "eager staged output and injection preserve baseline bits");
        check(hipStreamBeginCapture(staged_stream, hipStreamCaptureModeThreadLocal), "capture staged projection");
        staged_launcher.function(context, staged_launcher.user_data);
        check(hipStreamEndCapture(staged_stream, &staged_graph), "finish staged capture");
    }
    check(hipGraphInstantiate(&staged_executable, staged_graph, nullptr, nullptr, 0), "instantiate staged graph");
    for (unsigned replay = 0; replay < 3; ++replay) {
        poison_outputs(staged_stream);
        check(hipGraphLaunch(staged_executable, staged_stream), "replay staged graph");
        check(hipStreamSynchronize(staged_stream), "synchronize staged graph");
        expect(outputs_match_expected_bits(), "staged replay output and injection preserve baseline bits");
    }
    check(hipGraphExecDestroy(staged_executable), "destroy staged executable");
    check(hipGraphDestroy(staged_graph), "destroy staged graph");
    check(hipStreamDestroy(staged_stream), "destroy staged stream");
    auto fused_config = staged_config;
    fused_config.fused_injection = true;
    auto fused_launcher = strix::make_hip_hyper_mix_launcher("test.hyper_mix.fused.v5", &fused_config);
    expect(fused_launcher.implementation_identity != staged_launcher.implementation_identity,
           "fused injection v5 has a distinct implementation revision");
    hipStream_t fused_stream = nullptr;
    hipGraph_t fused_graph = nullptr;
    hipGraphExec_t fused_executable = nullptr;
    check(hipStreamCreateWithFlags(&fused_stream, hipStreamNonBlocking), "create fused stream");
    strix::HipLaunchContext fused_scope;
    fused_scope.stream = fused_stream;
    {
        strix::HipLaunchScope scope(fused_scope);
        poison_outputs(fused_stream);
        check(hipStreamSynchronize(fused_stream), "synchronize fused no-op control");
        expect(!outputs_match_expected_bits(), "poisoned no-op fails fused bit comparison");
        fused_launcher.function(context, fused_launcher.user_data);
        check(hipStreamSynchronize(fused_stream), "synchronize eager fused injection");
        expect(outputs_match_expected_bits(), "eager fused output and injection preserve baseline bits");
        check(hipStreamBeginCapture(fused_stream, hipStreamCaptureModeThreadLocal), "capture fused injection");
        fused_launcher.function(context, fused_launcher.user_data);
        check(hipStreamEndCapture(fused_stream, &fused_graph), "finish fused capture");
    }
    check(hipGraphInstantiate(&fused_executable, fused_graph, nullptr, nullptr, 0), "instantiate fused graph");
    for (unsigned replay = 0; replay < 3; ++replay) {
        poison_outputs(fused_stream);
        check(hipGraphLaunch(fused_executable, fused_stream), "replay fused graph");
        check(hipStreamSynchronize(fused_stream), "synchronize fused replay");
        expect(outputs_match_expected_bits(), "fused replay output and injection preserve baseline bits");
    }
    check(hipGraphExecDestroy(fused_executable), "destroy fused executable");
    check(hipGraphDestroy(fused_graph), "destroy fused graph");
    check(hipStreamDestroy(fused_stream), "destroy fused stream");
    if constexpr (Fixture::down_type == 8U && Fixture::inject_type == 0U) {
        auto typed_config = fused_config;
        typed_config.typed_projection = true;
        auto typed_launcher = strix::make_hip_hyper_mix_launcher("test.hyper_mix.typed.q8.f32", &typed_config);
        expect(typed_launcher.implementation_identity != fused_launcher.implementation_identity,
               "typed projection has a unique implementation identity");
        // Every compiled mixer is inspected, including the last binding.
        strix::CompiledSerialExecution preflight_compiled;
        preflight_compiled.execution = plan;
        preflight_compiled.execution_identity = 1U;
        strix::ExecutionTensorAccessPlan complete_access;
        for (const auto& mixer : plan.steps) {
            if (mixer.op != strix::SerialOp::attn_hyper_mix && mixer.op != strix::SerialOp::mlp_hyper_mix &&
                mixer.op != strix::SerialOp::final_hyper_mix) continue;
            complete_access.accesses.push_back({mixer.id, mixer.op, 1U, {1U}});
            if (with_injection && mixer.op != strix::SerialOp::final_hyper_mix)
                complete_access.accesses.push_back({mixer.id, mixer.op, 3U, {3U}});
        }
        auto preflight_resources = resources;
        preflight_resources.tensor_access = &complete_access;
        if (with_injection) strix::preflight_typed_hyper_projection_model(preflight_compiled, preflight_resources);
        else {
            bool incomplete_rejected = false;
            try { strix::preflight_typed_hyper_projection_model(preflight_compiled, preflight_resources); }
            catch (const std::invalid_argument&) { incomplete_rejected = true; }
            expect(incomplete_rejected, "final-only resources cannot preflight a complete model");
        }
        if (with_injection) {
            const auto old_down_type = manifest.tensors[1].ggml_type;
            manifest.tensors[1].ggml_type = 0U;
            bool wrong_model_rejected = false;
            try { strix::preflight_typed_hyper_projection_model(preflight_compiled, preflight_resources); }
            catch (const std::invalid_argument&) { wrong_model_rejected = true; }
            expect(wrong_model_rejected, "whole-model typed preflight fails closed on wrong down format");
            manifest.tensors[1].ggml_type = old_down_type;
            const auto old_inject_type = manifest.tensors[3].ggml_type;
            manifest.tensors[3].ggml_type = 8U;
            bool wrong_inject_rejected = false;
            try { strix::preflight_typed_hyper_projection_model(preflight_compiled, preflight_resources); }
            catch (const std::invalid_argument& e) {
                wrong_inject_rejected = std::string_view(e.what()) ==
                    "typed hyper model contains a non-F32 injection projection";
            }
            expect(wrong_inject_rejected, "whole-model typed preflight independently rejects wrong injection format");
            manifest.tensors[3].ggml_type = old_inject_type;
            auto late_manifest = manifest;
            auto bad_down = late_manifest.tensors[1]; bad_down.ggml_type = 0U;
            late_manifest.tensors.push_back(bad_down);
            auto late_access = complete_access;
            late_access.accesses.back().tensor_indices[0] = static_cast<std::uint32_t>(late_manifest.tensors.size() - 1U);
            auto late_resources = preflight_resources;
            late_resources.model_manifest = &late_manifest;
            late_resources.tensor_access = &late_access;
            bool late_rejected = false;
            try { strix::preflight_typed_hyper_projection_model(preflight_compiled, late_resources); }
            catch (const std::invalid_argument&) { late_rejected = true; }
            expect(late_rejected, "typed preflight inspects the last of all mixer bindings");
        }
        for (unsigned invalid = 0; invalid < (with_injection ? 2U : 1U); ++invalid) {
            const auto index = invalid == 0U ? 1U : 3U;
            const auto previous = manifest.tensors[index].ggml_type;
            manifest.tensors[index].ggml_type = invalid == 0U ? 0U : 8U;
            poison_outputs(nullptr);
            check(hipDeviceSynchronize(), "drain typed rejection poison");
            std::vector<std::byte> untouched(static_cast<std::size_t>(workspace_address.bytes));
            check(hipMemcpy(untouched.data(), reinterpret_cast<void*>(workspace_address.address), untouched.size(), hipMemcpyDeviceToHost),
                  "snapshot typed rejection workspace");
            std::vector<std::byte> output_before(Fixture::hidden * (bf16 ? sizeof(std::uint16_t) : sizeof(float)));
            std::vector<std::byte> injection_before(with_injection ? Fixture::hc * sizeof(float) : 0U);
            check(hipMemcpy(output_before.data(), reinterpret_cast<void*>(mixed_address.address), output_before.size(), hipMemcpyDeviceToHost),
                  "snapshot typed rejection output");
            if (with_injection)
                check(hipMemcpy(injection_before.data(), reinterpret_cast<void*>(injection_address->address), injection_before.size(), hipMemcpyDeviceToHost),
                      "snapshot typed rejection injection");
            bool rejected = false;
            try { typed_launcher.function(context, typed_launcher.user_data); }
            catch (const std::invalid_argument& e) {
                rejected = std::string_view(e.what()) ==
                    "typed hyper projection requires Q8_0 down and F32 injection";
            }
            expect(rejected, "typed per-launch format defense rejects before enqueue");
            check(hipDeviceSynchronize(), "drain typed rejected launch");
            std::vector<std::byte> after(untouched.size());
            check(hipMemcpy(after.data(), reinterpret_cast<void*>(workspace_address.address), after.size(), hipMemcpyDeviceToHost),
                  "read typed rejected workspace");
            std::vector<std::byte> output_after(output_before.size()), injection_after(injection_before.size());
            check(hipMemcpy(output_after.data(), reinterpret_cast<void*>(mixed_address.address), output_after.size(), hipMemcpyDeviceToHost),
                  "read typed rejection output");
            if (with_injection)
                check(hipMemcpy(injection_after.data(), reinterpret_cast<void*>(injection_address->address), injection_after.size(), hipMemcpyDeviceToHost),
                      "read typed rejection injection");
            expect(after == untouched && output_after == output_before && injection_after == injection_before,
                   "wrong typed formats preserve all poisoned buffers byte for byte");
            manifest.tensors[index].ggml_type = previous;
        }
        // A changed coefficient in the last block must affect the generic control.
        auto changed_down = fixture.down;
        changed_down[Fixture::expanded / 32U - 1U].values[30] = static_cast<std::int8_t>(
            static_cast<std::uint8_t>(changed_down[Fixture::expanded / 32U - 1U].values[30]) ^ 0x3fU);
        check(hipMemcpy(device_weights[1], changed_down.data(), changed_down.size() * sizeof(Q8Block), hipMemcpyHostToDevice),
              "mutate last Q8 block as indexing sensitivity control");
        poison_outputs(nullptr);
        fused_launcher.function(context, fused_launcher.user_data);
        check(hipDeviceSynchronize(), "drain generic mutation control");
        expect(!outputs_match_expected_bits(), "last-block Q8 mutation changes generic output bits");
        check(hipMemcpy(device_weights[1], fixture.down.data(), fixture.down.size() * sizeof(Q8Block), hipMemcpyHostToDevice),
              "restore Q8 control weights");
        hipStream_t typed_stream = nullptr;
        hipGraph_t typed_graph = nullptr;
        hipGraphExec_t typed_executable = nullptr;
        check(hipStreamCreateWithFlags(&typed_stream, hipStreamNonBlocking), "create typed stream");
        strix::HipLaunchContext typed_scope; typed_scope.stream = typed_stream;
        {
            strix::HipLaunchScope scope(typed_scope);
            poison_outputs(typed_stream);
            check(hipStreamSynchronize(typed_stream), "drain typed no-op poison");
            expect(!outputs_match_expected_bits(), "typed no-op poison comparison cannot pass");
            typed_launcher.function(context, typed_launcher.user_data);
            check(hipStreamSynchronize(typed_stream), "drain typed eager run");
            expect(outputs_match_expected_bits(), "typed eager output/injection preserve generic bits");
            check(hipStreamBeginCapture(typed_stream, hipStreamCaptureModeThreadLocal), "capture typed projection");
            typed_launcher.function(context, typed_launcher.user_data);
            check(hipStreamEndCapture(typed_stream, &typed_graph), "finish typed capture");
        }
        check(hipGraphInstantiate(&typed_executable, typed_graph, nullptr, nullptr, 0), "instantiate typed graph");
        for (unsigned replay = 0; replay < 3U; ++replay) {
            poison_outputs(typed_stream);
            check(hipGraphLaunch(typed_executable, typed_stream), "replay typed graph");
            check(hipStreamSynchronize(typed_stream), "drain typed replay");
            expect(outputs_match_expected_bits(), "poisoned typed replay preserves generic bits");
        }
        check(hipGraphExecDestroy(typed_executable), "destroy typed executable");
        check(hipGraphDestroy(typed_graph), "destroy typed graph");
        check(hipStreamDestroy(typed_stream), "destroy typed stream");
        if (enable_prefetch || enable_output_prefetch) {
            auto prefetch_config = typed_config;
            prefetch_config.prefetch_rows = true;
            prefetch_config.prefetch_q8_output = enable_output_prefetch;
            auto prefetch_launcher = strix::make_hip_hyper_mix_launcher("test.hyper_mix.prefetch.rows8.v2", &prefetch_config);
            expect(prefetch_launcher.implementation_identity != typed_launcher.implementation_identity,
                   "row-prefetch implementation differs from typed-v1");
            // Include normalized and low FP32 workspace, so activation/storage
            // rounding cannot hide a wrong down row or signed-zero result.
            const auto used_workspace_bytes = static_cast<std::size_t>(Fixture::expanded + Fixture::lowrank) * sizeof(float);
            const auto mixed_bytes = static_cast<std::size_t>(Fixture::hidden) * (bf16 ? sizeof(std::uint16_t) : sizeof(float));
            const auto inject_bytes = with_injection ? Fixture::hc * sizeof(float) : 0U;
            const auto read_state = [&] {
                std::vector<std::byte> state(used_workspace_bytes + mixed_bytes + inject_bytes);
                check(hipMemcpy(state.data(), reinterpret_cast<void*>(workspace_address.address), used_workspace_bytes,
                                hipMemcpyDeviceToHost), "read exact normalized and low state");
                check(hipMemcpy(state.data() + used_workspace_bytes, reinterpret_cast<void*>(mixed_address.address),
                                mixed_bytes, hipMemcpyDeviceToHost), "read exact prefetch mixed state");
                if (with_injection)
                    check(hipMemcpy(state.data() + used_workspace_bytes + mixed_bytes,
                                    reinterpret_cast<void*>(injection_address->address), inject_bytes,
                                    hipMemcpyDeviceToHost), "read exact prefetch injection state");
                return state;
            };
            const auto expected_state = read_state();
            const auto paired_current_weights = [&] {
                poison_outputs(nullptr);
                typed_launcher.function(context, typed_launcher.user_data);
                check(hipDeviceSynchronize(), "drain typed-v1 mutation oracle");
                const auto wanted = read_state();
                poison_outputs(nullptr);
                expect(read_state() != wanted, "prefetch no-op state cannot satisfy current typed oracle");
                prefetch_launcher.function(context, prefetch_launcher.user_data);
                check(hipDeviceSynchronize(), "drain prefetch mutation pair");
                expect(read_state() == wanted, "prefetch mutated weights preserve full typed-v1 state");
                return wanted;
            };
            std::vector<std::uint32_t> changed_rows{std::min(7U, Fixture::lowrank - 1U), Fixture::lowrank - 1U};
            if (changed_rows.front() == changed_rows.back()) changed_rows.pop_back();
            for (const auto changed_row : changed_rows) {
                auto changed = fixture.down;
                const auto blocks = Fixture::expanded / 32U;
                for (std::uint32_t b = 0; b < blocks; ++b)
                    std::fill(std::begin(changed[changed_row * blocks + b].values),
                              std::end(changed[changed_row * blocks + b].values), static_cast<std::int8_t>(0));
                auto& last = changed[changed_row * blocks + blocks - 1U];
                const std::uint16_t unit = 0x3c00U;
                std::memcpy(&last.scale, &unit, sizeof(unit));
                last.values[30] = 1;
                check(hipMemcpy(device_weights[1], changed.data(), changed.size() * sizeof(Q8Block), hipMemcpyHostToDevice),
                      "upload row-sensitive prefetch mutation A");
                const auto first = paired_current_weights();
                last.values[30] = 62;
                check(hipMemcpy(device_weights[1], changed.data(), changed.size() * sizeof(Q8Block), hipMemcpyHostToDevice),
                      "upload row-sensitive prefetch mutation B");
                expect(paired_current_weights() != first, "last/full-row mutation changes typed low state");
            }
            check(hipMemcpy(device_weights[1], fixture.down.data(), fixture.down.size() * sizeof(Q8Block), hipMemcpyHostToDevice),
                  "restore original prefetch down weights");
            if (with_injection) {
                auto changed = fixture.inject;
                const auto first = static_cast<std::ptrdiff_t>((Fixture::hc - 1U) * Fixture::expanded);
                std::fill(changed.begin() + first, changed.begin() + first + Fixture::expanded, 0.0F);
                changed.back() = 1.0F;
                check(hipMemcpy(device_weights[3], changed.data(), changed.size() * sizeof(float), hipMemcpyHostToDevice),
                      "upload last injection row/tail mutation A");
                const auto before = paired_current_weights();
                changed.back() = 62.0F;
                check(hipMemcpy(device_weights[3], changed.data(), changed.size() * sizeof(float), hipMemcpyHostToDevice),
                      "upload last injection row/tail mutation B");
                expect(paired_current_weights() != before, "last injection row/tail mutation changes typed state");
                check(hipMemcpy(device_weights[3], fixture.inject.data(), fixture.inject.size() * sizeof(float), hipMemcpyHostToDevice),
                      "restore original prefetch injection weights");
            }
            if (enable_output_prefetch) {
                if constexpr (Fixture::up_type == 8U) {
                    // Force a visible low-rank tail independently of the existing
                    // varied fixture. Then mutate first/full/last up rows at K-1.
                    auto visible_down = fixture.down;
                    const auto down_blocks = Fixture::expanded / 32U;
                    for (std::uint32_t b = 0; b < down_blocks; ++b)
                        std::fill(std::begin(visible_down[(Fixture::lowrank - 1U) * down_blocks + b].values),
                                  std::end(visible_down[(Fixture::lowrank - 1U) * down_blocks + b].values), static_cast<std::int8_t>(0));
                    const std::uint16_t unit = 0x3c00U;
                    auto& visible = visible_down[(Fixture::lowrank - 1U) * down_blocks];
                    std::memcpy(&visible.scale, &unit, sizeof(unit)); visible.values[0] = 1;
                    check(hipMemcpy(device_weights[1], visible_down.data(), visible_down.size() * sizeof(Q8Block), hipMemcpyHostToDevice),
                          "upload visible low-rank output tail control");
                    for (const auto row : {0U, std::min(7U, Fixture::expanded - 1U), Fixture::expanded - 1U}) {
                        auto changed_up = fixture.up;
                        const auto up_blocks = Fixture::lowrank / 32U;
                        for (std::uint32_t b = 0; b < up_blocks; ++b)
                            std::fill(std::begin(changed_up[row * up_blocks + b].values),
                                      std::end(changed_up[row * up_blocks + b].values), static_cast<std::int8_t>(0));
                        auto& tail = changed_up[row * up_blocks + up_blocks - 1U];
                        std::memcpy(&tail.scale, &unit, sizeof(unit)); tail.values[31] = 1;
                        check(hipMemcpy(device_weights[2], changed_up.data(), changed_up.size() * sizeof(Q8Block), hipMemcpyHostToDevice),
                              "upload output row/tail sensitivity A");
                        const auto first = paired_current_weights();
                        tail.values[31] = 62;
                        check(hipMemcpy(device_weights[2], changed_up.data(), changed_up.size() * sizeof(Q8Block), hipMemcpyHostToDevice),
                              "upload output row/tail sensitivity B");
                        expect(paired_current_weights() != first, "up first/full/last row and K-tail mutation changes output bits");
                    }
                    check(hipMemcpy(device_weights[1], fixture.down.data(), fixture.down.size() * sizeof(Q8Block), hipMemcpyHostToDevice),
                          "restore original output-control down weights");
                    check(hipMemcpy(device_weights[2], fixture.up.data(), fixture.up.size() * sizeof(Q8Block), hipMemcpyHostToDevice),
                          "restore original output-control up weights");
                }
            }
            expect(paired_current_weights() == expected_state, "restored prefetch model state equals original typed state");
            for (unsigned invalid = 0; invalid < (with_injection ? 2U : 1U); ++invalid) {
                const auto index = invalid == 0U ? 1U : 3U;
                const auto previous = manifest.tensors[index].ggml_type;
                manifest.tensors[index].ggml_type = invalid == 0U ? 0U : 8U;
                poison_outputs(nullptr);
                const auto untouched = read_state();
                std::vector<std::byte> entire_workspace(static_cast<std::size_t>(workspace_address.bytes));
                check(hipMemcpy(entire_workspace.data(), reinterpret_cast<void*>(workspace_address.address),
                                entire_workspace.size(), hipMemcpyDeviceToHost), "snapshot entire rejected prefetch workspace");
                bool rejected = false;
                try { prefetch_launcher.function(context, prefetch_launcher.user_data); }
                catch (const std::invalid_argument& e) {
                    rejected = std::string_view(e.what()) == "typed hyper projection requires Q8_0 down and F32 injection";
                }
                check(hipDeviceSynchronize(), "drain rejected prefetch launch");
                std::vector<std::byte> entire_after(entire_workspace.size());
                check(hipMemcpy(entire_after.data(), reinterpret_cast<void*>(workspace_address.address), entire_after.size(),
                                hipMemcpyDeviceToHost), "read entire rejected prefetch workspace");
                expect(rejected && read_state() == untouched && entire_after == entire_workspace,
                       "prefetch wrong format preserves entire workspace/output/injection byte for byte");
                manifest.tensors[index].ggml_type = previous;
            }
            if (enable_output_prefetch) {
                const auto previous = manifest.tensors[2].ggml_type;
                manifest.tensors[2].ggml_type = 0U;
                poison_outputs(nullptr); check(hipDeviceSynchronize(), "drain wrong-up poison");
                const auto before = read_state();
                std::vector<std::byte> scratch(static_cast<std::size_t>(workspace_address.bytes));
                check(hipMemcpy(scratch.data(), reinterpret_cast<void*>(workspace_address.address), scratch.size(), hipMemcpyDeviceToHost),
                      "snapshot complete wrong-up scratch");
                bool rejected = false;
                try { prefetch_launcher.function(context, prefetch_launcher.user_data); }
                catch (const std::invalid_argument& e) {
                    rejected = std::string_view(e.what()) == "Q8 hyper output prefetch requires Q8_0 up projection";
                }
                check(hipDeviceSynchronize(), "drain wrong-up rejection");
                std::vector<std::byte> after(scratch.size());
                check(hipMemcpy(after.data(), reinterpret_cast<void*>(workspace_address.address), after.size(), hipMemcpyDeviceToHost),
                      "read complete wrong-up scratch");
                expect(rejected && read_state() == before && after == scratch,
                       "wrong F32 up fails for the exact format reason before any workspace/output/injection mutation");
                manifest.tensors[2].ggml_type = previous;
            }
            hipStream_t stream = nullptr;
            hipGraph_t graph = nullptr;
            hipGraphExec_t executable = nullptr;
            check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create prefetch capture stream");
            strix::HipLaunchContext launch_scope; launch_scope.stream = stream;
            {
                strix::HipLaunchScope scope(launch_scope);
                poison_outputs(stream);
                check(hipStreamSynchronize(stream), "drain prefetch no-op poison");
                expect(read_state() != expected_state, "prefetch no-op full-state control fails equality");
                prefetch_launcher.function(context, prefetch_launcher.user_data);
                check(hipStreamSynchronize(stream), "drain prefetch eager");
                expect(read_state() == expected_state, "prefetch eager preserves typed normalized/low/output/injection bytes");
                check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "capture row-prefetch projection");
                prefetch_launcher.function(context, prefetch_launcher.user_data);
                check(hipStreamEndCapture(stream, &graph), "finish row-prefetch capture");
            }
            check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate row-prefetch graph");
            for (unsigned replay = 0; replay < 3U; ++replay) {
                poison_outputs(stream);
                check(hipGraphLaunch(executable, stream), "launch poisoned row-prefetch replay");
                check(hipStreamSynchronize(stream), "drain row-prefetch replay");
                expect(read_state() == expected_state, "row-prefetch replay preserves complete typed state");
            }
            check(hipGraphExecDestroy(executable), "destroy row-prefetch executable");
            check(hipGraphDestroy(graph), "destroy row-prefetch graph");
            check(hipStreamDestroy(stream), "destroy row-prefetch stream");
        }
    }
    auto invalid_fused_config = fused_config;
    invalid_fused_config.staged_projections = false;
    bool invalid_fused_rejected = false;
    try { (void)strix::make_hip_hyper_mix_launcher("test.invalid.fused", &invalid_fused_config); }
    catch (const std::invalid_argument&) { invalid_fused_rejected = true; }
    expect(invalid_fused_rejected, "fused injection rejects a nonstaged configuration");
    auto invalid_staged_config = staged_config;
    invalid_staged_config.parallel = false;
    bool invalid_staged_rejected = false;
    try { (void)strix::make_hip_hyper_mix_launcher("test.invalid.staged", &invalid_staged_config); }
    catch (const std::invalid_argument&) { invalid_staged_rejected = true; }
    expect(invalid_staged_rejected, "staged projections reject a nonparallel configuration");
    auto missing_workspace = context;
    missing_workspace.temporaries = {};
    bool rejected = false;
    try { parallel_launcher.function(missing_workspace, parallel_launcher.user_data); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "parallel hyper-mix rejects missing admitted scratch");
    rejected = false;
    try { staged_launcher.function(missing_workspace, staged_launcher.user_data); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "staged projections reject missing admitted scratch");
    rejected = false;
    try { fused_launcher.function(missing_workspace, fused_launcher.user_data); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "fused injection rejects missing admitted scratch");
    for (unsigned malformed = 0; malformed < 3; ++malformed) {
        auto invalid_arena = arena;
        auto invalid_context = context;
        auto invalid_temporaries = step_io.temporaries;
        const auto original_use = invalid_temporaries.front();
        if (malformed == 0)
            invalid_temporaries.front().buffer.buffer_class = strix::ExecutionBufferClass::hidden;
        for (auto& allocation : invalid_arena.allocations) {
            if (allocation.use != original_use) continue;
            allocation.use = invalid_temporaries.front();
            if (malformed == 1) allocation.bytes =
                (static_cast<std::uint64_t>(Fixture::expanded) + Fixture::lowrank) * sizeof(float) - 1U;
            if (malformed == 2) ++allocation.offset;
        }
        invalid_context.arena = &invalid_arena;
        invalid_context.temporaries = invalid_temporaries;
        rejected = false;
        try { staged_launcher.function(invalid_context, staged_launcher.user_data); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "staged projections reject wrong-class, short or unaligned scratch");
        rejected = false;
        try { fused_launcher.function(invalid_context, fused_launcher.user_data); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "fused injection rejects wrong-class, short or unaligned scratch");
    }

    for (auto* pointer :
         device_weights) {
        check(
            hipFree(pointer),
            "hipFree hyper-mix weight");
    }
    check(
        hipFree(arena_pointer),
        "hipFree hyper-mix arena");
}

void test_catalog_and_bundle() {
    const auto model =
        strix::ModelContract::
            qwen38_flash_next();

    strix::ExecutionBindingRequest request;
    request.logical_rows = 1;
    request.request_concurrency = 1;
    request.minimum_qualification =
        strix::kernels::
            QualificationStatus::
                experimental;

    strix::HipHyperMixLauncherConfig config;
    config.hidden_size =
        static_cast<std::uint32_t>(
            model.hidden_size);
    config.hc_count =
        static_cast<std::uint32_t>(
            model.hc_count);
    config.lowrank =
        static_cast<std::uint32_t>(
            model.hc_lowrank);
    config.rms_epsilon =
        1e-6F;
    config.activation =
        strix::HipActivationStorage::bf16;

    strix::kernels::KernelRegistry registry;
    strix::register_hip_hyper_mix_kernel_variant(
        model,
        request,
        config,
        registry);
    auto staged_config = config;
    staged_config.parallel = true;
    strix::kernels::KernelRegistry parallel_registry;
    strix::register_hip_hyper_mix_kernel_variant(model, request, staged_config, parallel_registry);
    expect(parallel_registry.variants().front().resources.lds_bytes >= (256U + 1U) * sizeof(float),
           "parallel LDS declaration includes reduction and shared inverse");
    staged_config.staged_projections = true;
    strix::kernels::KernelRegistry staged_registry;
    strix::register_hip_hyper_mix_kernel_variant(model, request, staged_config, staged_registry);
    const auto& staged_variant = staged_registry.variants().front();
    expect(staged_variant.resources.lds_bytes >= (128U + 8U * 129U) * sizeof(float) &&
           staged_variant.resources.lds_bytes >= (256U + 1U) * sizeof(float),
           "staged LDS declaration covers projection tiles and normalization");
    expect(staged_variant.resources.scratch_bytes ==
           (model.hidden_size * model.hc_count + model.hc_lowrank) * sizeof(float),
           "staged projections preserve admitted workspace size");
    expect(staged_variant.id != parallel_registry.variants().front().id &&
           staged_variant.compiler_identity != parallel_registry.variants().front().compiler_identity,
           "staged kernel and compiler identities distinguish the numerical path");
    auto fused_config = staged_config;
    fused_config.fused_injection = true;
    strix::kernels::KernelRegistry fused_registry;
    strix::register_hip_hyper_mix_kernel_variant(model, request, fused_config, fused_registry);
    const auto& fused_variant = fused_registry.variants().front();
    expect(fused_variant.id != staged_variant.id && fused_variant.compiler_identity != staged_variant.compiler_identity,
           "fused injection changes kernel and compiler revision identities");
    expect(fused_variant.resources.lds_bytes == staged_variant.resources.lds_bytes &&
           fused_variant.resources.scratch_bytes == staged_variant.resources.scratch_bytes,
           "fused injection retains staged LDS and admitted global workspace bounds");

    const std::array<std::uint64_t, 4> golden_implementation{11360272515784746702ULL,13592587922752336111ULL,12691105065209337701ULL,744827713494104566ULL};
    const std::array<std::string, 4> golden_compiler{"hip-gfx1151-hyper-mix-v2", "hip-gfx1151-hyper-mix-v3-parallel",
        "hip-gfx1151-hyper-mix-v4-staged", "hip-gfx1151-hyper-mix-v5-fused-injection"};
    const std::string golden_base = "hip.hyper_mix.bf16.h2560.hc4.lr320.r1.c1";
    const std::array<std::string, 4> golden_suffix{"", ".parallel", ".parallel.staged", ".parallel.staged.fused-injection-v5"};
    for (unsigned mode = 0; mode < 4U; ++mode) {
        auto golden_config = config;
        golden_config.parallel = mode >= 1U;
        golden_config.staged_projections = mode >= 2U;
        golden_config.fused_injection = mode >= 3U;
        const auto golden_launcher = strix::make_hip_hyper_mix_launcher("test.golden", &golden_config);
        strix::kernels::KernelRegistry golden_registry;
        strix::register_hip_hyper_mix_kernel_variant(model, request, golden_config, golden_registry);
        expect(golden_launcher.implementation_identity == golden_implementation[mode] &&
                   golden_registry.variants().front().id == golden_base + golden_suffix[mode] &&
                   golden_registry.variants().front().compiler_identity == golden_compiler[mode],
               "v2 through v5 implementation, kernel and compiler goldens remain unchanged");
    }
    auto typed_config = fused_config;
    typed_config.typed_projection = true;
    strix::kernels::KernelRegistry typed_registry;
    strix::register_hip_hyper_mix_kernel_variant(model, request, typed_config, typed_registry);
    expect(typed_registry.variants().front().id != fused_variant.id &&
               typed_registry.variants().front().compiler_identity != fused_variant.compiler_identity &&
               typed_registry.variants().front().resources.lds_bytes == fused_variant.resources.lds_bytes,
           "typed projection changes identity while retaining v5 geometry and LDS bounds");
    const auto typed_golden = strix::make_hip_hyper_mix_launcher("test.typed.golden", &typed_config);
    expect(typed_golden.implementation_identity == 13117369992274356318ULL &&
               typed_registry.variants().front().id == golden_base + ".parallel.staged.fused-injection-v5.typed-q8-f32-v1" &&
               typed_registry.variants().front().compiler_identity == "hip-gfx1151-hyper-mix-v6-typed-projection",
           "typed implementation, kernel and compiler goldens are stable");
    auto prefetch_config = typed_config;
    prefetch_config.prefetch_rows = true;
    strix::kernels::KernelRegistry prefetch_registry;
    strix::register_hip_hyper_mix_kernel_variant(model, request, prefetch_config, prefetch_registry);
    const auto prefetch_golden = strix::make_hip_hyper_mix_launcher("test.prefetch.golden", &prefetch_config);
    expect(prefetch_golden.implementation_identity == 7219967087554914409ULL &&
               prefetch_registry.variants().front().id == golden_base + ".parallel.staged.fused-injection-v5.typed-q8-f32-v1.prefetch-rows8-v2" &&
               prefetch_registry.variants().front().compiler_identity == "hip-gfx1151-hyper-mix-v8-typed-prefetch-rows8" &&
               prefetch_registry.variants().front().resources.lds_bytes == fused_variant.resources.lds_bytes &&
               prefetch_registry.variants().front().resources.scratch_bytes == fused_variant.resources.scratch_bytes,
           "prefetch identity golden is unique and retains admitted resources");
    auto output_config = prefetch_config; output_config.prefetch_q8_output = true;
    strix::kernels::KernelRegistry output_registry;
    strix::register_hip_hyper_mix_kernel_variant(model, request, output_config, output_registry);
    const auto output_launcher = strix::make_hip_hyper_mix_launcher("test.output.prefetch", &output_config);
    expect(output_launcher.implementation_identity != prefetch_golden.implementation_identity &&
               output_registry.variants().front().id == prefetch_registry.variants().front().id + ".prefetch-q8-output-v1" &&
               output_registry.variants().front().compiler_identity == "hip-gfx1151-hyper-mix-v9-q8-output-prefetch8-with-projection-prefetch" &&
               output_registry.variants().front().resources.lds_bytes == fused_variant.resources.lds_bytes &&
               output_registry.variants().front().resources.scratch_bytes == fused_variant.resources.scratch_bytes,
           "output prefetch has distinct identity and unchanged admitted resources");
    auto output_only_config = typed_config; output_only_config.prefetch_q8_output = true;
    strix::kernels::KernelRegistry output_only_registry;
    strix::register_hip_hyper_mix_kernel_variant(model, request, output_only_config, output_only_registry);
    expect(output_only_registry.variants().front().compiler_identity == "hip-gfx1151-hyper-mix-v9-q8-output-prefetch8" &&
               output_only_registry.variants().front().id != output_registry.variants().front().id,
           "output and down producer flags are independent identities");
    for (unsigned invalid = 0; invalid < 2U; ++invalid) {
        auto broken = output_config;
        if (invalid == 0U) broken.typed_projection = false;
        else broken.lowrank = 319U;
        bool rejected = false;
        try { (void)strix::make_hip_hyper_mix_launcher("test.invalid.output", &broken); }
        catch (const std::invalid_argument& e) {
            rejected = std::string_view(e.what()) == (invalid == 0U ?
                "Q8 hyper output prefetch requires typed projection mode" : "Q8 hyper output prefetch requires lowrank divisible by 32");
        }
        expect(rejected, "output producer requires typed mode and Q8 block-aligned lowrank");
    }
    auto invalid_prefetch = prefetch_config; invalid_prefetch.typed_projection = false;
    bool invalid_prefetch_rejected = false;
    try { (void)strix::make_hip_hyper_mix_launcher("invalid.prefetch", &invalid_prefetch); }
    catch (const std::invalid_argument& e) {
        invalid_prefetch_rejected = std::string_view(e.what()) == "hyper row prefetch requires typed projection mode";
    }
    expect(invalid_prefetch_rejected, "prefetch requires typed projection independently of other flags");
    auto invalid_typed = typed_config; invalid_typed.fused_injection = false;
    bool invalid_typed_rejected = false;
    try { (void)strix::make_hip_hyper_mix_launcher("invalid.typed", &invalid_typed); }
    catch (const std::invalid_argument&) { invalid_typed_rejected = true; }
    expect(invalid_typed_rejected, "typed projection requires fused staged parallel mode");
    const auto execution =
        strix::build_serial_decode_execution_plan(
            model);
    const auto bindings =
        strix::bind_serial_execution_plan(
            model,
            execution,
            registry,
            request);

    std::uint32_t expected = 1;
    expected +=
        static_cast<std::uint32_t>(
            model.hidden_layers) *
        2U;

    expect(
        bindings.bound_steps == expected,
        "hyper-mix catalog binds attention, MLP and final mixers");

    strix::CompiledSerialExecution compiled;
    compiled.execution_identity =
        0x6767ULL;
    compiled.execution =
        execution;
    compiled.bindings =
        bindings;

    strix::HipHyperMixLauncherBundle typed_bundle(typed_config);
    strix::ExecutionLauncherRegistry rejected_typed_registry;
    bool missing_typed_resources_rejected = false;
    try { typed_bundle.register_for_compiled_execution(compiled, rejected_typed_registry); }
    catch (const std::invalid_argument&) { missing_typed_resources_rejected = true; }
    expect(missing_typed_resources_rejected,
           "typed bundle refuses registration without complete model preflight resources");

    strix::ModelManifest typed_manifest;
    typed_manifest.tensors.resize(2);
    typed_manifest.tensors[0].ggml_type = 8U;
    typed_manifest.tensors[1].ggml_type = 0U;
    strix::ExecutionTensorAccessPlan typed_access;
    for (const auto& mixer : execution.steps) {
        if (mixer.op != strix::SerialOp::attn_hyper_mix && mixer.op != strix::SerialOp::mlp_hyper_mix &&
            mixer.op != strix::SerialOp::final_hyper_mix) continue;
        typed_access.accesses.push_back({mixer.id, mixer.op, 1U, {0U}});
        if (mixer.op != strix::SerialOp::final_hyper_mix)
            typed_access.accesses.push_back({mixer.id, mixer.op, 3U, {1U}});
    }
    strix::ExecutionRuntimeResources typed_resources;
    typed_resources.model_manifest = &typed_manifest;
    typed_resources.tensor_access = &typed_access;
    auto typed_compiled = compiled;
    typed_compiled.bindings = strix::bind_serial_execution_plan(model, execution, typed_registry, request);
    strix::ExecutionLauncherRegistry valid_typed_registry;
    typed_bundle.register_for_compiled_execution(typed_compiled, valid_typed_registry, &typed_resources);
    expect(valid_typed_registry.find_kernel_launcher(typed_registry.variants().front().id) != nullptr,
           "complete model resources permit typed bundle registration");
    strix::HipHyperMixLauncherBundle prefetch_bundle(prefetch_config);
    strix::ExecutionLauncherRegistry wrong_prefetch_registry;
    bool wrong_prefetch_id_rejected = false;
    try { prefetch_bundle.register_for_compiled_execution(typed_compiled, wrong_prefetch_registry, &typed_resources); }
    catch (const std::invalid_argument&) { wrong_prefetch_id_rejected = true; }
    expect(wrong_prefetch_id_rejected && wrong_prefetch_registry.find_kernel_launcher(prefetch_registry.variants().front().id) == nullptr,
           "typed-v1 binding cannot register under prefetch identity");
    auto output_manifest = typed_manifest;
    strix::ManifestTensor output_up; output_up.ggml_type = 8U;
    output_up.dimensions = {model.hc_lowrank, model.hidden_size * model.hc_count};
    output_manifest.tensors.push_back(output_up);
    auto output_access = typed_access;
    output_access.accesses.clear();
    // Lookup uses lower_bound(step_id, binding_slot). Build each mixer's full
    // slot sequence together; appending all up slots later is not a valid plan.
    for (const auto& mixer : execution.steps) {
        if (mixer.op != strix::SerialOp::attn_hyper_mix && mixer.op != strix::SerialOp::mlp_hyper_mix &&
            mixer.op != strix::SerialOp::final_hyper_mix) continue;
        output_access.accesses.push_back({mixer.id, mixer.op, 1U, {0U}});
        output_access.accesses.push_back({mixer.id, mixer.op, 2U, {2U}});
        if (mixer.op != strix::SerialOp::final_hyper_mix)
            output_access.accesses.push_back({mixer.id, mixer.op, 3U, {1U}});
    }
    for (const auto& mixer : execution.steps) {
        if (mixer.op != strix::SerialOp::attn_hyper_mix && mixer.op != strix::SerialOp::mlp_hyper_mix &&
            mixer.op != strix::SerialOp::final_hyper_mix) continue;
        for (const auto slot : {1U, 2U, 3U}) {
            if (slot == 3U && mixer.op == strix::SerialOp::final_hyper_mix) continue;
            const auto* access = strix::find_execution_tensor_access(output_access, mixer.id, slot);
            const auto expected_tensor = slot == 1U ? 0U : (slot == 2U ? 2U : 1U);
            expect(access != nullptr && access->op == mixer.op &&
                       access->tensor_indices == std::vector<std::uint32_t>{expected_tensor},
                   "every mixer slot positively resolves before output admission controls");
        }
    }
    auto output_resources = typed_resources;
    output_resources.model_manifest = &output_manifest; output_resources.tensor_access = &output_access;
    auto output_compiled = compiled;
    output_compiled.bindings = strix::bind_serial_execution_plan(model, execution, output_registry, request);
    strix::HipHyperMixLauncherBundle output_bundle(output_config);
    const auto& output_id = output_registry.variants().front().id;
    const auto& typed_id = typed_registry.variants().front().id;
    const auto& prefetch_id = prefetch_registry.variants().front().id;
    // A wrong final binding must reject before the first launcher is registered.
    auto late_output_manifest = output_manifest;
    late_output_manifest.tensors.push_back(output_up);
    auto late_output_access = output_access;
    late_output_access.accesses.back().tensor_indices[0] = 3U;
    for (unsigned invalid = 0; invalid < 4U; ++invalid) {
        late_output_manifest.tensors[3] = output_up;
        if (invalid == 0U) late_output_manifest.tensors[3].ggml_type = 0U;
        if (invalid == 1U) late_output_manifest.tensors[3].dimensions[0] = 0U;
        if (invalid == 2U) --late_output_manifest.tensors[3].dimensions[0];
        if (invalid == 3U) --late_output_manifest.tensors[3].dimensions[1];
        auto broken_resources = output_resources;
        broken_resources.model_manifest = &late_output_manifest; broken_resources.tensor_access = &late_output_access;
        strix::ExecutionLauncherRegistry untouched;
        bool rejected = false;
        try { output_bundle.register_for_compiled_execution(output_compiled, untouched, &broken_resources); }
        catch (const std::invalid_argument& e) {
            rejected = std::string_view(e.what()) == "Q8 hyper output model contains a non-Q8_0 or invalid-width up projection";
        }
        expect(rejected && untouched.find_kernel_launcher(output_id) == nullptr &&
                   untouched.find_kernel_launcher(typed_id) == nullptr && untouched.find_kernel_launcher(prefetch_id) == nullptr,
               "wrong final up format/shape rejects atomically before any output/typed/prefetch identity registration");
    }
    strix::ExecutionLauncherRegistry valid_output;
    output_bundle.register_for_compiled_execution(output_compiled, valid_output, &output_resources);
    expect(valid_output.find_kernel_launcher(output_id) != nullptr,
           "all 97 valid up bindings positively register output prefetch");
    for (std::size_t index = typed_compiled.execution.steps.size(); index-- > 0;) {
        const auto op = typed_compiled.execution.steps[index].op;
        if (op != strix::SerialOp::attn_hyper_mix && op != strix::SerialOp::mlp_hyper_mix &&
            op != strix::SerialOp::final_hyper_mix) continue;
        typed_compiled.bindings.bindings[index].kernel_id = golden_base;
        break;
    }
    strix::ExecutionLauncherRegistry atomic_typed_registry;
    bool late_id_rejected = false;
    try { typed_bundle.register_for_compiled_execution(typed_compiled, atomic_typed_registry, &typed_resources); }
    catch (const std::invalid_argument&) { late_id_rejected = true; }
    expect(late_id_rejected && atomic_typed_registry.find_kernel_launcher(typed_registry.variants().front().id) == nullptr,
           "wrong final mixer id rejects before registering the first typed launcher");

    strix::ExecutionLauncherRegistry launchers;
    strix::HipHyperMixLauncherBundle bundle(
        config);
    bundle.register_for_compiled_execution(
        compiled,
        launchers);

    const auto found =
        std::find_if(
            bindings.bindings.begin(),
            bindings.bindings.end(),
            [](const auto& binding) {
                return binding.op ==
                    strix::SerialOp::
                        attn_hyper_mix;
            });
    expect(
        found !=
            bindings.bindings.end() &&
        found->kernel_id.has_value() &&
        launchers.find_kernel_launcher(
            *found->kernel_id) !=
            nullptr,
        "hyper-mix launcher bundle materializes selected mixer id");
}

template <std::uint32_t Width, std::uint32_t Streams, std::uint32_t Lowrank, bool DenseUp>
void run_prefetch_geometry() {
    for (const auto op : {strix::SerialOp::attn_hyper_mix, strix::SerialOp::mlp_hyper_mix, strix::SerialOp::final_hyper_mix})
        for (const bool bf16 : {false, true})
            run_numeric_fixture<TypedProjectionFixture<Width, Streams, Lowrank, DenseUp, true>>(
                op, op != strix::SerialOp::final_hyper_mix, bf16, true);
}

}  // namespace

int main(int argc, char** argv) try {
    const bool prefetch = argc == 2 && std::string_view(argv[1]) == "prefetch";
    const bool output_prefetch = argc == 2 && std::string_view(argv[1]) == "output-prefetch";
    if (argc != 1 && !prefetch && !output_prefetch) throw std::invalid_argument("expected no arguments, prefetch or output-prefetch fixture mode");
    test_catalog_and_bundle();
    if (output_prefetch) {
        const auto run = []<unsigned Width, unsigned Streams, unsigned Lowrank>() {
            for (const auto op : {strix::SerialOp::attn_hyper_mix, strix::SerialOp::mlp_hyper_mix, strix::SerialOp::final_hyper_mix})
                for (const bool bf16 : {false, true})
                    run_numeric_fixture<OutputPrefetchFixture<Width, Streams, Lowrank>>(
                        op, op != strix::SerialOp::final_hyper_mix, bf16, true, true);
        };
        run.template operator()<288, 4, 32>();
        run.template operator()<320, 4, 96>();
        run.template operator()<384, 4, 128>();
        run.template operator()<288, 8, 160>(); // hidden36: the admitted partial-four block.
        run.template operator()<320, 8, 288>();
        run.template operator()<384, 8, 320>();
        run.template operator()<96, 3, 512>();
        run.template operator()<10240, 4, 320>();
    } else if (prefetch) {
        run_prefetch_geometry<288, 4, 320, false>();
        run_prefetch_geometry<320, 4, 320, false>();
        run_prefetch_geometry<384, 4, 320, false>();
        run_prefetch_geometry<288, 8, 32, false>();
        run_prefetch_geometry<320, 8, 32, false>();
        run_prefetch_geometry<384, 8, 32, false>();
        run_prefetch_geometry<288, 3, 32, false>();
        run_prefetch_geometry<384, 3, 32, false>();
        run_prefetch_geometry<160, 2, 1, true>();
        run_prefetch_geometry<288, 3, 3, true>();
        run_prefetch_geometry<320, 4, 7, true>();
        run_prefetch_geometry<320, 4, 8, true>();
        run_prefetch_geometry<384, 6, 9, true>();
        run_prefetch_geometry<320, 4, 10, true>();
        run_prefetch_geometry<288, 3, 11, true>();
        run_prefetch_geometry<288, 4, 12, true>();
        run_prefetch_geometry<320, 5, 13, true>();
        run_prefetch_geometry<384, 6, 14, true>();
        run_prefetch_geometry<448, 7, 15, true>();
        run_prefetch_geometry<512, 8, 16, true>();
        run_prefetch_geometry<96, 3, 505, true>();
        run_prefetch_geometry<96, 3, 511, true>();
        run_prefetch_geometry<96, 3, 512, false>();
        run_prefetch_geometry<10240, 4, 320, false>();
    } else {
    run_numeric_fixture(
        strix::SerialOp::attn_hyper_mix,
        true);
    run_numeric_fixture(
        strix::SerialOp::final_hyper_mix,
        false);
    run_numeric_fixture(strix::SerialOp::attn_hyper_mix, true, true);
    run_numeric_fixture(strix::SerialOp::mlp_hyper_mix, true, true);
    run_numeric_fixture(strix::SerialOp::final_hyper_mix, false, true);
    run_numeric_fixture<DenseTailFixture>(strix::SerialOp::attn_hyper_mix, true);
    run_numeric_fixture<DenseTailFixture>(strix::SerialOp::final_hyper_mix, false);
    run_numeric_fixture<DenseTailFixture>(strix::SerialOp::attn_hyper_mix, true, true);
    run_numeric_fixture<DenseTailFixture>(strix::SerialOp::final_hyper_mix, false, true);

    for (const auto op : {strix::SerialOp::attn_hyper_mix, strix::SerialOp::mlp_hyper_mix, strix::SerialOp::final_hyper_mix}) {
        const bool injection = op != strix::SerialOp::final_hyper_mix;
        for (const bool bf16 : {false, true}) {
            run_numeric_fixture<TypedProjectionFixture<288>>(op, injection, bf16);
            run_numeric_fixture<TypedProjectionFixture<320>>(op, injection, bf16);
            run_numeric_fixture<TypedProjectionFixture<384>>(op, injection, bf16);
        }
    }
    }
    if (failures == 0) {
        std::cout
            << "HIP hyper-mix launcher tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP hyper-mix launcher tests: "
        << error.what() << '\n';
    return 2;
}
