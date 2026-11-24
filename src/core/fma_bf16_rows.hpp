#pragma once

#include "core/controller_rms_rows.hpp"
#include "core/dense_linear.hpp"
#include "core/dense_row_plan.hpp"
#include "core/f16_tile_gemm.hpp"
#include "core/fma_rows.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <string_view>
#include <type_traits>

// HIP launch boundary: the row-owner body remains in the shared core header. The launch re-checks the admitted shape
// (rank selects the kernel's K16 unit count; a refused shape never reaches a specialization with other strides).
namespace strix::core::controller_rms_rows {
inline hipError_t enqueue(const Args& a, hipStream_t stream) {
    if (!admits(a, 0U)) return hipErrorInvalidValue;
    ::strix::hip::arch::require_current_device(::strix::hip::arch::LaunchRequirement::Rdna3Layout);
    const dim3 grid(grid_blocks(a.rows));
    switch (a.rank / 16U) {
    case 1U: detail::controller_rms_kernel<1U><<<grid, kThreads, 0, stream>>>(a); break;
    case 2U: detail::controller_rms_kernel<2U><<<grid, kThreads, 0, stream>>>(a); break;
    case 3U: detail::controller_rms_kernel<3U><<<grid, kThreads, 0, stream>>>(a); break;
    case 4U: detail::controller_rms_kernel<4U><<<grid, kThreads, 0, stream>>>(a); break;
    default: return hipErrorInvalidValue;
    }
    return hipGetLastError();
}
}  // namespace strix::core::controller_rms_rows

namespace strix::core::fma_rows {

// Small NK projections use the decode sweep. Wide projections stage RNE BF16
// activations and resident BF16 weights in the shared WMMA tile.
// A round_bf16 projection group of at most gemv_rows::max_rows rows runs the dense_linear tight-row body, whose
// store_tight_rows_bf16 epilogue rounds each completed FP32 sum: the value a later Bf16Round pass over the stored sum
// writes (a NaN stays NaN; only its low payload bits may differ). Larger groups keep their separate round.
inline constexpr bool rounds_in_store(unsigned rows) noexcept { return rows <= gemv_rows::max_rows; }

// The launch enqueue_bf16_rows makes for one group of projections on 16-byte aligned FP32 rows: refused, the tight-row
// body (decode rows), the shared BF16 WMMA tile, or the FP32 FMA rows (weights the tile's vector loads cannot read, or a
// grid past its launch limit). One source for enqueue_bf16_rows and for the routed entry, whose image routes stand in
// only for the WMMA tile.
enum class Bf16RowsPath : unsigned char { refused, tight_rows, wmma, fma };

// The store policy of a group (every projection of a group has the same one).
inline constexpr RoundStore round_store_of(const Bf16RowProjection& projection) noexcept {
    return projection.round_separate ? RoundStore::separate : RoundStore::incumbent;
}

// body(std::integral_constant<RoundStore, policy>) for the policy of the group's first projection.
template <class Body>
inline decltype(auto) with_round_store(const Bf16RowProjection& first, Body&& body) {
    if (first.round_separate) return body(std::integral_constant<RoundStore, RoundStore::separate>{});
    return body(std::integral_constant<RoundStore, RoundStore::incumbent>{});
}

inline Bf16RowsPath bf16_rows_path(unsigned input_width, unsigned rows, std::span<const Bf16RowProjection> projections,
                                   unsigned max_outputs) noexcept {
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (rows > std::numeric_limits<unsigned>::max() - 128U || !admits(input_width, rows, 16U) ||
        input_width > max_elements / rows || projections.empty() || projections.size() > kGroupMax)
        return Bf16RowsPath::refused;

    const bool round_bf16 = projections.front().round_bf16;
    const auto natural_row_bytes = std::size_t{input_width} * sizeof(std::uint16_t);
    const auto effective_pitch = [natural_row_bytes](const Bf16RowProjection& projection) {
        return projection.row_bytes != 0U ? projection.row_bytes : natural_row_bytes;
    };
    for (const auto& projection : projections) {
        const auto pitch = effective_pitch(projection);
        if (projection.round_bf16 != round_bf16 || projection.round_separate != projections.front().round_separate ||
            (projection.round_separate && !round_bf16) || !projection.weights || !projection.output || projection.outputs == 0 || projection.outputs > max_outputs ||
            projection.outputs > max_elements / rows ||
            projection.outputs > std::numeric_limits<unsigned>::max() - 128U ||
            pitch < natural_row_bytes || pitch % 16U != 0U ||
            projection.outputs > std::numeric_limits<std::size_t>::max() / pitch)
            return Bf16RowsPath::refused;
    }
    if (rows <= gemv_rows::max_rows)  // the tight-row epilogue has the incumbent store only
        return projections.front().round_separate ? Bf16RowsPath::refused : Bf16RowsPath::tight_rows;
    const auto aligned_weights = [](const Bf16RowProjection& projection) {
        return reinterpret_cast<std::uintptr_t>(projection.weights) % 16U == 0U;
    };
    if (rows <= 65535U * f16_tile_gemm::kBM &&
        input_width <= std::numeric_limits<unsigned>::max() - f16_tile_gemm::kBK &&
        std::ranges::all_of(projections, aligned_weights)) {
        const auto pitch = effective_pitch(projections.front());
        return std::ranges::all_of(projections, [&](const Bf16RowProjection& projection) {
                   return effective_pitch(projection) == pitch;
               }) ? Bf16RowsPath::wmma : Bf16RowsPath::refused;
    }
    // Preserve existing admission for callers whose weights cannot use vector
    // loads or whose row grid exceeds the WMMA launch limit.
    return std::ranges::all_of(projections, [&](const Bf16RowProjection& projection) {
               return projection.contiguous(input_width);
           }) ? Bf16RowsPath::fma : Bf16RowsPath::refused;
}

// store_used (optional) receives the round store the WMMA tile or the FMA rows were launched with (incumbent when the
// call does not round or runs the tight rows).
inline hipError_t enqueue_bf16_rows(const float* input, unsigned input_width, unsigned rows,
                                    std::span<const Bf16RowProjection> projections, unsigned max_outputs,
                                    hipStream_t stream, RoundStore* store_used = nullptr) {
    if (store_used) *store_used = RoundStore::incumbent;
    if (!input || reinterpret_cast<std::uintptr_t>(input) % 16U != 0) return hipErrorInvalidValue;
    const auto path = bf16_rows_path(input_width, rows, projections, max_outputs);
    if (path == Bf16RowsPath::refused) return hipErrorInvalidValue;

    const bool round_bf16 = projections.front().round_bf16;
    const auto natural_row_bytes = std::size_t{input_width} * sizeof(std::uint16_t);
    const auto effective_pitch = [natural_row_bytes](const Bf16RowProjection& projection) {
        return projection.row_bytes != 0U ? projection.row_bytes : natural_row_bytes;
    };
    Group group{};
    group.count = static_cast<unsigned>(projections.size());

    if (path == Bf16RowsPath::tight_rows) {
        // Same-input projections share one launch; each output keeps its single-launch instructions (bitwise).
        static_assert(kGroupMax <= dense_linear::kTightGroupMax);
        dense_linear::View views[dense_linear::kTightGroupMax]{};
        float* outputs[dense_linear::kTightGroupMax]{};
        for (unsigned p = 0; p < group.count; ++p) {
            views[p] = {reinterpret_cast<const std::uint8_t*>(projections[p].weights), input_width,
                        projections[p].outputs, effective_pitch(projections[p])};
            outputs[p] = projections[p].output;
        }
        if (group.count == 1U)
            return round_bf16 ? dense_linear::enqueue_bf16_tight_rows<true>(views[0], input, outputs[0], rows, stream)
                              : dense_linear::enqueue_bf16_tight_rows(views[0], input, outputs[0], rows, stream);
        return round_bf16
            ? dense_linear::enqueue_bf16_tight_rows_group<true>(views, outputs, group.count, input, rows, stream)
            : dense_linear::enqueue_bf16_tight_rows_group(views, outputs, group.count, input, rows, stream);
    }

    if (path == Bf16RowsPath::wmma) {
        f16_tile_gemm::Bf16ProjectionGroup<kGroupMax> wmma_group{};
        const auto pitch = effective_pitch(projections.front());
        for (const auto& projection : projections)
            wmma_group.append({projection.weights, projection.outputs, projection.output, projection.round_bf16});
        with_round_store(projections.front(), [&](auto policy) {
            constexpr RoundStore round = decltype(policy)::value;
            if (store_used) *store_used = round;
            if (pitch == natural_row_bytes)
                f16_tile_gemm::gemm_bf16_projections<round>(input, rows, input_width, wmma_group, stream);
            else
                f16_tile_gemm::gemm_bf16_projections<round>(input, rows, input_width, wmma_group, stream,
                                                            pitch / sizeof(std::uint16_t));
        });
        return hipGetLastError();
    }

    for (unsigned p = 0; p < group.count; ++p) {
        group.w[p] = projections[p].weights;
        group.y[p] = projections[p].output;
        group.n[p] = projections[p].outputs;
    }
    if (round_bf16) {
        with_round_store(projections.front(), [&](auto policy) {
            constexpr RoundStore round = decltype(policy)::value;
            if (store_used) *store_used = round;
            if (admits(input_width, rows, 32U))
                launch_auto<true, round>(input, Weight::BF16, input_width, rows, group, stream);
            else
                launch<128, 64, 1, 16, true, true, round>(input, Weight::BF16, input_width, rows, group, stream);
        });
    } else if (admits(input_width, rows, 32U)) {
        launch_auto(input, Weight::BF16, input_width, rows, group, stream);
    } else {
        launch<128, 64, 1, 16, true>(input, Weight::BF16, input_width, rows, group, stream);
    }
    return hipGetLastError();
}

// Rows whose activations arrive as a BF16 operand image (u16 [rows][input_width], element = the RNE BF16 encoding of
// the FP32 value): the shared WMMA tile reads the image as is, the operand bits its FP32 staging makes of those values,
// so each output keeps the bits enqueue_bf16_rows stores for the FP32 rows; half the activation bytes. The tile's
// prefill shapes only (decode rows keep their FP32 sweep): callers check admits_bf16_image_projections before they
// store the image.
inline constexpr bool admits_bf16_image_rows(unsigned input_width, unsigned rows) noexcept {
    return rows > gemv_rows::max_rows && rows <= 65535U * f16_tile_gemm::kBM && input_width >= f16_tile_gemm::kBK &&
           input_width % f16_tile_gemm::kBK == 0U &&
           input_width <= std::numeric_limits<unsigned>::max() - f16_tile_gemm::kBK;
}

// The image route for these projections, asked before the image is stored: the shape (admits_bf16_image_rows) and
// every weight the image launch reads (contiguous rows, 16-byte aligned, outputs within max_outputs, one rounding
// class). A projection outside it keeps its FP32 rows, where enqueue_bf16_rows serves pitched and unaligned weights.
inline bool admits_bf16_image_projections(unsigned input_width, unsigned rows,
                                          std::span<const Bf16RowProjection> projections,
                                          unsigned max_outputs) noexcept {
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (!admits_bf16_image_rows(input_width, rows) || input_width > max_elements / rows || projections.empty() ||
        projections.size() > kGroupMax)
        return false;
    const bool round_bf16 = projections.front().round_bf16;
    const bool separate = projections.front().round_separate;
    return std::ranges::all_of(projections, [&](const Bf16RowProjection& projection) {
        return projection.round_separate == separate && (!separate || round_bf16) && projection.weights &&
               projection.output && projection.outputs != 0U &&
               projection.outputs <= max_outputs && projection.outputs <= max_elements / rows &&
               projection.contiguous(input_width) &&
               reinterpret_cast<std::uintptr_t>(projection.weights) % 16U == 0U && projection.round_bf16 == round_bf16;
    });
}

// Image rows from this height up take the K64 packed schedule where the shape admits it. It is also the lowest height
// of a routed call: no receipt measures the image routes below it (review r17), so lower calls keep their rows.
inline constexpr unsigned kPackedImageMinRows = 4096U;
// The greatest height any shape covers: one 8,192-row prefill chunk with its absorbed tail (chunk / 8 rows). Each
// shape row carries the heights it was measured at.
inline constexpr unsigned kPackedImageMaxRows = 8192U + 8192U / 8U;

// The geometries the K64 schedule was measured on (gfx1151, one 128 x 128 CTA launch per shape, K64 / K32 image tile
// time). K 2,304, contiguous: 128 outputs 0.93 at 4,096 / 4,309 / 6,144 rows and 1.00 at 8,192 (slower above 8,192:
// 1.01-1.08), so measured to 8,192 only; 1,024 outputs 0.78-0.96 at 4,096..8,192 and 0.87-0.93 at 8,192..9,216; 2,048
// 0.88-0.92; 2,304 0.90-0.91; 3,072 0.91 (0.90-0.98 at 8,236..9,216); 5,120 0.90-0.92; the grouped launch 3,072 + 1,024 +
// 1,024 0.91-0.95 at 4,096..9,216. K 3,072, 2,304 outputs, contiguous: 0.73-0.80 at 4,096..9,216. As a whole routed call
// (image pass + K64 vs FP32 rows): 0.78-0.79 (5,120 outputs and the group), 0.82-0.90 (2,048..3,072), 0.76-0.79 (K 3,072),
// 0.67-0.71 (K 3,072 on a 3,200-element weight row pitch). Every projection of a call must match a row exactly (K,
// outputs, pitch) inside that row's heights; any other width (256, 512 ...), pitch or height keeps the K32 image tile
// or the FP32 rows.
inline constexpr ImageShapeRow kImageShapeRows[] = {
    {2304U, 128U, 0U, 4096U, 8192U},  {2304U, 1024U, 0U, 4096U, 9216U}, {2304U, 2048U, 0U, 4096U, 9216U},
    {2304U, 2304U, 0U, 4096U, 9216U}, {2304U, 3072U, 0U, 4096U, 9216U}, {2304U, 5120U, 0U, 4096U, 9216U},
    {3072U, 2304U, 0U, 4096U, 9216U}, {3072U, 2304U, 3200U, 4096U, 9216U}};

// The image routes of the shared tile on gfx11 (backend data, measured at 8,192 rows, K 2,304: FP32 route 0.150 ms per
// 128-column tile, packed K64 0.109 ms, image pass 0.48 ms -> the image pays from ~12 packed tiles in one call; 16 kept).
inline constexpr ImageRoutePolicy kImageRoutePolicy{kPackedImageMinRows, f16_tile_gemm::ImageProjectionSchedule::kStageK,
                                                    f16_tile_gemm::kBN, 16U * f16_tile_gemm::kBN, kPackedImageMaxRows,
                                                    kImageShapeRows};

// The image routes' profitability data (kImageRoutePolicy) is measured on one device class only: gfx1151 with wave32
// and kMeasuredComputeUnits (40) CUs (Strix Halo). arch is gcnArchName without its feature suffix; compute_units comes
// from the shared compute_units_from (HIP's multiProcessorCount counts WGPs, 2 CUs each, on RDNA). Any other HIP device
// keeps the incumbent launches of enqueue_bf16_rows.
constexpr bool image_routes_measured_device(std::string_view arch, unsigned wave_size, unsigned compute_units) {
    return wave_size == 32U && compute_units == kMeasuredComputeUnits && arch == "gfx1151";
}

// Test-only: a device class to use instead of the current device's properties (the device query below consults it on
// the calling thread, so the public entries see an unqualified device without another GPU). Production never sets one.
struct ImageDeviceClass {
    std::string_view arch;
    unsigned wave_size{};
    unsigned compute_units{};
    int device{-1};  // >= 0: also the current device ordinal the plan admission and consumption see
};
inline const ImageDeviceClass*& image_device_class_override() noexcept {
    static thread_local const ImageDeviceClass* injected = nullptr;
    return injected;
}
class ScopedImageDeviceClass {
public:
    explicit ScopedImageDeviceClass(const ImageDeviceClass& injected) noexcept
        : previous_(image_device_class_override()) {
        image_device_class_override() = &injected;
    }
    ~ScopedImageDeviceClass() { image_device_class_override() = previous_; }
    ScopedImageDeviceClass(const ScopedImageDeviceClass&) = delete;
    ScopedImageDeviceClass& operator=(const ScopedImageDeviceClass&) = delete;

private:
    const ImageDeviceClass* previous_;
};

// The current device ordinal (-1: unknown); the injected one while a ScopedImageDeviceClass with a device is alive.
inline int image_routes_current_device() {
    if (const auto* injected = image_device_class_override(); injected && injected->device >= 0)
        return injected->device;
    int dev = -1;
    return hipGetDevice(&dev) == hipSuccess ? dev : -1;
}

// image_routes_measured_device of the current device (first call per device ordinal, then cached; an unknown device
// or a failed query keeps the incumbent).
inline bool image_routes_device_qualified() {
    if (const auto* injected = image_device_class_override())
        return image_routes_measured_device(injected->arch, injected->wave_size, injected->compute_units);
    constexpr int kSlots = 16;
    static std::atomic<unsigned char> cached[kSlots] = {};  // 0 unknown, 1 qualified, 2 not
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0) return false;
    if (dev < kSlots) {
        const auto known = cached[dev].load(std::memory_order_relaxed);
        if (known != 0U) return known == 1U;
    }
    hipDeviceProp_t prop{};
    bool qualified = false;
    if (hipGetDeviceProperties(&prop, dev) == hipSuccess && prop.multiProcessorCount > 0) {
        std::string_view arch(prop.gcnArchName);
        arch = arch.substr(0, arch.find(':'));
        qualified = image_routes_measured_device(
            arch, static_cast<unsigned>(prop.warpSize),
            compute_units_from(arch, static_cast<unsigned>(prop.multiProcessorCount)));
    }
    if (dev < kSlots) cached[dev].store(qualified ? 1U : 2U, std::memory_order_relaxed);
    return qualified;
}

// launched (optional) receives the launch made: the K64 packed schedule or the K32 image tile; store_used (optional)
// the round store that launch was instantiated with.
inline hipError_t enqueue_bf16_image_rows(const std::uint16_t* input, unsigned input_width, unsigned rows,
                                          std::span<const Bf16RowProjection> projections, unsigned max_outputs,
                                          hipStream_t stream, ImageLaunch* launched = nullptr,
                                          RoundStore* store_used = nullptr) {
    if (!input || reinterpret_cast<std::uintptr_t>(input) % 16U != 0U ||
        !admits_bf16_image_projections(input_width, rows, projections, max_outputs))
        return hipErrorInvalidValue;
    f16_tile_gemm::Bf16ProjectionGroup<kGroupMax> group{};
    for (const auto& projection : projections)
        group.append({projection.weights, projection.outputs, projection.output, projection.round_bf16});
    // The packed-operand core's K64 schedule (same bits) runs only on the measured device class, inside the measured
    // heights and shapes (kImageRoutePolicy); every other call keeps the K32 image tile.
    return with_round_store(projections.front(), [&](auto policy) {
        constexpr RoundStore round = decltype(policy)::value;
        if (store_used) *store_used = round;
        if (image_packed_schedule(kImageRoutePolicy, image_routes_device_qualified(), input_width, rows, projections) &&
            f16_tile_gemm::admits_packed_image_projections(input_width, group)) {
            if (launched) *launched = ImageLaunch::k64_packed;
            return f16_tile_gemm::gemm_bf16_projections_image_packed<round>(input, rows, input_width, group, stream);
        }
        if (launched) *launched = ImageLaunch::k32_image;
        f16_tile_gemm::gemm_bf16_projections_image<round>(input, rows, input_width, group, stream);
        return hipGetLastError();
    });
}

namespace detail {
// One thread per 8 values: image[i] = bf16_encode(input[i]) (RNE, a NaN stays NaN), the operand bits the projection
// tile's FP32 staging (pack2<bf16>) makes of the same values.
static __global__ __launch_bounds__(256) void bf16_operand_image_kernel(const float* input, std::uint16_t* image,
                                                                 std::size_t groups) {
    const std::size_t g = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const auto* in = reinterpret_cast<const float4*>(input) + 2U * g;
    const float4 x = in[0], y = in[1];
    const auto pack = [](float lo, float hi) {
        return std::uint32_t{::strix::weight::scalar_float::bf16_encode(lo)} |
               (std::uint32_t{::strix::weight::scalar_float::bf16_encode(hi)} << 16U);
    };
    reinterpret_cast<uint4*>(image)[g] = make_uint4(pack(x.x, x.y), pack(x.z, x.w), pack(y.x, y.y), pack(y.z, y.w));
}
}  // namespace detail

// The BF16 operand image of contiguous FP32 rows ([rows][width] -> [rows][width] u16) for enqueue_bf16_image_rows,
// which then stores the bits enqueue_bf16_rows stores on the FP32 rows. Both pointers 16-byte aligned, width % 8 == 0,
// the two byte ranges disjoint (blocks run in no order: an image over its own input would depend on the schedule) and
// the grid within the launch limit; anything else is refused before a launch.
inline hipError_t enqueue_bf16_operand_image(const float* input, unsigned width, unsigned rows, std::uint16_t* image,
                                             hipStream_t stream) {
    if (!input || !image || width % 8U != 0U || reinterpret_cast<std::uintptr_t>(input) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(image) % 16U != 0U)
        return hipErrorInvalidValue;
    const std::uint64_t values = std::uint64_t{rows} * width;  // both factors < 2^32: no wrap
    const std::uint64_t groups = values / 8U;
    if (groups == 0U) return hipSuccess;
    const std::uint64_t blocks = (groups + 255U) / 256U;
    if (blocks > std::uint64_t{std::numeric_limits<int>::max()} ||
        values > std::uint64_t{std::numeric_limits<std::uintptr_t>::max() / 4U})
        return hipErrorInvalidValue;
    const auto in_lo = reinterpret_cast<std::uintptr_t>(input), out_lo = reinterpret_cast<std::uintptr_t>(image);
    const auto in_bytes = static_cast<std::uintptr_t>(values) * 4U, out_bytes = static_cast<std::uintptr_t>(values) * 2U;
    if (in_lo < out_lo + out_bytes && out_lo < in_lo + in_bytes) return hipErrorInvalidValue;
    detail::bf16_operand_image_kernel<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(input, image,
                                                                                       static_cast<std::size_t>(groups));
    return hipGetLastError();
}

inline constexpr std::size_t kImageRouteMaxProjections = 8U;

namespace detail {

// The routes of one routed call (false: every route fp32_rows, the call makes the launches enqueue_bf16_rows makes).
// Planned per incumbent group (kGroupMax consecutive projections): only a group enqueue_bf16_rows runs on the BF16
// WMMA tile may take an image route, so a group it runs on the FP32 FMA rows keeps them whole; a call with a group
// enqueue_bf16_rows refuses is not planned (the incumbent launches report it). The FP32 rows are 16-byte aligned (the
// caller checks them).
inline bool plan_bf16_image_call(unsigned input_width, unsigned rows, std::span<const Bf16RowProjection> projections,
                                 unsigned max_outputs, std::span<const std::uint16_t> image,
                                 std::span<ImageRoute> routes) {
    if (!image.data() || projections.size() > kImageRouteMaxProjections || input_width == 0U ||
        input_width % 8U != 0U || image.size() / input_width < rows ||
        reinterpret_cast<std::uintptr_t>(image.data()) % 16U != 0U)
        return false;
    const auto path = [&](std::span<const Bf16RowProjection> group) {
        return bf16_rows_path(input_width, rows, group, max_outputs);
    };
    if (!for_each_bf16_projection_group<kGroupMax>(projections, true, [&](std::span<const Bf16RowProjection> group) {
            return path(group) != Bf16RowsPath::refused;
        }))
        return false;
    return plan_image_routes<kGroupMax>(kImageRoutePolicy, input_width, rows, projections, max_outputs, routes,
        [&](std::span<const Bf16RowProjection> group) { return path(group) == Bf16RowsPath::wmma; },
        [&](const Bf16RowProjection& projection) {
            return admits_bf16_image_projections(input_width, rows, {&projection, 1U}, max_outputs);
        });
}

// The image (rows x input_width u16) is written before the first projection launch and read by the last one: its bytes
// may share no byte with the input rows (when stored), a weight matrix or an output of the call.
inline bool bf16_image_disjoint(const float* input, unsigned input_width, unsigned rows,
                                std::span<const Bf16RowProjection> projections, const std::uint16_t* image) noexcept {
    const auto lo = reinterpret_cast<std::uintptr_t>(image);
    const auto hi = lo + std::uintptr_t{rows} * input_width * sizeof(std::uint16_t);
    const auto apart = [&](const void* p, std::uintptr_t bytes) {
        const auto start = reinterpret_cast<std::uintptr_t>(p);
        return start + bytes <= lo || hi <= start;
    };
    const auto natural = std::size_t{input_width} * sizeof(std::uint16_t);
    if (input && !apart(input, std::uintptr_t{rows} * input_width * sizeof(float))) return false;
    return std::ranges::all_of(projections, [&](const Bf16RowProjection& projection) {
        const auto pitch = projection.row_bytes != 0U ? projection.row_bytes : natural;
        return apart(projection.weights, projection.outputs * pitch) &&
               apart(projection.output, std::uintptr_t{rows} * projection.outputs * sizeof(float));
    });
}

// The launches of a planned call: fp32_rows runs read the FP32 rows, image and packed runs read image (the operand
// image of input for the routed entry; any BF16 rows for a fixture that checks which launch served a projection).
// input is null when a producer stored only the image (every route then reads it; an fp32_rows run is refused).
inline hipError_t run_bf16_image_routes(const float* input, const std::uint16_t* image, unsigned input_width,
                                        unsigned rows, std::span<const Bf16RowProjection> projections,
                                        unsigned max_outputs, std::span<const ImageRoute> routes, hipStream_t stream,
                                        RoundStore* store_used = nullptr) {
    const auto natural = std::size_t{input_width} * sizeof(std::uint16_t);
    return for_each_image_route_run<kGroupMax>(projections, routes, input_width, hipSuccess,
        [&](ImageRoute route, std::span<const Bf16RowProjection> run) {
            if (route == ImageRoute::fp32_rows)
                return input ? enqueue_bf16_rows(input, input_width, rows, run, max_outputs, stream, store_used)
                             : hipErrorNotSupported;
            if (route == ImageRoute::image)
                return enqueue_bf16_image_rows(image, input_width, rows, run, max_outputs, stream, nullptr, store_used);
            f16_tile_gemm::Bf16ProjectionGroup<kGroupMax> group{};
            for (const auto& projection : run)
                group.append({projection.weights, projection.outputs, projection.output, projection.round_bf16});
            const auto pitch = run.front().row_bytes != 0U ? run.front().row_bytes : natural;
            return with_round_store(run.front(), [&](auto policy) {
                if (store_used) *store_used = decltype(policy)::value;
                return f16_tile_gemm::gemm_bf16_projections_image_packed<decltype(policy)::value>(
                    image, rows, input_width, group, stream, pitch / sizeof(std::uint16_t));
            });
        });
}

}  // namespace detail

// enqueue_bf16_rows for every projection of one call on the same FP32 rows (any count, consecutive runs of kGroupMax),
// where a call at prefill height with enough whole-tile columns stores the rows' BF16 operand image once (in image, at
// least rows x input_width u16, 16-byte aligned, sharing no byte with the input, a weight or an output: refused
// otherwise) and runs the packed K64 schedule (pitched weights included), the K32 image tile for the narrow contiguous
// rest and the FP32 rows for what neither reads, inside the groups enqueue_bf16_rows runs on the WMMA tile. Every output
// keeps the bits enqueue_bf16_rows stores; a call the plan keeps on its rows makes the launches enqueue_bf16_rows makes.
// image_used (optional) reports whether the call took the image routes; store_used (optional) the round store of the
// last launch the call made.
inline hipError_t enqueue_bf16_rows_routed(const float* input, unsigned input_width, unsigned rows,
                                           std::span<const Bf16RowProjection> projections, unsigned max_outputs,
                                           std::span<std::uint16_t> image, hipStream_t stream, bool* image_used = nullptr,
                                           RoundStore* store_used = nullptr) {
    ImageRoute routes[kImageRouteMaxProjections]{};
    if (image_used) *image_used = false;
    if (store_used) *store_used = RoundStore::incumbent;
    if (!input || reinterpret_cast<std::uintptr_t>(input) % 16U != 0U || !image_routes_device_qualified() ||
        !image_routes_order_free(input, input_width, rows, projections) ||
        !detail::plan_bf16_image_call(input_width, rows, projections, max_outputs, image, routes))
        return for_each_bf16_projection_group<kGroupMax>(projections, hipSuccess, [&](auto group) {
            return enqueue_bf16_rows(input, input_width, rows, group, max_outputs, stream, store_used);
        });
    if (!detail::bf16_image_disjoint(input, input_width, rows, projections, image.data())) return hipErrorInvalidValue;
    if (image_used) *image_used = true;
    if (const auto status = enqueue_bf16_operand_image(input, input_width, rows, image.data(), stream);
        status != hipSuccess)
        return status;
    return detail::run_bf16_image_routes(input, image.data(), input_width, rows, projections, max_outputs,
                                         std::span<const ImageRoute>(routes, projections.size()), stream, store_used);
}

class Bf16ImageCallPlan;

// Admission of a producer-stored operand image, decided before anything is enqueued (review r17): the routed entry's
// rules (measured device and height, no ordered alias between outputs), every projection of the call reads the image
// (no FP32 run), and the image (rows x input_width u16) shares no byte with a weight or an output of the call nor with
// any FP32 row block [rows][input_width] the producer reads (sources; null entries skipped). Not admitted: nothing may
// be enqueued from the plan; the caller runs its FP32 producer and the routed entry instead.
inline Bf16ImageCallPlan plan_bf16_rows_from_image(unsigned input_width, unsigned rows,
                                                   std::span<const Bf16RowProjection> projections, unsigned max_outputs,
                                                   std::span<const std::uint16_t> image,
                                                   std::span<const float* const> sources);

inline hipError_t enqueue_bf16_rows_from_image(const Bf16ImageCallPlan& plan, std::span<const std::uint16_t> image,
                                               unsigned input_width, unsigned rows,
                                               std::span<const Bf16RowProjection> projections, unsigned max_outputs,
                                               hipStream_t stream);

// The admitted plan of a call whose producer stores the call's operand image itself (plan_bf16_rows_from_image), with
// the call and the device it was admitted for: the consumer runs exactly these routes and refuses any other call or a
// launch while another device is current. Sealed: construction and every field are private, the two friend functions
// are the only code that builds an admitted plan (plan_bf16_rows_from_image, after its checks) or reads one
// (enqueue_bf16_rows_from_image, which launches from the plan's own copy). Copies are inert values; a plan nobody
// admitted (the refusal result) is not admitted. Stream ordering is the caller's: the producer that stores the image and
// the consumer enqueue_bf16_rows_from_image run on one stream (or the caller orders them), the image span stays
// untouched in between, and the consumer's launches follow the producer's on that stream.
class Bf16ImageCallPlan {
public:
    // Whether admission accepted the call: only then may the producer store the image and the consumer run.
    [[nodiscard]] bool admitted() const noexcept { return admitted_; }

private:
    friend Bf16ImageCallPlan plan_bf16_rows_from_image(unsigned, unsigned, std::span<const Bf16RowProjection>,
                                                       unsigned, std::span<const std::uint16_t>,
                                                       std::span<const float* const>);
    friend hipError_t enqueue_bf16_rows_from_image(const Bf16ImageCallPlan&, std::span<const std::uint16_t>, unsigned,
                                                   unsigned, std::span<const Bf16RowProjection>, unsigned, hipStream_t);
    Bf16ImageCallPlan() = default;
    ImageRoute routes_[kImageRouteMaxProjections]{};
    Bf16RowProjection projections_[kImageRouteMaxProjections]{};  // the admitted descriptors: the launches use these
    const std::uint16_t* image_{};
    std::size_t image_size_{};  // the admitted span's length
    unsigned input_width_{};
    unsigned rows_{};
    unsigned max_outputs_{};
    std::size_t count_{};
    int device_{-1};  // the device current at admission
    bool admitted_{};
};

inline Bf16ImageCallPlan plan_bf16_rows_from_image(unsigned input_width, unsigned rows,
                                                   std::span<const Bf16RowProjection> projections, unsigned max_outputs,
                                                   std::span<const std::uint16_t> image,
                                                   std::span<const float* const> sources) {
    ImageRoute routes[kImageRouteMaxProjections]{};
    const int device = image_routes_current_device();
    if (device < 0 || !image_routes_device_qualified() ||
        !image_routes_order_free(nullptr, input_width, rows, projections) ||
        !detail::plan_bf16_image_call(input_width, rows, projections, max_outputs, image, routes) ||
        std::any_of(routes, routes + projections.size(),
                    [](ImageRoute route) { return route == ImageRoute::fp32_rows; }) ||
        !detail::bf16_image_disjoint(nullptr, input_width, rows, projections, image.data()))
        return {};
    for (const float* source : sources)
        if (source && !detail::bf16_image_disjoint(source, input_width, rows, {}, image.data())) return {};
    Bf16ImageCallPlan plan;
    std::copy(routes, routes + kImageRouteMaxProjections, plan.routes_);
    std::copy(projections.begin(), projections.end(), plan.projections_);
    plan.image_ = image.data();
    plan.image_size_ = image.size();
    plan.input_width_ = input_width;
    plan.rows_ = rows;
    plan.max_outputs_ = max_outputs;
    plan.count_ = projections.size();
    plan.device_ = device;
    plan.admitted_ = true;
    return plan;
}

// The launches of an admitted producer-stored image call after the producer stored the image: the routed entry's
// launches without its image pass, the same bits, run from the plan's own descriptors. hipErrorNotSupported, nothing
// launched, for a plan that was not admitted, was admitted for another call (any other image span (address or length),
// shape, output bound or descriptor (weights, width, output, rounding, pitch), even at the same count) or was admitted
// on another device than the current one. The producer and this call are ordered on one stream by the caller.
inline hipError_t enqueue_bf16_rows_from_image(const Bf16ImageCallPlan& plan, std::span<const std::uint16_t> image,
                                               unsigned input_width, unsigned rows,
                                               std::span<const Bf16RowProjection> projections, unsigned max_outputs,
                                               hipStream_t stream) {
    const auto same = [](const Bf16RowProjection& x, const Bf16RowProjection& y) {
        return x.weights == y.weights && x.outputs == y.outputs && x.output == y.output &&
               x.round_bf16 == y.round_bf16 && x.row_bytes == y.row_bytes;
    };
    if (!plan.admitted_ || plan.device_ != image_routes_current_device() || plan.image_ != image.data() ||
        plan.image_size_ != image.size() || plan.input_width_ != input_width || plan.rows_ != rows ||
        plan.max_outputs_ != max_outputs || plan.count_ != projections.size() ||
        plan.count_ > kImageRouteMaxProjections ||
        !std::equal(projections.begin(), projections.end(), plan.projections_, same))
        return hipErrorNotSupported;
    return detail::run_bf16_image_routes(nullptr, plan.image_, plan.input_width_, plan.rows_,
                                         std::span<const Bf16RowProjection>(plan.projections_, plan.count_),
                                         plan.max_outputs_, std::span<const ImageRoute>(plan.routes_, plan.count_), stream);
}

// The [forget, gain] pair of a low-rank controller with the controller as the tile epilogue:
// output[r][n] = controller_mix(current[r][n], message[r][n], forget sum, gain sum), the same bits as
// enqueue_bf16_rows (FP32 sums) followed by the pointwise controller. hipErrorNotSupported where the shared WMMA tile
// does not take the pair (decode rows, unequal widths, rounded sums, unaligned operands): the caller runs the two-step
// form instead.
inline hipError_t enqueue_bf16_controller_rows(const float* input, unsigned input_width, unsigned rows,
                                               std::span<const Bf16RowProjection> gates, const float* current,
                                               const float* message, float* output, hipStream_t stream) {
    constexpr auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    for (const auto& gate : gates)
        if (!gate.contiguous(input_width)) return hipErrorInvalidValue;
    if (gates.size() != 2U || rows <= gemv_rows::max_rows) return hipErrorNotSupported;
    const auto& forget = gates[0];
    const auto& gain = gates[1];
    const auto aligned = [](const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0U; };
    if (forget.outputs != gain.outputs || forget.round_bf16 || gain.round_bf16 || forget.outputs == 0U ||
        rows > 65535U * f16_tile_gemm::kBM || input_width == 0U || input_width % 16U != 0U ||
        input_width > std::numeric_limits<unsigned>::max() - f16_tile_gemm::kBK || !aligned(input) ||
        !aligned(forget.weights) || !aligned(gain.weights) || forget.outputs > max_elements / rows)
        return hipErrorNotSupported;
    if (!current || !message || !output || !forget.weights || !gain.weights) return hipErrorInvalidValue;
    f16_tile_gemm::Bf16ProjectionGroup<2U> group{};
    group.append({forget.weights, forget.outputs, forget.output, forget.round_bf16});
    group.append({gain.weights, gain.outputs, gain.output, gain.round_bf16});
    f16_tile_gemm::gemm_bf16_controller(input, rows, input_width, group, {current, message, output, forget.outputs}, stream);
    return hipGetLastError();
}

// The [forget, gain] pair with the controller epilogue followed by the norm of the conditioned rows
// (ops::norm::enqueue_rms_bf16_rows, parallel form, BF16 scale) as one row-owner launch: output = the norm of
// controller_mix(current, message, forget sum, gain sum), the bits of enqueue_bf16_controller_rows into a separate
// buffer followed by that norm into output. hipErrorNotSupported where the row-owner form does not take the shape
// (controller_rms_rows::admits): the caller runs the two launches.
inline hipError_t enqueue_bf16_controller_rms_rows(const float* input, unsigned input_width, unsigned rows,
                                                   std::span<const Bf16RowProjection> gates, const float* current,
                                                   const float* message, const std::uint16_t* scale, float epsilon,
                                                   float* output, hipStream_t stream) {
    if (gates.size() != 2U) return hipErrorNotSupported;
    const auto& forget = gates[0];
    const auto& gain = gates[1];
    if (!forget.contiguous(input_width) || !gain.contiguous(input_width) || forget.round_bf16 || gain.round_bf16 ||
        forget.outputs != gain.outputs)
        return hipErrorNotSupported;
    const controller_rms_rows::Args args{input,   forget.weights, gain.weights, current, message, scale,
                                         output,  rows,           forget.outputs, input_width, epsilon};
    if (!controller_rms_rows::admits(args, gemv_rows::max_rows)) return hipErrorNotSupported;
    return controller_rms_rows::enqueue(args, stream);
}

}  // namespace strix::core::fma_rows
