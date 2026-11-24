#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace strix::core {

// The function a projection store applies to each completed FP32 sum when the projection rounds BF16 (a compile-time
// trait of the store, so a kernel instantiation carries exactly one function; round_stored in core/round_store.hpp):
//   incumbent  attn::group_indexer::bf16_round: a NaN keeps its low payload bits (quieted), every other value rounds
//              to nearest even. The stores of every rounded projection before the separate policy existed.
//   separate   weight::scalar_float::bf16_round, the function of row_transforms::Bf16Round: the store has the bits of
//              a separate Bf16Round pass over the stored FP32 sum for every sum, NaN payloads included.
enum class RoundStore : unsigned char { incumbent, separate };

// Resident BF16 weights [output][input] and tight FP32 output [row][output].
struct Bf16RowProjection {
    const std::uint16_t* weights{};
    unsigned outputs{};
    float* output{};
    bool round_bf16{};  // Round the completed FP32 sum and store it widened to FP32.
    std::size_t row_bytes{};  // Zero uses input_width * sizeof(std::uint16_t).
    // With round_bf16: the stored sum goes through RoundStore::separate (the function of the separate Bf16Round pass)
    // instead of the incumbent RoundStore. The decode (tight rows) epilogue has only the incumbent function.
    bool round_separate{};

    constexpr bool contiguous(unsigned input_width) const {
        return row_bytes == 0U ||
            (row_bytes % sizeof(std::uint16_t) == 0U && row_bytes / sizeof(std::uint16_t) == input_width);
    }
};

// Value class of a dense activation operand, declared by the caller that produced it.
// bf16_exact: every element is a finite FP32 value exactly representable in BF16 (the
// producer stored RN_bf16 of its result, or widened a BF16 tensor).
enum class Bf16Operand : unsigned char { fp32, bf16_exact };

// Activation planes the exact BF16 split needs. The split writes hi = RN_bf16(x),
// mid = RN_bf16(x - hi), lo = RN_bf16(x - hi - mid), and each K16 step of the tall
// chain issues lo, mid, hi into one accumulator that starts at +0. For a BF16-exact
// finite x, mid and lo are +0, so the two leading MMAs add only zero products to an
// accumulator that is never -0: one plane (hi) gives the same bits. This is an EXACT
// schedule of the three-term arithmetic, not a FAST operand class.
constexpr int exact_bf16_planes(Bf16Operand operand) {
    return operand == Bf16Operand::bf16_exact ? 1 : 3;
}

// A shared caller whose plan rounds every element of a dense input to BF16 (`rounded`) passes that operand class
// to a backend dense(input, width, rows, projections, Bf16Operand); a backend without that overload keeps its
// FP32 operand. Both give the same bits (exact_bf16_planes); the declared form only skips the zero planes.
template <class Operations, class Projections>
void dense_declared(Operations& operations, const float* input, unsigned width, unsigned rows,
                    const Projections& projections, bool rounded) {
    if constexpr (requires { operations.dense(input, width, rows, projections, Bf16Operand::fp32); })
        operations.dense(input, width, rows, projections, rounded ? Bf16Operand::bf16_exact : Bf16Operand::fp32);
    else
        operations.dense(input, width, rows, projections);
}

// Verify step rows of the BF16 -> FP32 dense family: one warp reads its weight row once for a tile of up to
// kDenseStepRowTile descriptor rows (each (output, row) keeps the one-row accumulation chain), so a width-W window
// reads every weight W / tile times instead of W times. Grid y = width / tile.
inline constexpr unsigned kDenseStepRowTile = 8U;
constexpr unsigned dense_step_row_tile(unsigned width) {
    return width < kDenseStepRowTile ? width : kDenseStepRowTile;
}

template <auto Limit, class Size, class Status, class Launch>
Status for_each_dense_chunk(Size total, Status success, const Launch& launch) {
    static_assert(Limit > 0);
    constexpr Size limit = Limit;
    for (Size first = 0; first < total;) {
        const Size count = std::min(total - first, limit);
        if (const auto status = launch(first, count); status != success) return status;
        first += count;
    }
    return success;
}

template <std::size_t Limit, class Status, class Launch>
Status for_each_bf16_projection_group(std::span<const Bf16RowProjection> projections, Status success,
                                      const Launch& launch) {
    return for_each_dense_chunk<Limit>(projections.size(), success, [&](std::size_t first, std::size_t count) {
        return launch(projections.subspan(first, count));
    });
}

// Route of one projection of a call whose FP32 rows may be stored once as a BF16 operand image. Every route stores the
// bits the FP32 route stores (the image holds the operand bits the FP32 staging makes of the same values).
enum class ImageRoute : unsigned char { fp32_rows, packed, image };

// One measured geometry of the packed schedule's calibration: a projection of K input values with exactly `outputs`
// outputs and a weight row pitch of pitch_elements (0 = K, contiguous), measured for call heights [min_rows, max_rows].
struct ImageShapeRow {
    unsigned k{};
    unsigned outputs{};
    unsigned pitch_elements{};
    unsigned min_rows{};
    unsigned max_rows{};
};

// Backend tile data of the image routes (not model data): the height from which the packed schedule is faster, its K
// stage and tile width, the packed columns one call needs to pay for its image pass (the image costs rows x K x 6
// bytes; the packed schedule saves a share of each whole tile's FP32-route time; both scale with rows x K), the
// heights [min_rows, max_rows] and the shapes (K, output tiles, pitch) the measurements cover. A call outside any of
// them keeps the incumbent launches (an image call the K32 image tile, an FP32 call its FP32 rows); no shape listed
// means none is measured.
struct ImageRoutePolicy {
    unsigned min_rows{};
    unsigned k_step{};
    unsigned tile_columns{};
    unsigned min_packed_columns{};
    unsigned max_rows{};
    std::span<const ImageShapeRow> shapes{};
};

// Whether the policy's measurements cover a call of this height.
constexpr bool image_rows_measured(const ImageRoutePolicy& policy, unsigned rows) noexcept {
    return rows != 0U && rows >= policy.min_rows && rows <= policy.max_rows;
}

// Whether the packed schedule's calibration covers one projection of a call of this height: a listed row with exactly
// this K, output count and pitch (row_bytes 0 = contiguous) whose own height range holds the call. This is the one
// shape test of both entries that can take the packed schedule (the image entry and the call that stores the image).
constexpr bool image_shape_measured(const ImageRoutePolicy& policy, unsigned input_width, unsigned outputs,
                                    std::size_t row_bytes, unsigned rows) noexcept {
    if (!image_rows_measured(policy, rows) || policy.tile_columns == 0U || outputs == 0U ||
        outputs % policy.tile_columns != 0U)
        return false;
    const auto natural = std::size_t{input_width} * sizeof(std::uint16_t);
    const auto pitch = row_bytes != 0U ? row_bytes : natural;
    for (const auto& row : policy.shapes) {
        const auto measured_pitch = row.pitch_elements != 0U ? std::size_t{row.pitch_elements} * sizeof(std::uint16_t)
                                                              : std::size_t{row.k} * sizeof(std::uint16_t);
        if (row.k == input_width && outputs == row.outputs && pitch == measured_pitch && rows >= row.min_rows &&
            rows <= row.max_rows)
            return true;
    }
    return false;
}

// Which launch an image call takes.
enum class ImageLaunch : unsigned char { k32_image, k64_packed };

// Whether a call on an existing BF16 image takes the packed schedule instead of the image tile: only on the device
// class the policy was measured on, and only when every projection is inside the measured heights and shapes.
constexpr bool image_packed_schedule(const ImageRoutePolicy& policy, bool measured_device, unsigned input_width,
                                     unsigned rows, std::span<const Bf16RowProjection> projections) noexcept {
    if (!measured_device || projections.empty()) return false;
    for (const auto& projection : projections)
        if (!image_shape_measured(policy, input_width, projection.outputs, projection.row_bytes, rows)) return false;
    return true;
}

// Routes of the projections of one call on FP32 rows [rows][input_width] (routes.size() >= projections.size()).
// The incumbent launches the call in groups of GroupMax consecutive projections; group_admits(group) tells whether
// that launch runs the BF16 operand arithmetic the image routes keep (the WMMA tile). A group it refuses keeps every
// projection on the FP32 rows, so its launch is the incumbent one. Inside an admitted group: packed = whole tiles
// within max_outputs and the policy's measured shapes, 16-byte aligned weights, a row pitch of whole 16-byte units no shorter than the row and
// outputs x pitch addressable; image = what image_admits(projection) accepts; fp32_rows otherwise. Returns false,
// every route fp32_rows, when the call keeps its FP32 rows: a height outside [min_rows, max_rows], K not a multiple
// of the stage, or fewer packed columns than min_packed_columns.
template <std::size_t GroupMax, class GroupAdmits, class ImageAdmits>
bool plan_image_routes(const ImageRoutePolicy& policy, unsigned input_width, unsigned rows,
                       std::span<const Bf16RowProjection> projections, unsigned max_outputs,
                       std::span<ImageRoute> routes, const GroupAdmits& group_admits, const ImageAdmits& image_admits) {
    static_assert(GroupMax > 0);
    if (routes.size() < projections.size()) return false;
    std::fill_n(routes.begin(), projections.size(), ImageRoute::fp32_rows);
    if (policy.k_step == 0U || policy.tile_columns == 0U || !image_rows_measured(policy, rows) || input_width == 0U ||
        input_width % policy.k_step != 0U)
        return false;
    const auto natural = std::size_t{input_width} * sizeof(std::uint16_t);
    std::size_t packed_columns = 0;
    for (std::size_t first = 0; first < projections.size(); first += GroupMax) {
        const auto group = projections.subspan(first, std::min(projections.size() - first, GroupMax));
        if (!group_admits(group)) continue;
        for (std::size_t p = first; p < first + group.size(); ++p) {
            const auto& projection = projections[p];
            const auto pitch = projection.row_bytes != 0U ? projection.row_bytes : natural;
            if (projection.weights && projection.output && projection.outputs != 0U &&
                image_shape_measured(policy, input_width, projection.outputs, projection.row_bytes, rows) &&
                projection.outputs <= max_outputs &&
                projection.outputs <= std::numeric_limits<std::size_t>::max() / sizeof(float) / rows &&
                projection.outputs % policy.tile_columns == 0U && pitch >= natural && pitch % 16U == 0U &&
                projection.outputs <= std::numeric_limits<std::size_t>::max() / pitch &&
                reinterpret_cast<std::uintptr_t>(projection.weights) % 16U == 0U) {
                routes[p] = ImageRoute::packed;
                packed_columns += projection.outputs;
            } else if (image_admits(projection)) {
                routes[p] = ImageRoute::image;
            }
        }
    }
    if (packed_columns >= policy.min_packed_columns) return true;
    std::fill_n(routes.begin(), projections.size(), ImageRoute::fp32_rows);
    return false;
}

// A routed call launches its projections by route (every packed run, then image, then FP32 rows), not in the
// incumbent's group order. That keeps the incumbent's bytes only while no projection writes a byte another projection
// writes or reads: every output [rows][outputs] FP32 shares no byte with the FP32 input rows [rows][input_width], with
// another projection's output or with another projection's weights [outputs][pitch]. A call with such an ordered alias
// (for example an FMA group and a later packed group writing the same output buffer) keeps the incumbent sequence.
// input is null for a call that reads only a producer-stored operand image. Sizes that do not fit an address count as
// overlapping.
inline bool image_routes_order_free(const float* input, unsigned input_width, unsigned rows,
                                    std::span<const Bf16RowProjection> projections) noexcept {
    using Address = std::uintptr_t;
    constexpr Address kMax = std::numeric_limits<Address>::max();
    struct Range {
        Address lo, hi;
        bool valid;
    };
    const auto range = [](const void* p, Address count, Address size) {
        const auto lo = reinterpret_cast<Address>(p);
        if (count != 0U && size > kMax / count) return Range{0U, 0U, false};
        const Address bytes = count * size;
        return bytes > kMax - lo ? Range{0U, 0U, false} : Range{lo, lo + bytes, true};
    };
    const auto apart = [](const Range& a, const Range& b) {
        return a.valid && b.valid && (a.hi <= b.lo || b.hi <= a.lo);
    };
    const auto natural = Address{input_width} * sizeof(std::uint16_t);
    const auto in = range(input, Address{rows} * input_width, sizeof(float));
    for (std::size_t i = 0; i < projections.size(); ++i) {
        const auto out = range(projections[i].output, Address{rows} * projections[i].outputs, sizeof(float));
        if (input && !apart(out, in)) return false;
        for (std::size_t j = 0; j < projections.size(); ++j) {
            if (j == i) continue;
            const auto& other = projections[j];
            const auto pitch = other.row_bytes != 0U ? Address{other.row_bytes} : natural;
            if (!apart(out, range(other.output, Address{rows} * other.outputs, sizeof(float))) ||
                !apart(out, range(other.weights, other.outputs, pitch)))
                return false;
        }
    }
    return true;
}

// The launches of a routed call: for each route (packed, image, fp32_rows) the projections of that route in call order,
// in runs of at most Limit that share the rounding class and the row pitch; launch(route, run). An fp32_rows run stays
// inside one incumbent group (Limit consecutive projections, as plan_image_routes groups them), so it never joins a
// projection of a group the incumbent runs on the WMMA tile with one of a group it runs elsewhere.
template <std::size_t Limit, class Status, class Launch>
Status for_each_image_route_run(std::span<const Bf16RowProjection> projections, std::span<const ImageRoute> routes,
                                unsigned input_width, Status success, const Launch& launch) {
    static_assert(Limit > 0);
    const auto natural = std::size_t{input_width} * sizeof(std::uint16_t);
    const auto pitch = [natural](const Bf16RowProjection& p) { return p.row_bytes != 0U ? p.row_bytes : natural; };
    for (const auto route : {ImageRoute::packed, ImageRoute::image, ImageRoute::fp32_rows}) {
        Bf16RowProjection run[Limit]{};
        std::size_t count = 0, group = 0;
        const auto flush = [&]() {
            const auto status = count == 0U ? success : launch(route, std::span<const Bf16RowProjection>(run, count));
            count = 0;
            return status;
        };
        for (std::size_t p = 0; p < projections.size(); ++p) {
            if (routes[p] != route) continue;
            if (count != 0U && (count == Limit || run[0].round_bf16 != projections[p].round_bf16 ||
                                run[0].round_separate != projections[p].round_separate ||
                                pitch(run[0]) != pitch(projections[p]) ||
                                (route == ImageRoute::fp32_rows && p / Limit != group)))
                if (const auto status = flush(); status != success) return status;
            if (count == 0U) group = p / Limit;
            run[count++] = projections[p];
        }
        if (const auto status = flush(); status != success) return status;
    }
    return success;
}

}  // namespace strix::core
