#include "strix/ds41_moe.hpp"
#include "strix/moe/route_project.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <stdexcept>

namespace strix::ds41 {

namespace {
// The DS41 router: sqrt(softplus) scores, selection bias, serial float keys, selection order, renormalised, and a
// single pick keeps its raw score. Experts, top-k and scale are the checkpoint's, read at load (reference_router;
// HIP route_block_topk in hip_ds41_formats.hip; CUDA ds41_core_moe.cu).
constexpr strix::moe::RoutePolicy route_policy(std::uint32_t experts, std::uint32_t top_k, float scale) {
    return {strix::moe::RouteScore::sqrt_softplus, true, strix::moe::RouteKey::float_serial,
            strix::moe::RouteOrder::selection, experts, top_k, 1U, 1U, true, false, 1e-20F, scale};
}
}  // namespace

WindowRoutes build_window_routes(std::span<const std::uint32_t> ids, std::uint32_t rows, std::uint32_t top_k, std::uint32_t split) {
    if (rows == 0U || top_k == 0U || ids.size() != static_cast<std::size_t>(rows) * top_k)
        throw std::invalid_argument("ds41 window routes need rows * top_k expert ids");
    std::map<std::uint32_t, std::vector<std::uint32_t>> by_expert; // expert -> slots (ascending row)
    WindowRoutes routes;
    routes.order.resize(ids.size());
    routes.low_counts.resize(rows);
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::vector<std::pair<std::uint32_t, std::uint32_t>> row; // (expert, slot)
        for (std::uint32_t k = 0; k < top_k; ++k) {
            const auto slot = r * top_k + k, expert = ids[slot];
            for (const auto& [seen, unused] : row)
                if (seen == expert) throw std::invalid_argument("ds41 window routes: an expert repeats within a row");
            row.emplace_back(expert, slot);
            by_expert[expert].push_back(slot);
        }
        std::sort(row.begin(), row.end());
        std::uint32_t low = 0;
        for (std::uint32_t k = 0; k < top_k; ++k) {
            routes.order[r * top_k + k] = row[k].second;
            low += row[k].first < split ? 1U : 0U;
        }
        routes.low_counts[r] = low;
    }
    routes.begins.push_back(0);
    for (const auto& [expert, slots] : by_expert) {
        routes.experts.push_back(expert);
        for (const auto slot : slots) { routes.rows.push_back(slot / top_k); routes.slots.push_back(slot); }
        routes.begins.push_back(static_cast<std::uint32_t>(routes.slots.size()));
        routes.max_rows = std::max(routes.max_rows, static_cast<std::uint32_t>(slots.size()));
    }
    return routes;
}

WindowRoutes restrict_routes(const WindowRoutes& routes, std::uint32_t first, std::uint32_t end) {
    WindowRoutes out;
    out.order = routes.order;
    out.low_counts = routes.low_counts;
    out.begins.push_back(0);
    for (std::size_t j = 0; j + 1U < routes.begins.size(); ++j) {
        if (routes.experts[j] < first || routes.experts[j] >= end) continue;
        out.experts.push_back(routes.experts[j]);
        for (auto i = routes.begins[j]; i < routes.begins[j + 1U]; ++i) {
            out.rows.push_back(routes.rows[i]);
            out.slots.push_back(routes.slots[i]);
        }
        out.begins.push_back(static_cast<std::uint32_t>(out.slots.size()));
        out.max_rows = std::max(out.max_rows, routes.begins[j + 1U] - routes.begins[j]);
    }
    return out;
}

RouterOutput reference_router(std::span<const float> x, std::uint32_t rows, std::uint32_t hidden, std::span<const float> weight,
                              std::uint32_t experts, std::span<const float> bias, std::span<const float> bias_image,
                              std::span<const std::uint8_t> image_rows, std::uint32_t top_k, float route_scale) {
    if (rows == 0U || hidden == 0U || experts == 0U || top_k == 0U || top_k > experts || x.size() < static_cast<std::size_t>(rows) * hidden ||
        weight.size() < static_cast<std::size_t>(experts) * hidden || bias.size() < experts ||
        (!image_rows.empty() && (image_rows.size() < rows || bias_image.size() < experts)))
        throw std::invalid_argument("ds41 router inputs are inconsistent");
    // Scores, selection and weights: the shared projected router (include/strix/moe/route_project.hpp: unfused lane
    // chain, router_score, per-row bias class, policy from the model description).
    struct Math {
        static float exp(float v) { return declared_exp(v); }
        static float log1p(float v) { return declared_log1p(v); }
        static float sqrt(float v) { return declared_sqrt(v); }
    };
    RouterOutput out;
    out.scores.resize(static_cast<std::size_t>(rows) * experts);
    out.ids.resize(static_cast<std::size_t>(rows) * top_k);
    out.weights.resize(static_cast<std::size_t>(rows) * top_k);
    strix::moe::route_project_rows_ref<strix::moe::RouteLaneDot<strix::moe::RouteAccMulAdd, product_lanes>, Math>(
        route_policy(experts, top_k, route_scale), x, rows, hidden, weight, bias, bias_image, image_rows, out.scores,
        out.ids, out.weights);
    return out;
}

std::vector<float> reference_half_sum(const WindowRoutes& routes, std::span<const float> weights, std::span<const float> raw,
                                      std::uint32_t rows, std::uint32_t top_k, std::uint32_t hidden, unsigned half) {
    std::vector<float> out(static_cast<std::size_t>(rows) * hidden, 0.0F);
    for (std::uint32_t r = 0; r < rows; ++r) {
        const auto low = routes.low_counts[r];
        const auto first = half == 0U ? 0U : low, end = half == 0U ? low : top_k;
        for (std::uint32_t o = 0; o < hidden; ++o) {
            float sum = 0.0F;
            for (auto k = first; k < end; ++k) {
                const std::size_t slot = routes.order[r * top_k + k];
                sum = sum + weights[slot] * raw[slot * hidden + o];
            }
            out[static_cast<std::size_t>(r) * hidden + o] = sum;
        }
    }
    return out;
}

float product(WeightFormat format, std::span<const std::uint8_t> row, const A8Activation& x, std::size_t width) {
    if (format == WeightFormat::iq2_xxs) return product_iq2_xxs(row, x, width);
    if (format == WeightFormat::q2_k) return product_q2_k(row, x, width);
    if (format == WeightFormat::q8_0) return product_q8_0(row, x, width);
    return product_mxfp4(row, x, width);
}

WindowReference reference_window(const ExpertTensors& t, const std::vector<A8Activation>& x, const WindowRoutes& routes,
                                 std::span<const float> weights, std::span<const float> shared, std::uint32_t top_k) {
    const auto slots = weights.size(), rows = slots / top_k;
    if (rows == 0U || rows * top_k != slots || x.size() < rows || shared.size() != rows * t.hidden)
        throw std::invalid_argument("ds41 window reference inputs are inconsistent");
    const auto gate_row = t.gate_row_bytes(), down_row = t.down_row_bytes();
    const auto gate_stride = gate_row * t.inter, down_stride = down_row * t.hidden;
    WindowReference out;
    out.gate.assign(slots * t.inter, 0.0F); out.up.assign(slots * t.inter, 0.0F);
    out.activation.resize(slots); out.raw.assign(slots * t.hidden, 0.0F);
    for (std::size_t j = 0; j + 1U < routes.begins.size(); ++j) {
        const std::uint64_t expert = routes.experts[j];
        for (auto i = routes.begins[j]; i < routes.begins[j + 1U]; ++i) {
            const auto row = routes.rows[i], slot = routes.slots[i];
            std::vector<float> act(t.inter);
            for (std::uint32_t o = 0; o < t.inter; ++o) {
                const auto at = expert * gate_stride + o * gate_row;
                const float g = product(t.gate_up_format, t.gate.subspan(at, gate_row), x[row], t.hidden);
                const float u = product(t.gate_up_format, t.up.subspan(at, gate_row), x[row], t.hidden);
                out.gate[slot * t.inter + o] = g; out.up[slot * t.inter + o] = u;
                act[o] = swiglu_clamped(g, u, t.limit);
            }
            out.activation[slot] = quantize_a8(act);
            for (std::uint32_t o = 0; o < t.hidden; ++o)
                out.raw[slot * t.hidden + o] = product(t.down_format, t.down.subspan(expert * down_stride + o * down_row, down_row),
                                                       out.activation[slot], t.inter);
        }
    }
    out.y.assign(rows * t.hidden, 0.0F);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::uint32_t o = 0; o < t.hidden; ++o) {
            float low = 0.0F, high = 0.0F;
            for (std::uint32_t k = 0; k < top_k; ++k) {
                const auto slot = routes.order[r * top_k + k];
                const float term = weights[slot] * out.raw[slot * t.hidden + o];
                if (k < routes.low_counts[r]) low = low + term;
                else high = high + term;
            }
            out.y[r * t.hidden + o] = (shared[r * t.hidden + o] + low) + high;
        }
    return out;
}

} // namespace strix::ds41
