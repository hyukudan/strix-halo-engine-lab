#include "strix/ds41_moe.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <stdexcept>

namespace strix::ds41 {

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
    RouterOutput out;
    out.scores.resize(static_cast<std::size_t>(rows) * experts);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t e = 0; e < experts; ++e) {
            std::array<float, product_lanes> lanes{};
            for (std::uint32_t i = 0; i < hidden; ++i)
                lanes[i % product_lanes] = lanes[i % product_lanes] + weight[static_cast<std::size_t>(e) * hidden + i] *
                    x[static_cast<std::size_t>(r) * hidden + i];
            for (unsigned d = 16U; d; d /= 2U) {
                std::array<float, product_lanes> next{};
                for (unsigned l = 0; l < product_lanes; ++l) next[l] = lanes[l] + lanes[l ^ d];
                lanes = next;
            }
            out.scores[static_cast<std::size_t>(r) * experts + e] = router_score(lanes[0]);
        }
    out.ids.resize(static_cast<std::size_t>(rows) * top_k);
    out.weights.resize(static_cast<std::size_t>(rows) * top_k);
    for (std::uint32_t r = 0; r < rows; ++r) {
        const auto* scores = out.scores.data() + static_cast<std::size_t>(r) * experts;
        const auto& b = !image_rows.empty() && image_rows[r] ? bias_image : bias;
        std::vector<bool> taken(experts, false);
        for (std::uint32_t k = 0; k < top_k; ++k) {
            std::uint32_t best = experts;
            float best_value = 0.0F;
            for (std::uint32_t e = 0; e < experts; ++e) {
                if (taken[e]) continue;
                const float v = scores[e] + b[e];
                if (best == experts || v > best_value) { best = e; best_value = v; }
            }
            taken[best] = true;
            out.ids[static_cast<std::size_t>(r) * top_k + k] = best;
        }
        float sum = 0.0F;
        for (std::uint32_t k = 0; k < top_k; ++k) sum = sum + scores[out.ids[static_cast<std::size_t>(r) * top_k + k]];
        const float denominator = sum + 1e-20F;
        for (std::uint32_t k = 0; k < top_k; ++k) {
            const float s = scores[out.ids[static_cast<std::size_t>(r) * top_k + k]];
            out.weights[static_cast<std::size_t>(r) * top_k + k] = (top_k > 1U ? s / denominator : s) * route_scale;
        }
    }
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
