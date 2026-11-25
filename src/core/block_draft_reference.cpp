#include "strix/spec/block_draft_reference.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace strix::spec::reference {
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::invalid_argument(message); }
void valid(const Matrix& m) {
    require(m.cols > 0 && m.rows <= std::numeric_limits<std::size_t>::max() / m.cols &&
            m.data.size() == m.rows * m.cols, "block reference: matrix shape");
    for (double x : m.data) require(std::isfinite(x), "block reference: nonfinite matrix");
}
Matrix add(const Matrix& a, const Matrix& b) {
    require(a.rows == b.rows && a.cols == b.cols, "block reference: residual shape");
    Matrix result = a;
    for (std::size_t i = 0; i < a.data.size(); ++i) result.data[i] += b.data[i];
    return result;
}
void rotary(Matrix& m, std::uint32_t dim, std::uint64_t first, double theta) {
    for (std::size_t row = 0; row < m.rows; ++row)
        for (std::size_t head = 0; head < m.cols / dim; ++head)
            for (std::uint32_t d = 0; d < dim / 2; ++d) {
                const double angle = static_cast<double>(first + row) * std::pow(theta, -2.0 * d / dim);
                const auto lo = head * dim + d, hi = lo + dim / 2;
                const auto a = m(row, lo), b = m(row, hi);
                m(row, lo) = a * std::cos(angle) - b * std::sin(angle);
                m(row, hi) = b * std::cos(angle) + a * std::sin(angle);
            }
}
Matrix head_norm(Matrix m, std::span<const float> gain, double epsilon) {
    const auto rows = m.rows, cols = m.cols;
    require(!gain.empty() && cols % gain.size() == 0, "block reference: head norm shape");
    m.rows *= cols / gain.size(); m.cols = gain.size();
    m = rms(m, gain, epsilon); m.rows = rows; m.cols = cols;
    return m;
}
Matrix join(const Matrix& a, const Matrix& b) {
    require(a.cols == b.cols, "block reference: join shape");
    Matrix out(a.rows + b.rows, a.cols);
    std::copy(a.data.begin(), a.data.end(), out.data.begin());
    std::copy(b.data.begin(), b.data.end(), out.data.begin() + static_cast<std::ptrdiff_t>(a.data.size()));
    return out;
}
}
Matrix::Matrix(std::size_t r, std::size_t c) : rows(r), cols(c) {
    require(c > 0 && r <= std::numeric_limits<std::size_t>::max() / c, "block reference: matrix overflow");
    data.resize(r * c);
}
Matrix linear(const Matrix& x, std::span<const float> w, std::size_t outputs) {
    valid(x);
    require(outputs > 0 && outputs <= std::numeric_limits<std::size_t>::max() / x.cols &&
            w.size() == outputs * x.cols, "block reference: projection shape");
    Matrix y(x.rows, outputs);
    for (std::size_t r = 0; r < x.rows; ++r)
        for (std::size_t o = 0; o < outputs; ++o)
            for (std::size_t i = 0; i < x.cols; ++i) y(r, o) += x(r, i) * w[o*x.cols+i];
    return y;
}
Matrix rms(const Matrix& x, std::span<const float> gain, double epsilon) {
    valid(x);
    require(gain.size() == x.cols && epsilon > 0 && std::isfinite(epsilon), "block reference: norm shape");
    Matrix y(x.rows, x.cols);
    for (std::size_t r = 0; r < x.rows; ++r) {
        double squares = 0;
        for (std::size_t j = 0; j < x.cols; ++j) squares += x(r, j) * x(r, j);
        const double scale = 1 / std::sqrt(squares / static_cast<double>(x.cols) + epsilon);
        for (std::size_t j = 0; j < x.cols; ++j) y(r, j) = x(r, j) * scale * gain[j];
    }
    return y;
}
Matrix convolve(const Matrix& x, const Matrix& dynamic, std::span<const float> base,
                std::uint32_t group_size, std::uint32_t taps, std::uint32_t phase) {
    valid(x); valid(dynamic);
    require(group_size > 0 && x.cols % group_size == 0 && taps > 0 && phase < 2 &&
            dynamic.rows == x.rows && dynamic.cols == 2 * taps * (x.cols / group_size) &&
            base.size() == 2 * taps * x.cols, "block reference: convolution shape");
    Matrix out(x.rows, x.cols);
    const auto groups = x.cols / group_size;
    for (std::size_t r = 0; r < x.rows; ++r)
        for (std::size_t c = 0; c < x.cols; ++c)
            for (std::uint32_t t = 0; t < taps && t <= r; ++t)
                out(r, c) += (base[(phase*taps+t)*x.cols+c] +
                    dynamic(r, (phase*taps+t)*groups+c/group_size)) * x(r-t, c);
    return out;
}

Matrix forward(const weight::BlockDraftConfig& c, const Weights& w, const Matrix& features,
               const Matrix& noise, std::uint64_t first_position) {
    valid(features); valid(noise);
    require(c.group > 0 && c.hidden % c.group == 0 && c.kv_heads > 0 && c.heads % c.kv_heads == 0 &&
            c.head_dim > 0 && c.head_dim % 2 == 0 && c.window > 0 && c.taps == 2 &&
            c.theta > 0 && std::isfinite(c.theta), "block reference: geometry");
    require(features.cols == c.hidden * c.target_layers.size() && noise.cols == c.hidden &&
            noise.rows > 0 && noise.rows <= c.block &&
            first_position <= UINT64_MAX - features.rows - noise.rows, "block reference: input shape/positions");
    for (const auto& [name, shape] : weight::block_draft_shapes(c)) {
        std::size_t n = 1;
        for (auto d : shape) {
            require(d > 0 && n <= std::numeric_limits<std::size_t>::max() / d, "block reference: weight overflow");
            n *= static_cast<std::size_t>(d);
        }
        const auto tensor = w(name);
        require(tensor.size() == n, "block reference: weight shape");
        for (float v : tensor) require(std::isfinite(v), "block reference: nonfinite weight");
    }
    const auto context = rms(linear(features, w("fc.weight"), c.hidden), w("hidden_norm.weight"), c.epsilon);
    Matrix state = noise;
    const auto query_width = std::size_t(c.heads) * c.head_dim;
    const auto kv_width = std::size_t(c.kv_heads) * c.head_dim;
    const auto groups = c.hidden / c.group;
    for (std::uint32_t layer = 0; layer < c.layers; ++layer) {
        const auto p = "layers." + std::to_string(layer) + ".";
        auto norm = rms(state, w(p + "input_layernorm.weight"), c.epsilon);
        const auto dynamic = linear(norm, w(p + "attention_conv.kernel_projection.weight"), 2*c.taps*groups);
        norm = convolve(norm, dynamic, w(p + "attention_conv.base_kernel"), c.group, c.taps, 0);
        auto q = head_norm(linear(norm, w(p + "self_attn.q_proj.weight"), query_width),
                           w(p + "self_attn.q_norm.weight"), c.epsilon);
        const auto kv_input = join(context, norm);
        auto k = head_norm(linear(kv_input, w(p + "self_attn.k_proj.weight"), kv_width),
                           w(p + "self_attn.k_norm.weight"), c.epsilon);
        const auto v = linear(kv_input, w(p + "self_attn.v_proj.weight"), kv_width);
        rotary(q, c.head_dim, first_position + features.rows, c.theta);
        rotary(k, c.head_dim, first_position, c.theta);
        Matrix attended(noise.rows, query_width);
        for (std::size_t r = 0; r < noise.rows; ++r) for (std::uint32_t h = 0; h < c.heads; ++h) {
            std::vector<double> scores(k.rows, -std::numeric_limits<double>::infinity());
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t j = 0; j < k.rows; ++j) {
                const auto qp = features.rows + r;
                const auto distance = qp > j ? qp-j : j-qp;
                if (distance >= c.window) continue; // bidirectional within the draft block
                double dot = 0;
                for (std::uint32_t d = 0; d < c.head_dim; ++d)
                    dot += q(r, h*c.head_dim+d) * k(j, (h/(c.heads/c.kv_heads))*c.head_dim+d);
                scores[j] = dot / std::sqrt(static_cast<double>(c.head_dim));
                maximum = std::max(maximum, scores[j]);
            }
            double sum = 0;
            for (auto& score : scores) { score = std::exp(score - maximum); sum += score; }
            for (std::size_t j = 0; j < k.rows; ++j) for (std::uint32_t d = 0; d < c.head_dim; ++d)
                attended(r, h*c.head_dim+d) += scores[j]/sum * v(j, (h/(c.heads/c.kv_heads))*c.head_dim+d);
        }
        const auto output = linear(attended, w(p + "self_attn.o_proj.weight"), c.hidden);
        state = add(state, convolve(output, dynamic, w(p + "attention_conv.base_kernel"), c.group, c.taps, 1));
        norm = rms(state, w(p + "post_attention_layernorm.weight"), c.epsilon);
        const auto md = linear(norm, w(p + "mlp_conv.kernel_projection.weight"), 2*c.taps*groups);
        norm = convolve(norm, md, w(p + "mlp_conv.base_kernel"), c.group, c.taps, 0);
        auto gate = linear(norm, w(p + "mlp.gate_proj.weight"), c.intermediate);
        const auto up = linear(norm, w(p + "mlp.up_proj.weight"), c.intermediate);
        for (std::size_t i = 0; i < gate.data.size(); ++i)
            gate.data[i] = gate.data[i] / (1 + std::exp(-gate.data[i])) * up.data[i];
        const auto down = linear(gate, w(p + "mlp.down_proj.weight"), c.hidden);
        state = add(state, convolve(down, md, w(p + "mlp_conv.base_kernel"), c.group, c.taps, 1));
    }
    return rms(state, w("norm.weight"), c.epsilon);
}

Candidates select(const weight::BlockDraftConfig& c, const Weights& w, const Matrix& hidden,
                  const Matrix& logits, std::uint32_t anchor, double temperature, std::span<const double> uniforms) {
    valid(hidden); valid(logits);
    require(hidden.rows == logits.rows && hidden.cols == c.hidden && logits.cols <= c.vocab &&
            anchor < logits.cols && c.top_k > 0 && c.top_k <= logits.cols && c.rank > 0 &&
            c.softcap > 0 && std::isfinite(c.softcap) && temperature >= 0 && std::isfinite(temperature) &&
            (temperature == 0 || uniforms.size() == logits.rows), "block selector: geometry/sampling");
    const auto code = linear(hidden, w("candidate_selector.hidden_projection.weight"), c.rank);
    const auto pred = w("candidate_selector.predecessor_codebook");
    const auto succ = w("candidate_selector.successor_codebook");
    require(pred.size() == std::size_t(c.vocab)*c.rank && succ.size() == pred.size(), "block selector: codebook shape");
    Candidates result;
    auto previous = anchor;
    for (std::size_t row = 0; row < logits.rows; ++row) {
        std::vector<std::uint32_t> ids(logits.cols);
        std::iota(ids.begin(), ids.end(), 0U);
        const auto unary_score = [&](std::uint32_t id) {
            return c.post_softcap ? logits(row, id) : c.softcap * std::tanh(logits(row, id)/c.softcap);
        };
        std::partial_sort(ids.begin(), ids.begin()+c.top_k, ids.end(), [&](auto a, auto b) {
            return unary_score(a) != unary_score(b) ? unary_score(a) > unary_score(b) : a < b;
        });
        ids.resize(c.top_k);
        std::vector<double> scores(c.top_k);
        for (std::size_t j = 0; j < ids.size(); ++j) {
            double correction = 0;
            for (std::uint32_t d = 0; d < c.rank; ++d)
                correction += pred[std::size_t(previous)*c.rank+d] * code(row, d) * succ[std::size_t(ids[j])*c.rank+d];
            const auto unary = logits(row, ids[j]);
            scores[j] = c.post_softcap ? c.softcap * std::tanh((unary+correction)/c.softcap) :
                c.softcap * std::tanh(unary/c.softcap) + correction;
            require(std::isfinite(scores[j]), "block selector: nonfinite score");
        }
        std::size_t chosen = 0;
        for (std::size_t j = 1; j < scores.size(); ++j)
            if (scores[j] > scores[chosen] || (scores[j] == scores[chosen] && ids[j] < ids[chosen])) chosen = j;
        std::vector<double> probabilities(scores.size(), 0);
        if (temperature == 0) probabilities[chosen] = 1;
        else {
            require(uniforms[row] >= 0 && uniforms[row] < 1, "block selector: uniform outside [0,1)");
            const auto maximum = scores[chosen];
            double sum = 0;
            for (std::size_t j = 0; j < scores.size(); ++j) { probabilities[j] = std::exp((scores[j]-maximum)/temperature); sum += probabilities[j]; }
            double cumulative = 0;
            chosen = scores.size()-1;
            for (auto& probability : probabilities) probability /= sum;
            for (std::size_t j = 0; j < scores.size(); ++j) {
                cumulative += probabilities[j];
                if (uniforms[row] < cumulative) { chosen = j; break; }
            }
        }
        previous = ids[chosen]; result.path.push_back(previous);
        result.ids.push_back(std::move(ids)); result.probabilities.push_back(std::move(probabilities));
    }
    return result;
}
} // namespace strix::spec::reference
