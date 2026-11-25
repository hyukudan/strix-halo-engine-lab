#pragma once

#include "inkling2/inkling2_vision.hpp"
#include "strix/serve/json.hpp"
#include "strix/weight/confined_file.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>

namespace strix::inkling2::gate {

inline bool simple_name(const std::string& name) {
    return !name.empty() && name.size() <= 128 && name.front() != '.' &&
           name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-") == std::string::npos;
}

struct Shape {
    unsigned rows, cols, patches, patch_dim, hidden_width, projected_width;
    [[nodiscard]] unsigned tokens() const { return rows * cols; }
    [[nodiscard]] std::size_t pixels() const { return std::size_t{patches - 1U} * patch_dim; }
    [[nodiscard]] std::size_t hidden() const { return std::size_t{patches} * hidden_width; }
    [[nodiscard]] std::size_t projected() const { return std::size_t{tokens()} * projected_width; }

    void check_patches(const vision::PatchGrid& grid, unsigned token_rows, unsigned token_cols) const {
        if (token_rows != rows || token_cols != cols || grid.rows != patches ||
            grid.patch_dim != patch_dim || grid.data.size() != std::size_t{patches} * patch_dim)
            throw std::runtime_error("patch result differs from the reference grid/spec");
        if (std::any_of(grid.data.begin(), grid.data.begin() + patch_dim, [](float v) { return v != 0.0F; }))
            throw std::runtime_error("class patch must have zero input pixels");
    }
    void check_hidden(const vision::EncoderOutput& out) const {
        if (out.rows != patches || out.width != hidden_width || out.hidden.size() != hidden())
            throw std::runtime_error("tower result differs from the reference grid/spec");
    }
    void check_projected(const vision::ProjectedRows& out) const {
        if (out.rows != tokens() || out.width != projected_width || out.bf16.size() != projected())
            throw std::runtime_error("projector result differs from the reference grid/spec");
    }
};

inline Shape shape(const serve::Json& grid, const VisionSpec& spec) {
    const auto count = [&](const char* key) {
        const auto* v = grid.is_object() ? grid.find(key) : nullptr;
        if (!v || !v->is_number()) throw std::runtime_error(std::string("grid.json: missing ") + key);
        const auto n = v->as_int();
        if (n < 1 || n > kVisionDefaultMaxTokens)
            throw std::runtime_error(std::string("grid.json: out-of-range ") + key);
        return static_cast<unsigned>(n);
    };
    const unsigned rows = count("rows"), cols = count("cols"), tokens = rows * cols;
    if (tokens > kVisionDefaultMaxTokens) throw std::runtime_error("grid.json: image exceeds the token limit");
    if (grid.find("tokens") && count("tokens") != tokens) throw std::runtime_error("grid.json: token count differs");
    return {rows, cols, 1U + tokens * spec.merge * spec.merge, 3U * spec.patch * spec.patch,
            spec.width, spec.text_width};
}

inline std::vector<float> read_f32(const weight::ConfinedDirectory& directory, const std::string& name,
                                   std::size_t count) {
    if (count > (std::size_t{1} << 30U) / 4U) throw std::runtime_error("float dump too large: " + name);
    const auto raw = directory.read(name, count * 4U, false, count * 4U);
    const auto* bytes = reinterpret_cast<const weight::Byte*>(raw->data());
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = weight::decode_f32_le(bytes + i * 4U);
        if (!std::isfinite(values[i])) throw std::runtime_error("non-finite dump: " + name);
    }
    return values;
}

struct RowStats {
    double rel_l2 = 0.0, worst_cos = 1.0, max_abs = 0.0;
};

inline RowStats compare_rows(std::span<const float> got, std::span<const float> want, std::size_t width) {
    if (!width || got.empty() || got.size() != want.size() || got.size() % width)
        throw std::runtime_error("comparison requires matching complete rows");
    RowStats r;
    double num = 0.0, den = 0.0;
    for (std::size_t row = 0; row < want.size(); row += width) {
        double dot = 0.0, a2 = 0.0, b2 = 0.0;
        for (std::size_t c = 0; c < width; ++c) {
            const double a = got[row + c], b = want[row + c];
            if (!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error("non-finite comparison operand");
            dot += a * b;
            a2 += a * a;
            b2 += b * b;
            num += (a - b) * (a - b);
            r.max_abs = std::max(r.max_abs, std::fabs(a - b));
        }
        den += b2;
        if (a2 > 0.0 && b2 > 0.0) r.worst_cos = std::min(r.worst_cos, dot / std::sqrt(a2 * b2));
        else if (a2 != b2) r.worst_cos = std::min(r.worst_cos, 0.0);
    }
    r.rel_l2 = den > 0.0 ? std::sqrt(num / den) : (num == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
    return r;
}

inline bool passes(double patch_error, RowStats hidden, RowStats projected, double max_rel_l2) {
    return patch_error == 0.0 && std::isfinite(max_rel_l2) && max_rel_l2 >= 0.0 &&
           hidden.rel_l2 <= max_rel_l2 && projected.rel_l2 <= max_rel_l2;
}

}  // namespace strix::inkling2::gate
