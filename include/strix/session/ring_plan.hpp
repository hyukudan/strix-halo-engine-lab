#pragma once

#include <cstddef>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace strix::session {

enum class RollbackMode { position_rings };

// Validate the decoded value of the TOML rollback field.
[[nodiscard]] constexpr RollbackMode parse_rollback_mode(std::string_view value) {
    if (value == "position_rings") return RollbackMode::position_rings;
    throw std::invalid_argument("rollback must be \"position_rings\"");
}

class RingPlan {
public:
    constexpr RingPlan(std::size_t window, std::size_t taps, std::size_t max_window)
        : window_(window), taps_(taps), max_window_(max_window) {
        if (window == 0) throw std::invalid_argument("RingPlan: zero window");
        if (max_window == 0) throw std::invalid_argument("RingPlan: max_window is zero");
        window_cells_ = checked_add(window, max_window, "RingPlan: window cell count overflow");
        convolution_cells_ = checked_add(taps, max_window, "RingPlan: convolution cell count overflow");
    }

    [[nodiscard]] constexpr std::size_t window() const noexcept { return window_; }
    [[nodiscard]] constexpr std::size_t taps() const noexcept { return taps_; }
    [[nodiscard]] constexpr std::size_t max_window() const noexcept { return max_window_; }
    [[nodiscard]] constexpr std::size_t window_cells() const noexcept { return window_cells_; }
    [[nodiscard]] constexpr std::size_t convolution_cells() const noexcept { return convolution_cells_; }

    [[nodiscard]] constexpr std::size_t window_cell(std::size_t position) const noexcept {
        return position % window_cells_;
    }
    [[nodiscard]] constexpr std::size_t convolution_cell(std::size_t position) const noexcept {
        return position % convolution_cells_;
    }

    [[nodiscard]] constexpr std::size_t staged_end(std::size_t first, std::size_t rows) const {
        if (rows == 0 || rows > max_window_)
            throw std::invalid_argument("RingPlan: rows must be in [1, max_window]");
        return checked_add(first, rows, "RingPlan: staged frontier overflow");
    }

private:
    static constexpr std::size_t checked_add(std::size_t base, std::size_t extra, const char* error) {
        if (extra > std::numeric_limits<std::size_t>::max() - base) throw std::overflow_error(error);
        return base + extra;
    }

    std::size_t window_;
    std::size_t taps_;
    std::size_t max_window_;
    std::size_t window_cells_;
    std::size_t convolution_cells_;
};

// One externally serialized lane, starting at position zero. The caller owns
// the disjoint storage and provides one pair of row payloads per position.
// Committed reads expose positions below the frontier within the live ranges.
template <class WindowRow, class ConvolutionRow>
    requires (std::is_trivially_copyable_v<WindowRow> && std::is_trivially_copyable_v<ConvolutionRow>)
class PositionRingReference {
public:
    PositionRingReference(RingPlan plan, std::span<WindowRow> window, std::span<ConvolutionRow> convolution)
        : plan_(plan), window_(window), convolution_(convolution) {
        if (window.size() != plan_.window_cells() || convolution.size() != plan_.convolution_cells())
            throw std::invalid_argument("RingPlan: storage size does not match ring cells");
    }

    PositionRingReference(const PositionRingReference&) = delete;
    PositionRingReference& operator=(const PositionRingReference&) = delete;

    [[nodiscard]] const RingPlan& plan() const noexcept { return plan_; }
    [[nodiscard]] std::size_t committed() const noexcept { return committed_; }
    [[nodiscard]] std::size_t staged_rows() const noexcept { return staged_rows_; }

    // Stage the contiguous append starting at committed(). Sources are disjoint
    // from the rings. The old live range and new rows together fit each ring.
    void stage(std::span<const WindowRow> window, std::span<const ConvolutionRow> convolution) {
        if (staged_rows_ != 0) throw std::logic_error("RingPlan: append already staged");
        if (window.size() != convolution.size()) throw std::invalid_argument("RingPlan: staged row counts differ");
        (void)plan_.staged_end(committed_, window.size());
        for (std::size_t row = 0; row < window.size(); ++row) {
            const auto position = committed_ + row;
            std::memcpy(window_.data() + plan_.window_cell(position), window.data() + row, sizeof(WindowRow));
            std::memcpy(convolution_.data() + plan_.convolution_cell(position), convolution.data() + row,
                        sizeof(ConvolutionRow));
        }
        staged_rows_ = window.size();
    }

    void commit_staged(std::size_t keep) {
        if (staged_rows_ == 0) throw std::logic_error("RingPlan: no staged append");
        if (keep > staged_rows_) throw std::invalid_argument("RingPlan: keep exceeds staged rows");
        committed_ += keep;
        staged_rows_ = 0;
    }

    void rollback_staged() { commit_staged(0); }

    [[nodiscard]] const WindowRow& window_at(std::size_t position) const {
        require_live(position, plan_.window(), "RingPlan: window position outside committed range");
        return window_[plan_.window_cell(position)];
    }

    [[nodiscard]] const ConvolutionRow& convolution_at(std::size_t position) const {
        require_live(position, plan_.taps(), "RingPlan: convolution position outside committed range");
        return convolution_[plan_.convolution_cell(position)];
    }

private:
    void require_live(std::size_t position, std::size_t retained, const char* error) const {
        if (position >= committed_ || committed_ - position > retained) throw std::out_of_range(error);
    }

    const RingPlan plan_;
    std::span<WindowRow> window_;
    std::span<ConvolutionRow> convolution_;
    std::size_t committed_ = 0;
    std::size_t staged_rows_ = 0;
};

}  // namespace strix::session
