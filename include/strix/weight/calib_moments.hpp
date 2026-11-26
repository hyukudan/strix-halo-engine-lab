#pragma once
// Calibration input statistics of dense linears, host side, any model and backend (off unless an engine is given a
// MomentSet). An engine hands the rows that feed a set of matrices (one named input, e.g. the normalised attention
// input of a layer) to add(); the set keeps H = sum x x^T over every row it saw, which the calibrated low-bit packer
// (strix/weight/host_lowbit_gptq.hpp) reads. An input seen at several points of the forward (a layer visited more than
// once) accumulates all of them. Each input lists its consumers: the checkpoint names of the matrices it feeds.
//
// The set also carries the provenance of its rows (strix/weight/calib_provenance.hpp). Every row belongs to a window:
// the engine registers a calibration window (add_window, which returns its index) and feeds its rows through
// add(window, ...), so each input counts its rows per window. There is no add without a window. write() and read()
// refuse a set in which an input's rows are not all attributed to registered windows; a set without windows cannot be
// packed.
//
// File "strix-calib-moments/v4" (little-endian): the text line "strix-calib-moments/v4 COUNT WINDOWS SHINGLES\n",
// WINDOWS records "window HEX64 TOKENS\n" followed by TOKENS uint32 token ids, SHINGLES uint64 run hashes, then per input the line
// "input NAME WIDTH ROWS ROWS_W0[,ROWS_W1...]|- CONSUMER[,CONSUMER...]\n" (the per-window row counts, one per window,
// summing to ROWS; "-" when there are no windows) followed by WIDTH * (WIDTH + 1) / 2 float32 values, the lower
// triangle of H row by row (row i holds H[i][0..i]). The token ids are the source of the window identities: read()
// recomputes every window digest and every 32-id run hash from them and refuses a file whose stored digests or run
// hashes disagree, so a file cannot carry an identity that its ids do not produce. A v1 file (no provenance), a v2 file
// (rows not attributed to windows) and a v3 file (identities without their ids) are refused.
#include "strix/weight/calib_provenance.hpp"
#include "strix/weight/host_lowbit_gptq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <map>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace strix::weight::calib {

inline constexpr std::string_view kMomentFileTag = "strix-calib-moments/v4";

inline constexpr std::string_view kDiagonalMomentFileTag = "strix-calib-moments/diagonal-v1";
enum class MomentLayout { full, diagonal };

struct InputMoments {
    std::string name;
    std::vector<std::string> consumers;
    std::uint32_t width = 0;
    std::uint64_t rows = 0;
    std::vector<std::uint64_t> window_rows;  // rows per registered window (index = window); sums to rows
    std::vector<double> h;  // full: width x width with lower triangle filled; diagonal: width entries
    MomentLayout layout = MomentLayout::full;
    [[nodiscard]] double diagonal(std::size_t column) const {
        return h[layout == MomentLayout::diagonal ? column : column * width + column];
    }
};

class MomentSet {
public:
    // max_moment_bytes bounds the resident moments (values_). accumulate() also holds one input's transient `next`
    // buffer outside that bound while it folds a batch in, so the peak is the bound plus one input's moments.
    explicit MomentSet(unsigned threads = 0U, MomentLayout layout = MomentLayout::full,
                       std::size_t max_moment_bytes = std::numeric_limits<std::size_t>::max())
        : threads_(threads ? threads : std::max(1U, std::thread::hardware_concurrency())),
          layout_(layout), max_values_(max_moment_bytes / sizeof(double)) {}

    [[nodiscard]] MomentLayout layout() const noexcept { return layout_; }
    [[nodiscard]] std::size_t moment_bytes() const noexcept { return values_ * sizeof(double); }

    // Publish one layer's moments and release its arrays only after a successful
    // write. Registered windows are reusable when replaying the next layer.
    // Every input handle from the flushed layer expires here.
    void flush_layer(const std::string& path) {
        write(path);
        inputs_.clear();
        index_.clear();
        values_ = 0;
    }

    // Declares (once) or re-finds an input; a second declaration must agree on width and consumers.
    InputMoments& declare(std::string_view name, std::uint32_t width, std::span<const std::string> consumers) {
        if (width == 0U || width > 65536U) throw std::invalid_argument("calibration input width out of range");
        const auto found = index_.find(std::string(name));
        if (found != index_.end()) {
            auto& entry = inputs_[found->second];
            if (entry.width != width || !std::equal(entry.consumers.begin(), entry.consumers.end(), consumers.begin(),
                                                    consumers.end()))
                throw std::invalid_argument("calibration input redeclared differently: " + std::string(name));
            return entry;
        }
        const std::size_t stride = layout_ == MomentLayout::diagonal ? 1U : width;
        if (stride > (max_values_ - values_) / width)
            throw std::runtime_error("calibration moment memory limit exceeded declaring " + std::string(name));
        InputMoments entry;
        entry.name = std::string(name);
        entry.consumers.assign(consumers.begin(), consumers.end());
        entry.width = width;
        entry.layout = layout_;
        entry.h.assign(std::size_t{width} * stride, 0.0);
        inputs_.push_back(std::move(entry));
        try {
            index_.emplace(inputs_.back().name, inputs_.size() - 1U);
        } catch (...) {
            inputs_.pop_back();
            throw;
        }
        values_ += inputs_.back().h.size();
        return inputs_.back();
    }

    // Adds `rows` rows of width floats (row-major), all produced while the engine ran registered window `window`, to
    // every listed input (all of the same width). Each input gets the same fixed-order sum it would get alone and
    // counts the rows against the window.
    //
    // Strong guarantee: every allocation (per-window counts of each target, the new sums) happens before the first
    // destination changes; the sums and the counts are then published together with non-throwing operations. A throw
    // leaves every input exactly as it was.
    void add(std::size_t window, std::span<InputMoments* const> targets, const float* x, std::size_t rows) {
        if (window >= provenance_.windows().size())
            throw std::invalid_argument("calibration rows added for a window that is not registered");
        if (targets.empty() || rows == 0U) return;
        const auto width = targets.front()->width;
        const auto values = targets.front()->h.size();
        for (const auto* t : targets)
            if (t->width != width || t->layout != layout_)
                throw std::invalid_argument("calibration inputs of different widths or layouts");
        if (values > max_values_ - values_)
            throw std::runtime_error("calibration moment memory limit exceeded accumulating " + targets.front()->name);
        // Growing counts with zeros changes no serialized byte (write pads the same way).
        for (auto* t : targets)
            if (t->window_rows.size() <= window) t->window_rows.resize(window + 1U, 0U);
        if (prepare_hook_) prepare_hook_(0);
        std::vector<double> next(values, 0.0);
        if (targets.size() == 1U) std::ranges::copy(targets.front()->h, next.begin());
        if (layout_ == MomentLayout::diagonal) {
            for (std::size_t row = 0; row < rows; ++row)
                for (std::size_t col = 0; col < width; ++col) {
                    const double value = x[row * width + col];
                    next[col] += value * value;
                }
        } else {
            lowbit::gptq::accumulate_second_moments(x, rows, width, next, threads_);
        }
        if (prepare_hook_) prepare_hook_(1);
        if (targets.size() == 1U) {
            targets.front()->h.swap(next);
        } else {
            for (auto* t : targets)
                for (std::size_t i = 0; i < values; ++i) t->h[i] += next[i];
        }
        for (auto* t : targets) {
            t->window_rows[window] += rows;
            t->rows += rows;
        }
    }
    // Test-only seam: called during add() preparation (stage 0 after the counts are grown, stage 1 after the new sums
    // are computed), before anything is published. A throwing hook simulates an allocation failure. Null in production.
    void set_prepare_hook(void (*hook)(int stage)) noexcept { prepare_hook_ = hook; }

    // Registers one calibration window (the token ids of one sequence the engine runs while the taps feed add());
    // returns its index.
    std::size_t add_window(std::span<const std::uint32_t> ids) {
        window_ids_.reserve(window_ids_.size() + 1U);
        std::vector<std::uint32_t> copy(ids.begin(), ids.end());
        Provenance next = provenance_;
        next.add_window(ids);
        window_ids_.push_back(std::move(copy));  // cannot throw: reserved
        provenance_ = std::move(next);
        return provenance_.windows().size() - 1U;
    }
    // Throws std::runtime_error unless every input's rows are all attributed to registered windows.
    void check_attribution(const std::string& where) const {
        const auto windows = provenance_.windows().size();
        for (const auto& e : inputs_) {
            std::uint64_t attributed = 0;
            for (const auto r : e.window_rows) attributed += r;
            if (e.window_rows.size() > windows || attributed != e.rows)
                throw std::runtime_error("calibration input " + e.name + " has " + std::to_string(e.rows) + " rows, " +
                                         std::to_string(attributed) + " attributed to registered windows: " + where);
        }
    }
    [[nodiscard]] const Provenance& provenance() const noexcept { return provenance_; }

    [[nodiscard]] const std::deque<InputMoments>& inputs() const noexcept { return inputs_; }
    [[nodiscard]] const InputMoments* find(std::string_view name) const {
        const auto found = index_.find(std::string(name));
        return found == index_.end() ? nullptr : &inputs_[found->second];
    }
    // The input that feeds a matrix (checkpoint name), or null.
    [[nodiscard]] const InputMoments* find_consumer(std::string_view tensor) const {
        for (const auto& entry : inputs_)
            for (const auto& c : entry.consumers)
                if (c == tensor) return &entry;
        return nullptr;
    }

    static void write_ids(std::ostream& out, const std::vector<std::uint32_t>& ids) {
        std::array<char, 4> le{};
        for (const auto id : ids) {
            for (unsigned i = 0; i < 4U; ++i) le[i] = static_cast<char>((id >> (8U * i)) & 0xffU);
            out.write(le.data(), static_cast<std::streamsize>(le.size()));
        }
    }

    void write(const std::string& path) const {
        check_attribution(path);
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write calibration moments " + path);
        static constexpr char digits[] = "0123456789abcdef";
        const auto windows = provenance_.windows().size();
        out << (layout_ == MomentLayout::diagonal ? kDiagonalMomentFileTag : kMomentFileTag) << ' ' << inputs_.size() << ' ' << windows << ' '
            << provenance_.shingles().size() << '\n';
        for (const auto& w : provenance_.windows()) {
            out << "window ";
            for (const auto b : w.digest.bytes) out << digits[b >> 4U] << digits[b & 15U];
            out << ' ' << w.tokens << '\n';
            write_ids(out, window_ids_[static_cast<std::size_t>(&w - provenance_.windows().data())]);
        }
        for (const auto h : provenance_.shingles()) {
            std::array<char, 8> le{};
            for (unsigned i = 0; i < 8U; ++i) le[i] = static_cast<char>((h >> (8U * i)) & 0xffU);
            out.write(le.data(), static_cast<std::streamsize>(le.size()));
        }
        std::vector<float> row;
        for (const auto& e : inputs_) {
            out << "input " << e.name << ' ' << e.width << ' ' << e.rows << ' ';
            if (windows == 0U) out << '-';
            for (std::size_t w = 0; w < windows; ++w)
                out << (w ? "," : "") << (w < e.window_rows.size() ? e.window_rows[w] : std::uint64_t{0});
            out << ' ';
            for (std::size_t c = 0; c < e.consumers.size(); ++c) out << (c ? "," : "") << e.consumers[c];
            out << '\n';
            if (layout_ == MomentLayout::diagonal) {
                row.resize(e.width);
                std::ranges::transform(e.h, row.begin(), [](double value) { return static_cast<float>(value); });
                out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
                continue;
            }
            for (std::size_t i = 0; i < e.width; ++i) {
                row.resize(i + 1U);
                for (std::size_t j = 0; j <= i; ++j) row[j] = static_cast<float>(e.h[i * e.width + j]);
                out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
            }
        }
        out.close();
        if (!out) throw std::runtime_error("short write of calibration moments " + path);
    }

    static MomentSet read(const std::string& path,
                          std::size_t max_moment_bytes = std::numeric_limits<std::size_t>::max()) {
        static_assert(sizeof(float) == 4U);
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot read calibration moments " + path);
        std::string line;
        std::getline(in, line);
        std::istringstream head(line);
        std::string tag;
        std::size_t count = 0, windows = 0, shingles = 0;
        if (!(head >> tag) || tag == "strix-calib-moments/v1")
            throw std::runtime_error("calibration moments without provenance (v1): recapture " + path);
        if (tag == "strix-calib-moments/v3")
            throw std::runtime_error("calibration moments with window identities but not their token ids (v3): recapture " + path);
        if (tag == "strix-calib-moments/v2")
            throw std::runtime_error("calibration moments with rows not attributed to windows (v2): recapture " + path);
        if (!(head >> count >> windows >> shingles) || (tag != kMomentFileTag && tag != kDiagonalMomentFileTag))
            throw std::runtime_error("not a calibration moment file: " + path);
        MomentSet set(1U, tag == kMomentFileTag ? MomentLayout::full : MomentLayout::diagonal, max_moment_bytes);
        std::vector<WindowId> ids(windows);
        for (auto& w : ids) {
            std::getline(in, line);
            std::istringstream fields(line);
            std::string word, digest;
            if (!(fields >> word >> digest >> w.tokens) || word != "window" || digest.size() != 64U || w.tokens == 0U)
                throw std::runtime_error("malformed calibration window line in " + path);
            for (std::size_t i = 0; i < 32U; ++i) {
                const auto nib = [&](char c) -> unsigned {
                    if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
                    if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
                    throw std::runtime_error("malformed calibration window digest in " + path);
                };
                w.digest.bytes[i] = static_cast<std::uint8_t>(nib(digest[2U * i]) * 16U + nib(digest[2U * i + 1U]));
            }
            if (w.tokens > (std::uint64_t{1} << 24U)) throw std::runtime_error("calibration window too long in " + path);
            std::vector<std::uint32_t> wid(static_cast<std::size_t>(w.tokens));
            std::array<unsigned char, 4> le{};
            for (auto& id : wid) {
                in.read(reinterpret_cast<char*>(le.data()), static_cast<std::streamsize>(le.size()));
                if (!in) throw std::runtime_error("truncated calibration window token ids in " + path);
                id = 0;
                for (unsigned i = 0; i < 4U; ++i) id |= std::uint32_t{le[i]} << (8U * i);
            }
            set.window_ids_.push_back(std::move(wid));
        }
        std::vector<std::uint64_t> runs(shingles);
        for (auto& h : runs) {
            std::array<unsigned char, 8> le{};
            in.read(reinterpret_cast<char*>(le.data()), static_cast<std::streamsize>(le.size()));
            if (!in) throw std::runtime_error("truncated calibration provenance in " + path);
            h = 0;
            for (unsigned i = 0; i < 8U; ++i) h |= std::uint64_t{le[i]} << (8U * i);
        }
        // The identities come from the ids: the stored digests and run hashes must be what the ids produce.
        Provenance recomputed;
        for (std::size_t w = 0; w < windows; ++w) {
            recomputed.add_window(set.window_ids_[w]);
            if (recomputed.windows()[w].digest.bytes != ids[w].digest.bytes || recomputed.windows()[w].tokens != ids[w].tokens)
                throw std::runtime_error("calibration window " + std::to_string(w) + " digest does not match its token ids in " + path);
        }
        std::sort(runs.begin(), runs.end());
        runs.erase(std::unique(runs.begin(), runs.end()), runs.end());
        if (runs != recomputed.shingles())
            throw std::runtime_error("calibration run hashes do not match the window token ids in " + path);
        set.provenance_ = std::move(recomputed);
        std::vector<float> row;
        for (std::size_t n = 0; n < count; ++n) {
            std::getline(in, line);
            std::istringstream fields(line);
            std::string word, name, counts, consumers;
            std::uint32_t width = 0;
            std::uint64_t rows = 0;
            if (!(fields >> word >> name >> width >> rows >> counts >> consumers) || word != "input" || width == 0U || width > 65536U)
                throw std::runtime_error("malformed calibration input header in " + path);
            std::vector<std::string> list;
            for (std::size_t at = 0; at <= consumers.size();) {
                const auto comma = std::min(consumers.find(',', at), consumers.size());
                if (comma > at) list.push_back(consumers.substr(at, comma - at));
                at = comma + 1U;
            }
            if (set.find(name)) throw std::runtime_error("duplicate calibration input " + name + " in " + path);
            auto& e = set.declare(name, width, list);
            e.rows = rows;
            if (counts != "-") {
                for (std::size_t at = 0; at <= counts.size();) {
                    const auto comma = std::min(counts.find(',', at), counts.size());
                    const auto field = counts.substr(at, comma - at);
                    if (field.empty() || field.size() > 19U || field.find_first_not_of("0123456789") != std::string::npos)
                        throw std::runtime_error("malformed calibration window row counts in " + path);
                    e.window_rows.push_back(std::stoull(field));
                    at = comma + 1U;
                }
            }
            if (e.window_rows.size() != (counts == "-" ? std::size_t{0} : windows))
                throw std::runtime_error("calibration input " + name + " counts rows for the wrong number of windows in " + path);
            if (set.layout_ == MomentLayout::diagonal) {
                row.resize(width);
                in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
                if (!in) throw std::runtime_error("truncated diagonal calibration moments in " + path);
                std::ranges::copy(row, e.h.begin());
                continue;
            }
            for (std::size_t i = 0; i < width; ++i) {
                row.resize(i + 1U);
                in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
                if (!in) throw std::runtime_error("truncated calibration moments in " + path);
                for (std::size_t j = 0; j <= i; ++j) e.h[i * width + j] = static_cast<double>(row[j]);
            }
        }
        set.check_attribution(path);
        return set;
    }

private:
    unsigned threads_;
    MomentLayout layout_;
    std::size_t max_values_, values_ = 0;
    Provenance provenance_;
    std::vector<std::vector<std::uint32_t>> window_ids_;  // token ids of each registered window (index = window)
    void (*prepare_hook_)(int) = nullptr;
    std::deque<InputMoments> inputs_;  // stable addresses: callers keep declare() results
    std::map<std::string, std::size_t> index_;
};

// Pack-time guard: a fit or proxy input that no attributed row fed carries a zero covariance, which would make the
// fit a no-op or the proxy error vanish. Throws std::runtime_error naming the input and its role.
inline void require_rows(const InputMoments& input, const std::string& role) {
    std::uint64_t attributed = 0;
    for (const auto r : input.window_rows) attributed += r;
    if (input.rows == 0U || attributed == 0U)
        throw std::runtime_error(role + " statistics input " + input.name + " has no attributed rows: refused");
}

}  // namespace strix::weight::calib
