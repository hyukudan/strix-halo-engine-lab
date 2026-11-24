#pragma once
// Calibration input statistics of dense linears, host side, any model and backend (off unless an engine is given a
// MomentSet). An engine hands the rows that feed a set of matrices (one named input, e.g. the normalised attention
// input of a layer) to add(); the set keeps H = sum x x^T over every row it saw, which the calibrated low-bit packer
// (strix/weight/host_lowbit_gptq.hpp) reads. An input seen at several points of the forward (a layer visited more than
// once) accumulates all of them. Each input lists its consumers: the checkpoint names of the matrices it feeds.
//
// File "strix-calib-moments/v1" (little-endian): the text line "strix-calib-moments/v1 COUNT\n", then per input the
// line "input NAME WIDTH ROWS CONSUMER[,CONSUMER...]\n" followed by WIDTH * (WIDTH + 1) / 2 float32 values, the lower
// triangle of H row by row (row i holds H[i][0..i]).
#include "strix/weight/host_lowbit_gptq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace strix::weight::calib {

inline constexpr std::string_view kMomentFileTag = "strix-calib-moments/v1";

struct InputMoments {
    std::string name;
    std::vector<std::string> consumers;
    std::uint32_t width = 0;
    std::uint64_t rows = 0;
    std::vector<double> h;  // width x width, lower triangle filled
};

class MomentSet {
public:
    explicit MomentSet(unsigned threads = 0U)
        : threads_(threads ? threads : std::max(1U, std::thread::hardware_concurrency())) {}

    // Declares (once) or re-finds an input; a second declaration must agree on width and consumers.
    InputMoments& declare(std::string_view name, std::uint32_t width, std::span<const std::string> consumers) {
        if (width == 0U) throw std::invalid_argument("calibration input width is zero");
        const auto found = index_.find(std::string(name));
        if (found != index_.end()) {
            auto& entry = inputs_[found->second];
            if (entry.width != width || !std::equal(entry.consumers.begin(), entry.consumers.end(), consumers.begin(),
                                                    consumers.end()))
                throw std::invalid_argument("calibration input redeclared differently: " + std::string(name));
            return entry;
        }
        index_.emplace(std::string(name), inputs_.size());
        auto& entry = inputs_.emplace_back();
        entry.name = std::string(name);
        entry.consumers.assign(consumers.begin(), consumers.end());
        entry.width = width;
        entry.h.assign(std::size_t{width} * width, 0.0);
        return entry;
    }

    // Adds `rows` rows of width floats (row-major) to every listed input (all of the same width). Each input gets the
    // same fixed-order sum it would get alone.
    void add(std::span<InputMoments* const> targets, const float* x, std::size_t rows) {
        if (targets.empty() || rows == 0U) return;
        const auto width = targets.front()->width;
        for (const auto* t : targets)
            if (t->width != width) throw std::invalid_argument("calibration inputs of different widths");
        if (targets.size() == 1U) {
            lowbit::gptq::accumulate_second_moments(x, rows, width, targets.front()->h, threads_);
        } else {
            scratch_.assign(std::size_t{width} * width, 0.0);
            lowbit::gptq::accumulate_second_moments(x, rows, width, scratch_, threads_);
            for (auto* t : targets)
                for (std::size_t i = 0; i < std::size_t{width}; ++i)
                    for (std::size_t j = 0; j <= i; ++j) t->h[i * width + j] += scratch_[i * width + j];
        }
        for (auto* t : targets) t->rows += rows;
    }

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

    void write(const std::string& path) const {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write calibration moments " + path);
        out << kMomentFileTag << ' ' << inputs_.size() << '\n';
        std::vector<float> row;
        for (const auto& e : inputs_) {
            out << "input " << e.name << ' ' << e.width << ' ' << e.rows << ' ';
            for (std::size_t c = 0; c < e.consumers.size(); ++c) out << (c ? "," : "") << e.consumers[c];
            out << '\n';
            for (std::size_t i = 0; i < e.width; ++i) {
                row.resize(i + 1U);
                for (std::size_t j = 0; j <= i; ++j) row[j] = static_cast<float>(e.h[i * e.width + j]);
                out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
            }
        }
        if (!out) throw std::runtime_error("short write of calibration moments " + path);
    }

    static MomentSet read(const std::string& path) {
        static_assert(sizeof(float) == 4U);
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot read calibration moments " + path);
        std::string line;
        std::getline(in, line);
        std::istringstream head(line);
        std::string tag;
        std::size_t count = 0;
        if (!(head >> tag >> count) || tag != kMomentFileTag) throw std::runtime_error("not a calibration moment file: " + path);
        MomentSet set(1U);
        std::vector<float> row;
        for (std::size_t n = 0; n < count; ++n) {
            std::getline(in, line);
            std::istringstream fields(line);
            std::string word, name, consumers;
            std::uint32_t width = 0;
            std::uint64_t rows = 0;
            if (!(fields >> word >> name >> width >> rows >> consumers) || word != "input" || width == 0U || width > 65536U)
                throw std::runtime_error("malformed calibration input header in " + path);
            std::vector<std::string> list;
            for (std::size_t at = 0; at <= consumers.size();) {
                const auto comma = std::min(consumers.find(',', at), consumers.size());
                if (comma > at) list.push_back(consumers.substr(at, comma - at));
                at = comma + 1U;
            }
            auto& e = set.declare(name, width, list);
            e.rows = rows;
            for (std::size_t i = 0; i < width; ++i) {
                row.resize(i + 1U);
                in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
                if (!in) throw std::runtime_error("truncated calibration moments in " + path);
                for (std::size_t j = 0; j <= i; ++j) e.h[i * width + j] = static_cast<double>(row[j]);
            }
        }
        return set;
    }

private:
    unsigned threads_;
    std::deque<InputMoments> inputs_;  // stable addresses: callers keep declare() results
    std::map<std::string, std::size_t> index_;
    std::vector<double> scratch_;
};

}  // namespace strix::weight::calib
