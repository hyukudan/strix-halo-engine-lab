// Reader of STRIX_QRTX_STEP_DUMP files (src/hip_known_prefix_window.hip): every read and write buffer of every row of
// every step of a HIP tiled prefill block, used as bit-exact fixtures by the CUDA Qwen prefill tests.
#pragma once

#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace strix::qwen_fixture {

struct Record {
    bool is_read = false;
    std::uint32_t step = 0, op = 0, layer = UINT32_MAX, row = 0, index = 0, rows = 0;
    std::uint64_t position = 0;
    std::string kernel_id;
    std::uint64_t offset = 0, bytes = 0;  // payload location in the file
};

class Dump {
public:
    explicit Dump(const std::string& path) : path_(path) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("fixture missing: " + path);
        for (;;) {
            std::uint32_t h[8];
            std::uint64_t h2[2];
            if (std::fread(h, sizeof(h), 1, f) != 1) break;
            if (std::fread(h2, sizeof(h2), 1, f) != 1) throw std::runtime_error("fixture truncated header");
            if (h[0] != 0x31445351U && h[0] != 0x31445352U) throw std::runtime_error("fixture bad magic");
            Record r;
            r.is_read = h[0] == 0x31445352U;
            r.step = h[1]; r.op = h[2]; r.layer = h[3]; r.row = h[4]; r.index = h[5]; r.rows = h[6];
            r.position = h2[0]; r.bytes = h2[1];
            r.kernel_id.resize(h[7]);
            if (h[7] && std::fread(r.kernel_id.data(), 1, h[7], f) != h[7]) throw std::runtime_error("fixture truncated id");
            r.offset = static_cast<std::uint64_t>(_ftelli64(f));
            _fseeki64(f, static_cast<long long>(r.bytes), SEEK_CUR);
            index_[{r.step, r.is_read, r.index, r.row}] = records_.size();
            records_.push_back(std::move(r));
        }
        std::fclose(f);
    }
    [[nodiscard]] const std::vector<Record>& records() const noexcept { return records_; }
    [[nodiscard]] const Record* find(std::uint32_t step, bool is_read, std::uint32_t index, std::uint32_t row) const {
        const auto it = index_.find({step, is_read, index, row});
        return it == index_.end() ? nullptr : &records_[it->second];
    }
    [[nodiscard]] std::vector<std::uint8_t> load(const Record& r) const {
        std::vector<std::uint8_t> v(static_cast<std::size_t>(r.bytes));
        std::FILE* f = std::fopen(path_.c_str(), "rb");
        if (!f) throw std::runtime_error("fixture reopen");
        _fseeki64(f, static_cast<long long>(r.offset), SEEK_SET);
        const bool ok = v.empty() || std::fread(v.data(), 1, v.size(), f) == v.size();
        std::fclose(f);
        if (!ok) throw std::runtime_error("fixture short payload");
        return v;
    }
    // All rows of one buffer of one step, concatenated in row order (row-major [rows][bytes]).
    [[nodiscard]] std::vector<std::uint8_t> load_rows(std::uint32_t step, bool is_read, std::uint32_t index) const {
        std::vector<std::uint8_t> all;
        for (std::uint32_t row = 0;; ++row) {
            const Record* r = find(step, is_read, index, row);
            if (!r) break;
            const auto v = load(*r);
            all.insert(all.end(), v.begin(), v.end());
        }
        return all;
    }

private:
    std::string path_;
    std::vector<Record> records_;
    std::map<std::tuple<std::uint32_t, bool, std::uint32_t, std::uint32_t>, std::size_t> index_;
};

}  // namespace strix::qwen_fixture
