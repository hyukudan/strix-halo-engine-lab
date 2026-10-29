#pragma once

#include "strix/serial_request.hpp"
#include "strix/model_content_hash.hpp"

#include <bit>
#include <charconv>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

namespace strix::decode_cli {

class BitcalcMeasurement {
public:
    BitcalcMeasurement() {
        const char* out = std::getenv("STRIX_BITCALC_LOGITS_OUT");
        const char* force = std::getenv("STRIX_BITCALC_FORCE_TOKENS");
        enabled_ = out || force;
        if ((out && !*out) || (force && !*force))
            throw std::invalid_argument("STRIX_BITCALC paths must be nonempty");
        if (out) output_path_ = out;
        if (force) force_path_ = force;
    }
    ~BitcalcMeasurement() { if (output_) std::fclose(output_); }
    BitcalcMeasurement(const BitcalcMeasurement&) = delete;
    BitcalcMeasurement& operator=(const BitcalcMeasurement&) = delete;

    bool enabled() const noexcept { return enabled_; }
    std::uint64_t forced_steps() const noexcept { return forced_steps_; }
    const std::string& force_tokens_sha256() const noexcept { return force_sha256_; }

    void prepare(std::uint64_t vocab, std::uint64_t prompt, std::uint64_t steps) {
        if (!enabled_) return;
        if (prepared_) throw std::logic_error("STRIX_BITCALC already prepared");
        if (!vocab || vocab > UINT32_MAX || !prompt || !steps || prompt > UINT32_MAX ||
            steps > UINT32_MAX || steps - 1U > UINT32_MAX - (prompt - 1U))
            throw std::invalid_argument("STRIX_BITCALC record dimensions exceed uint32");
        const auto row_bytes = 24U + 4U * vocab;
        if (steps > UINT64_MAX / row_bytes)
            throw std::overflow_error("STRIX_BITCALC capture byte ceiling overflow");
        byte_ceiling_ = row_bytes * steps;
        if (!force_path_.empty()) load_forcing(vocab, steps);
        if (!output_path_.empty()) {
            partial_path_ = output_path_ + ".partial";
            require_absent(output_path_);
            require_absent(partial_path_);
            output_ = std::fopen(partial_path_.c_str(), "wbx");
            if (!output_) throw std::runtime_error("cannot exclusively create STRIX_BITCALC_LOGITS_OUT.partial");
        }
        prepared_ = true;
    }

    void finish() {
        if (!enabled_) return;
        if (!prepared_ || finished_) throw std::logic_error("STRIX_BITCALC invalid finish");
        finished_ = true;
        if (!output_) return;
        const int flushed = std::fflush(output_);
        const int closed = std::fclose(output_);
        output_ = nullptr;
        if (flushed != 0 || closed != 0 || write_failed_)
            throw std::runtime_error("cannot finalize STRIX_BITCALC_LOGITS_OUT.partial");
        require_absent(output_path_);
        std::filesystem::rename(partial_path_, output_path_);
    }

    GreedyEmissionHook hook() {
        if (!enabled_) return {};
        return [this](std::uint64_t step, std::uint64_t position, std::uint32_t argmax,
                      std::span<const float> logits) {
            if (!prepared_ || finished_) throw std::logic_error("STRIX_BITCALC selection outside execution");
            const bool forced = step < forced_tokens_.size();
            const auto emitted = forced ? forced_tokens_[static_cast<std::size_t>(step)] : argmax;
            static_assert(std::endian::native == std::endian::little && sizeof(float) == 4 &&
                std::numeric_limits<float>::is_iec559);
            if (output_) {
                if (logits.size() > UINT32_MAX || step > UINT32_MAX || position > UINT32_MAX)
                    throw std::invalid_argument("STRIX_BITCALC record dimensions exceed uint32");
                const auto bytes = 24U + 4U * static_cast<std::uint64_t>(logits.size());
                if (bytes > byte_ceiling_ - bytes_written_) {
                    write_failed_ = true;
                    throw std::runtime_error("STRIX_BITCALC capture byte ceiling exceeded");
                }
                const std::uint32_t header[]{0x42434C47U, static_cast<std::uint32_t>(step),
                    static_cast<std::uint32_t>(position), argmax, emitted, static_cast<std::uint32_t>(logits.size())};
                bytes_written_ += bytes;
                if (std::fwrite(header, 1U, sizeof(header), output_) != sizeof(header) ||
                    std::fwrite(logits.data(), 1U, logits.size_bytes(), output_) != logits.size_bytes()) {
                    write_failed_ = true;
                    throw std::runtime_error("cannot write STRIX_BITCALC_LOGITS_OUT.partial");
                }
            }
            if (forced) ++forced_steps_;
            return emitted;
        };
    }

private:
    static void require_absent(const std::string& path) {
        if (std::filesystem::symlink_status(path).type() != std::filesystem::file_type::not_found)
            throw std::invalid_argument("STRIX_BITCALC_LOGITS_OUT requires fresh destination: " + path);
    }
    void load_forcing(std::uint64_t vocab, std::uint64_t steps) {
        if (!std::filesystem::is_regular_file(std::filesystem::symlink_status(force_path_)))
            throw std::invalid_argument("STRIX_BITCALC_FORCE_TOKENS requires a regular file");
        if (steps > (UINT64_MAX - 4096U) / 32U)
            throw std::overflow_error("STRIX_BITCALC forcing byte ceiling overflow");
        const auto limit = 32U * steps + 4096U;
        const auto size = std::filesystem::file_size(force_path_);
        if (size > limit || size > std::numeric_limits<std::size_t>::max() ||
            size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()))
            throw std::invalid_argument("STRIX_BITCALC_FORCE_TOKENS file exceeds byte ceiling");
        std::ifstream input(force_path_, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open STRIX_BITCALC_FORCE_TOKENS");
        std::string text(static_cast<std::size_t>(size), '\0');
        if (!input.read(text.data(), static_cast<std::streamsize>(size)) || input.peek() != std::char_traits<char>::eof() || input.bad())
            throw std::runtime_error("cannot read stable STRIX_BITCALC_FORCE_TOKENS");
        std::size_t offset = 0;
        while (offset < text.size()) {
            if (text[offset] == ',' || std::isspace(static_cast<unsigned char>(text[offset]))) {
                ++offset;
                continue;
            }
            std::uint32_t token = 0;
            const auto parsed = std::from_chars(text.data() + offset, text.data() + text.size(), token);
            if (parsed.ec != std::errc{} || (parsed.ptr != text.data() + text.size() &&
                *parsed.ptr != ',' && !std::isspace(static_cast<unsigned char>(*parsed.ptr))))
                throw std::invalid_argument("STRIX_BITCALC_FORCE_TOKENS requires comma/space-separated uint32 token IDs");
            if (forced_tokens_.size() >= steps)
                throw std::invalid_argument("STRIX_BITCALC_FORCE_TOKENS token count exceeds generation budget");
            if (token >= vocab)
                throw std::invalid_argument("STRIX_BITCALC_FORCE_TOKENS token is outside vocabulary");
            forced_tokens_.push_back(token);
            offset = static_cast<std::size_t>(parsed.ptr - text.data());
        }
        force_sha256_ = model_content_identity_hex(sha256_bytes(std::as_bytes(std::span<const char>(text.data(), text.size()))));
    }

    bool enabled_ = false, prepared_ = false, finished_ = false, write_failed_ = false;
    std::string output_path_, partial_path_, force_path_, force_sha256_;
    std::FILE* output_ = nullptr;
    std::vector<std::uint32_t> forced_tokens_;
    std::uint64_t forced_steps_ = 0, byte_ceiling_ = 0, bytes_written_ = 0;
};

}  // namespace strix::decode_cli
