#pragma once

#include "inkling2/inkling2_mini_resident_decoder.hpp"

#include <array>
#include <cstring>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::inkling2::test {

inline void verify_rows(const char* checkpoint, weight::FormatId storage, bool packed) {
    // Cross a local KV ring wrap and global attention key tile boundaries.
    constexpr unsigned prefix_rows = 2047U;
    constexpr unsigned prefill_chunk = 128U;
    constexpr unsigned max_width = MiniResidentDecoder::kMaxVerifyRows;
    constexpr unsigned decode_rows = max_width * (max_width + 1U) / 2U;
    // Include full commits, one- and multi-row prefix replay, and discard.
    constexpr std::array<unsigned, max_width - 1U> kept_rows{2U, 1U, 4U, 3U, 6U, 0U, 8U};
    std::vector<unsigned> tokens(prefix_rows + decode_rows);
    for (std::size_t i = 0; i < tokens.size(); ++i)
        tokens[i] = 1000U + static_cast<unsigned>((i * 173U + i / 13U) % 20000U);
    const auto ids = std::span(tokens);
    MiniResidentDecoder decoder(checkpoint, static_cast<unsigned>(ids.size()), prefill_chunk,
                                 packed, MiniResidentBackend::default_step_graphs, storage, max_width);
    const auto prompt = ids.first(prefix_rows);
    (void)decoder.prefill(prompt);

    std::vector<float> reference(std::size_t{decode_rows} * kOutputVocabulary);
    const auto serial_row = [&](unsigned row) {
        return std::span(reference).subspan(std::size_t{row} * kOutputVocabulary, kOutputVocabulary);
    };
    std::array<unsigned, decode_rows> next{};
    for (unsigned row = 0; row < decode_rows; ++row)
        next[row] = decoder.decode(ids[prefix_rows + row], serial_row(row));

    decoder.reset_gate_session();
    (void)decoder.prefill(prompt);
    std::vector<float> verified(std::size_t{max_width} * kOutputVocabulary);
    unsigned cursor = 0U;
    unsigned last_top0 = 0U;
    const auto storage_name = storage == weight::kQ8WeightStorage ? "q8_0" : "bf16";
    const auto moe = packed ? "packed" : "grouped";
    for (unsigned width = 2U; width <= max_width; ++width) {
        const auto window = ids.subspan(prefix_rows + cursor, width);
        const auto scores = std::span(verified).first(std::size_t{width} * kOutputVocabulary);
        const auto result = decoder.forward_rows(window.data(), width, MiniResidentDecoder::LogitRows::all, scores);
        std::array<unsigned, max_width> top{};
        decoder.read_top(result, width, top.data());
        for (unsigned row = 0; row < width; ++row) {
            const auto actual = scores.subspan(std::size_t{row} * kOutputVocabulary, kOutputVocabulary);
            const auto expected = serial_row(cursor + row);
            if (std::memcmp(actual.data(), expected.data(), actual.size_bytes()) != 0 || top[row] != next[cursor + row])
                throw std::runtime_error("verify mismatch: width " + std::to_string(width) +
                                         ", row " + std::to_string(row) + ", position " +
                                         std::to_string(prefix_rows + cursor + row));
        }
        const auto keep = kept_rows[width - 2U];
        decoder.commit(keep);
        cursor += keep;
        last_top0 = top[0];
        std::cout << "PASS Inkling Mini verify rows " << storage_name << ' ' << moe
                  << " width=" << width << '\n';
    }
    {
        // A one-row window without a logits span takes the greedy (on CUDA: screened) head pick; its reply must be
        // this window's pick, not row 0 of the previous window.
        const auto one = decoder.forward_rows(&ids[prefix_rows + cursor], 1U, MiniResidentDecoder::LogitRows::all);
        unsigned top = 0U;
        decoder.read_top(one, 1U, &top);
        if (top != next[cursor])
            throw std::runtime_error("one-row verify reply differs from serial decode (stale " +
                                     std::string(top == last_top0 ? "previous window" : "value") + ")");
        decoder.commit(0U);
        std::cout << "PASS Inkling Mini verify one row " << storage_name << ' ' << moe
                  << " discriminating=" << (last_top0 != next[cursor] ? "yes" : "no") << '\n';
    }
    const auto actual = std::span(verified).first(kOutputVocabulary);
    if (decoder.decode(ids[prefix_rows + cursor], actual) != next[cursor] ||
        std::memcmp(actual.data(), serial_row(cursor).data(), actual.size_bytes()) != 0)
        throw std::runtime_error("decode mismatch after verify commits");
    std::cout << "PASS Inkling Mini verify continuation " << storage_name << ' ' << moe << '\n';
}

}  // namespace strix::inkling2::test

int main(int argc, char** argv) {
    using namespace strix;
    try {
        if (argc < 2) throw std::invalid_argument("expected MODEL.safetensors [--qualify-q8] [--grouped-decode]");
        auto storage = weight::kBf16WeightStorage;
        bool packed = true;
        for (int i = 2; i < argc; ++i) {
            const std::string_view option{argv[i]};
            if (option == "--qualify-q8") storage = weight::kQ8WeightStorage;
            else if (option == "--grouped-decode") packed = false;
            else throw std::invalid_argument("unknown option: " + std::string(option));
        }
        inkling2::test::verify_rows(argv[1], storage, packed);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL Inkling Mini verify rows: " << error.what() << '\n';
        return 1;
    }
}
