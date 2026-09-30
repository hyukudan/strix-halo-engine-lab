#include "strix/ple_lookup.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto contract =
        strix::build_ple_hash_contract(
            model);

    expect(
        contract.head_count == 16 &&
            contract.head_dim == 160 &&
            contract.row_bytes == 320,
        "PLE hash contract exposes canonical row geometry");
    expect(
        contract.minimum_table_rows >
            model.ngram_vocab_size_base,
        "PLE hash contract builds disjoint head ranges");

    const std::array<std::uint32_t, 2> initial{
        static_cast<std::uint32_t>(
            model.ple_eos_token_id),
        static_cast<std::uint32_t>(
            model.ple_eos_token_id),
    };
    const auto first =
        strix::hash_ple_token(
            contract,
            5,
            initial);
    expect(
        first.row_count == 16 &&
            first.context_count == 2 &&
            first.next_context[0] ==
                model.ple_eos_token_id &&
            first.next_context[1] == 5,
        "first PLE token advances EOS-padded history");

    const std::array<std::uint32_t, 2> second_context{
        first.next_context[0],
        first.next_context[1],
    };
    const auto second =
        strix::hash_ple_token(
            contract,
            7,
            second_context);
    expect(
        second.next_context[0] == 5 &&
            second.next_context[1] == 7,
        "second PLE token retains previous token");

    const std::array<std::uint32_t, 2> after_eos_context{
        11,
        static_cast<std::uint32_t>(
            model.ple_eos_token_id),
    };
    const auto after_eos =
        strix::hash_ple_token(
            contract,
            13,
            after_eos_context);
    const std::array<std::uint32_t, 2> clean_context{
        static_cast<std::uint32_t>(
            model.ple_eos_token_id),
        static_cast<std::uint32_t>(
            model.ple_eos_token_id),
    };
    const auto clean =
        strix::hash_ple_token(
            contract,
            13,
            clean_context);
    expect(
        after_eos.row_ids == clean.row_ids,
        "EOS in immediate history resets older PLE context");

    bool bad_context = false;
    try {
        const std::array<std::uint32_t, 1> short_context{1};
        static_cast<void>(
            strix::hash_ple_token(
                contract,
                2,
                short_context));
    } catch (const std::invalid_argument&) {
        bad_context = true;
    }
    expect(
        bad_context,
        "PLE hashing rejects wrong history length");

    if (failures == 0) {
        std::cout
            << "PLE lookup tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
