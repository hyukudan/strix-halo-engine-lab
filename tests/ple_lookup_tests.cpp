#include "strix/ple_lookup.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
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


    {
        constexpr std::uint64_t row_elements = 160;
        constexpr std::uint64_t rows = 4;
        constexpr std::uint64_t row_bytes =
            row_elements * sizeof(std::uint16_t);

        std::vector<std::uint16_t> table(
            static_cast<std::size_t>(
                row_elements * rows));
        for (std::uint64_t row = 0;
             row < rows;
             ++row) {
            for (std::uint64_t column = 0;
                 column < row_elements;
                 ++column) {
                table[
                    static_cast<std::size_t>(
                        row * row_elements +
                        column)] =
                    static_cast<std::uint16_t>(
                        row * 1000ULL +
                        column);
            }
        }

        strix::ManifestTensor table_manifest;
        table_manifest.name =
            "per_layer_token_embd.weight";
        table_manifest.dimensions = {
            row_elements,
            rows,
        };
        table_manifest.ggml_type = 30;
        table_manifest.byte_size =
            table.size() *
            sizeof(std::uint16_t);
        table_manifest.role =
            strix::TensorRole::ngram_table;

        strix::RuntimeTensorBinding table_runtime;
        table_runtime.storage =
            strix::RuntimeTensorStorage::file_mapped;
        table_runtime.address =
            reinterpret_cast<std::uintptr_t>(
                table.data());
        table_runtime.bytes =
            table_manifest.byte_size;
        table_runtime.allocation_generation = 1;

        const std::array<std::uint64_t, 2>
            requested_rows{2, 0};
        strix::PleLookupRequest lookup;
        lookup.row_ids =
            requested_rows;
        lookup.row_bytes =
            row_bytes;
        lookup.table_manifest =
            &table_manifest;
        lookup.table_runtime =
            &table_runtime;

        strix::FileMappedPleLookupBackend backend;
        expect(
            backend.implementation_identity() != 0,
            "file-mapped PLE backend has stable non-zero identity");

        const auto handle =
            backend.start(lookup);
        std::vector<std::byte> gathered(
            static_cast<std::size_t>(
                row_bytes *
                requested_rows.size()));
        backend.finish(
            handle,
            gathered);

        const auto* gathered_values =
            reinterpret_cast<
                const std::uint16_t*>(
                    gathered.data());
        expect(
            gathered_values[0] == 2000 &&
                gathered_values[
                    row_elements - 1] ==
                    2159 &&
                gathered_values[
                    row_elements] == 0 &&
                gathered_values[
                    2 * row_elements - 1] ==
                    159,
            "file-mapped backend gathers requested rows in order");

        bool concurrent_rejected = false;
        const auto active =
            backend.start(lookup);
        try {
            static_cast<void>(
                backend.start(lookup));
        } catch (const std::logic_error&) {
            concurrent_rejected = true;
        }
        expect(
            concurrent_rejected,
            "file-mapped backend rejects concurrent C1 lookup");
        backend.cancel(active);

        auto resident = table_runtime;
        resident.storage =
            strix::RuntimeTensorStorage::resident;
        lookup.table_runtime =
            &resident;
        bool resident_rejected = false;
        try {
            static_cast<void>(
                backend.start(lookup));
        } catch (const std::invalid_argument&) {
            resident_rejected = true;
        }
        expect(
            resident_rejected,
            "file-mapped backend does not assume resident GPU memory is host-readable");

        lookup.table_runtime =
            &table_runtime;
        const std::array<std::uint64_t, 1>
            bad_row{rows};
        lookup.row_ids = bad_row;
        bool row_rejected = false;
        try {
            static_cast<void>(
                backend.start(lookup));
        } catch (const std::out_of_range&) {
            row_rejected = true;
        }
        expect(
            row_rejected,
            "file-mapped backend rejects row outside table geometry");
    }

    if (failures == 0) {
        std::cout
            << "PLE lookup tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
