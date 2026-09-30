#include "strix/ple_lookup.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strix {
namespace {

constexpr std::uint64_t mask64 = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t splitmix_gamma = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t splitmix_m1 = 0xBF58476D1CE4E5B9ULL;
constexpr std::uint64_t splitmix_m2 = 0x94D049BB133111EBULL;
constexpr std::uint64_t prime_1 = 10007ULL;

std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value = (value + splitmix_gamma) & mask64;
    value = ((value ^ (value >> 30U)) * splitmix_m1) & mask64;
    value = ((value ^ (value >> 27U)) * splitmix_m2) & mask64;
    return (value ^ (value >> 31U)) & mask64;
}

bool is_prime(std::uint64_t value) {
    if (value < 2) {
        return false;
    }
    if ((value % 2ULL) == 0ULL) {
        return value == 2;
    }
    for (std::uint64_t divisor = 3;
         divisor <= value / divisor;
         divisor += 2) {
        if ((value % divisor) == 0ULL) {
            return false;
        }
    }
    return true;
}

std::uint64_t next_prime(std::uint64_t value) {
    if (value == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("PLE prime search overflow");
    }
    ++value;
    while (!is_prime(value)) {
        if (value == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("PLE prime search overflow");
        }
        ++value;
    }
    return value;
}

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error("PLE hash addition overflow");
    }
    return left + right;
}

}  // namespace

PleHashContract build_ple_hash_contract(
    const ModelContract& model) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }

    const auto head_count64 =
        model.ple_head_count();
    if (model.ngram_size >
            PleHashContract::maximum_ngram ||
        head_count64 >
            PleHashContract::maximum_heads ||
        model.ngram_size < 2) {
        throw std::invalid_argument(
            "PLE hash geometry exceeds fixed runtime bounds");
    }

    PleHashContract contract;
    contract.ngram_size =
        static_cast<std::uint32_t>(
            model.ngram_size);
    contract.heads_per_ngram =
        static_cast<std::uint32_t>(
            model.heads_per_ngram);
    contract.head_count =
        static_cast<std::uint32_t>(
            head_count64);
    contract.head_dim =
        static_cast<std::uint32_t>(
            model.ple_head_dim());
    contract.eos_token_id =
        static_cast<std::uint32_t>(
            model.ple_eos_token_id);
    contract.row_bytes =
        model.ple_row_bytes_bf16();

    const auto maximum =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max());
    const auto multiplier_max =
        maximum /
        std::max<std::uint64_t>(
            model.vocab_size,
            1ULL);
    const auto half_bound =
        std::max<std::uint64_t>(
            1ULL,
            multiplier_max / 2ULL);

    const auto base_seed =
        model.ngram_hash_seed;
    for (std::uint32_t index = 0;
         index < contract.ngram_size;
         ++index) {
        const auto value =
            base_seed +
            splitmix_gamma *
                static_cast<std::uint64_t>(
                    index + 1U);
        contract.multipliers[index] =
            2ULL *
                (splitmix64(value) %
                 half_bound) +
            1ULL;
    }

    auto prime =
        model.ngram_vocab_size_base - 1ULL;
    std::uint64_t offset = 0;
    for (std::uint32_t head = 0;
         head < contract.head_count;
         ++head) {
        prime =
            next_prime(prime);
        contract.head_vocab_sizes[head] =
            prime;
        contract.head_offsets[head] =
            offset;
        offset =
            checked_add(
                offset,
                prime);
    }
    contract.minimum_table_rows =
        offset;
    return contract;
}

PleHashResult hash_ple_token(
    const PleHashContract& contract,
    std::uint32_t token_id,
    std::span<const std::uint32_t> previous_context) {
    if (contract.ngram_size < 2 ||
        contract.head_count == 0 ||
        contract.head_count >
            PleHashContract::maximum_heads ||
        previous_context.size() !=
            static_cast<std::size_t>(
                contract.ngram_size - 1U)) {
        throw std::invalid_argument(
            "PLE token hash context does not match contract");
    }

    std::array<std::uint64_t, PleHashContract::maximum_ngram> context{};
    context[0] =
        token_id;

    bool cut = false;
    for (std::uint32_t shift = 1;
         shift < contract.ngram_size;
         ++shift) {
        const auto source_index =
            previous_context.size() -
            static_cast<std::size_t>(
                shift);
        const auto token =
            previous_context[
                source_index];
        cut =
            cut ||
            token ==
                contract.eos_token_id;
        context[shift] =
            cut
                ? contract.eos_token_id
                : token;
    }

    PleHashResult result;
    result.row_count =
        contract.head_count;
    result.context_count =
        contract.ngram_size - 1U;

    for (std::uint32_t ngram = 2;
         ngram <= contract.ngram_size;
         ++ngram) {
        std::uint64_t mixed =
            context[0] *
            contract.multipliers[0];
        for (std::uint32_t position = 1;
             position < ngram;
             ++position) {
            mixed ^=
                context[position] *
                contract.multipliers[position];
        }

        const auto start =
            (ngram - 2U) *
            contract.heads_per_ngram;
        for (std::uint32_t slot = 0;
             slot <
                contract.heads_per_ngram;
             ++slot) {
            const auto head =
                start + slot;
            result.row_ids[head] =
                mixed %
                    contract.head_vocab_sizes[
                        head] +
                contract.head_offsets[
                    head];
        }
    }

    for (std::uint32_t index = 0;
         index + 1U <
            result.context_count;
         ++index) {
        result.next_context[index] =
            previous_context[
                previous_context.size() -
                result.context_count +
                index +
                1U];
    }
    if (result.context_count != 0) {
        result.next_context[
            result.context_count - 1U] =
            token_id;
    }

    return result;
}

}  // namespace strix


namespace strix {
namespace {

constexpr std::uint64_t file_mapped_ple_backend_identity =
    0x504c45464d415001ULL;

std::uint64_t checked_mul_lookup(
    std::uint64_t left,
    std::uint64_t right) {
    if (left != 0 &&
        right >
            std::numeric_limits<std::uint64_t>::max() /
                left) {
        throw std::overflow_error(
            "PLE lookup multiplication overflow");
    }
    return left * right;
}

}  // namespace

std::uint64_t
FileMappedPleLookupBackend::implementation_identity() const noexcept {
    return file_mapped_ple_backend_identity;
}

PleLookupHandle
FileMappedPleLookupBackend::start(
    const PleLookupRequest& request) {
    if (pending_.active) {
        throw std::logic_error(
            "file-mapped PLE backend already has an active lookup");
    }
    if (request.row_ids.empty() ||
        request.row_ids.size() >
            PleHashContract::maximum_heads ||
        request.row_bytes == 0 ||
        request.table_manifest == nullptr ||
        request.table_runtime == nullptr) {
        throw std::invalid_argument(
            "file-mapped PLE lookup request is incomplete");
    }

    const auto& manifest =
        *request.table_manifest;
    const auto& runtime =
        *request.table_runtime;

    if (manifest.role !=
            TensorRole::ngram_table ||
        manifest.dimensions.size() < 2 ||
        manifest.dimensions[0] == 0 ||
        manifest.dimensions[1] == 0) {
        throw std::invalid_argument(
            "PLE lookup table manifest has invalid geometry");
    }
    if (manifest.ggml_type != 30) {
        throw std::invalid_argument(
            "file-mapped PLE baseline requires BF16 table rows");
    }
    const auto expected_row_bytes =
        checked_mul_lookup(
            manifest.dimensions[0],
            2ULL);
    if (request.row_bytes !=
        expected_row_bytes) {
        throw std::invalid_argument(
            "PLE lookup row size does not match BF16 table geometry");
    }
    if (runtime.storage !=
            RuntimeTensorStorage::file_mapped ||
        runtime.address == 0 ||
        runtime.bytes < manifest.byte_size) {
        throw std::invalid_argument(
            "file-mapped PLE backend requires a host-readable file mapping");
    }

    const auto minimum_bytes =
        checked_mul_lookup(
            manifest.dimensions[1],
            request.row_bytes);
    if (runtime.bytes < minimum_bytes) {
        throw std::invalid_argument(
            "PLE file mapping is smaller than declared table rows");
    }

    for (const auto row :
         request.row_ids) {
        if (row >=
            manifest.dimensions[1]) {
            throw std::out_of_range(
                "PLE row id exceeds table row count");
        }
    }

    if (next_handle_ == 0) {
        throw std::overflow_error(
            "PLE lookup handle space exhausted");
    }

    pending_.handle =
        next_handle_++;
    pending_.table_address =
        runtime.address;
    pending_.table_bytes =
        runtime.bytes;
    pending_.row_bytes =
        request.row_bytes;
    pending_.row_count =
        static_cast<std::uint32_t>(
            request.row_ids.size());
    std::copy(
        request.row_ids.begin(),
        request.row_ids.end(),
        pending_.row_ids.begin());
    pending_.active = true;
    return pending_.handle;
}

void FileMappedPleLookupBackend::finish(
    PleLookupHandle handle,
    std::span<std::byte> destination) {
    if (!pending_.active ||
        handle == 0 ||
        handle != pending_.handle) {
        throw std::invalid_argument(
            "PLE lookup handle is not active");
    }

    const auto required =
        checked_mul_lookup(
            pending_.row_count,
            pending_.row_bytes);
    if (destination.size() <
        required) {
        throw std::invalid_argument(
            "PLE lookup destination is undersized");
    }

    const auto* table =
        reinterpret_cast<
            const std::byte*>(
                pending_.table_address);
    for (std::uint32_t index = 0;
         index < pending_.row_count;
         ++index) {
        const auto offset =
            checked_mul_lookup(
                pending_.row_ids[index],
                pending_.row_bytes);
        if (offset >
                pending_.table_bytes ||
            pending_.row_bytes >
                pending_.table_bytes -
                    offset) {
            throw std::out_of_range(
                "PLE lookup row range exceeds mapped table");
        }
        std::memcpy(
            destination.data() +
                static_cast<std::size_t>(
                    checked_mul_lookup(
                        index,
                        pending_.row_bytes)),
            table +
                static_cast<std::size_t>(
                    offset),
            static_cast<std::size_t>(
                pending_.row_bytes));
    }

    pending_ = {};
}

void FileMappedPleLookupBackend::cancel(
    PleLookupHandle handle) noexcept {
    if (pending_.active &&
        handle != 0 &&
        handle == pending_.handle) {
        pending_ = {};
    }
}

}  // namespace strix
