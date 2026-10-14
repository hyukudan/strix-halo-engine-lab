#include "strix/ple_lookup.hpp"
#include "strix/iq4_nl.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

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

std::uint64_t ple_hash_contract_identity(
    const PleHashContract& contract) noexcept {
    std::uint64_t hash = 14695981039346656037ULL;
    const auto mix = [&](std::uint64_t value) {
        for (unsigned shift = 0; shift < 64; shift += 8) {
            hash ^= (value >> shift) & 0xffULL;
            hash *= 1099511628211ULL;
        }
    };
    mix(contract.ngram_size);
    mix(contract.heads_per_ngram);
    mix(contract.head_count);
    mix(contract.head_dim);
    mix(contract.eos_token_id);
    mix(contract.row_bytes);
    mix(contract.minimum_table_rows);
    for (const auto value : contract.multipliers) mix(value);
    for (const auto value : contract.head_offsets) mix(value);
    for (const auto value : contract.head_vocab_sizes) mix(value);
    return hash;
}

void start_ple_lookup_prefetch(
    const PleHashContract& contract,
    std::uint32_t ple_step_id,
    std::uint32_t input_token_id,
    std::uint64_t position,
    const ModelManifest& manifest,
    const ModelRuntimeBinding& weights,
    const ExecutionTensorAccessPlan& tensor_access,
    const SerialSessionStateLayout& state_layout,
    const SessionStateMemoryBinding& state_memory,
    PleLookupBackend& backend,
    PleLookupPrefetchState& prefetch) {
    if (prefetch.active) {
        throw std::logic_error("PLE prefetch already active");
    }
    if (backend.implementation_identity() == 0 ||
        ple_hash_contract_identity(contract) == 0) {
        throw std::invalid_argument(
            "PLE prefetch requires initialized backend/hash contract");
    }

    const auto* access =
        find_execution_tensor_access(tensor_access, ple_step_id, 0);
    if (access == nullptr ||
        access->op != SerialOp::ple ||
        access->tensor_indices.size() != 1) {
        throw std::invalid_argument(
            "PLE prefetch cannot resolve unique n-gram table");
    }
    const auto tensor_index = access->tensor_indices.front();
    if (tensor_index >= manifest.tensors.size()) {
        throw std::out_of_range(
            "PLE prefetch table index exceeds manifest");
    }
    const auto* runtime =
        find_runtime_tensor_binding(weights, tensor_index);
    if (runtime == nullptr) {
        throw std::invalid_argument(
            "PLE prefetch table runtime binding is missing");
    }

    const auto ngram =
        resolve_session_state(
            state_layout,
            state_memory,
            SessionStateComponent::ngram_history);
    const auto context_count =
        static_cast<std::size_t>(contract.ngram_size - 1U);
    if (ngram.storage != SessionStateStorage::host ||
        ngram.address == 0 ||
        ngram.bytes < context_count * sizeof(std::uint32_t)) {
        throw std::invalid_argument(
            "PLE prefetch requires host n-gram history");
    }

    std::array<
        std::uint32_t,
        PleHashContract::maximum_ngram - 1> previous{};
    if (position == 0) {
        std::fill_n(
            previous.begin(),
            context_count,
            contract.eos_token_id);
    } else {
        std::memcpy(
            previous.data(),
            reinterpret_cast<const void*>(ngram.address),
            context_count * sizeof(std::uint32_t));
    }

    const auto hashed =
        hash_ple_token(
            contract,
            input_token_id,
            std::span<const std::uint32_t>(
                previous.data(), context_count));

    PleLookupRequest request;
    request.row_ids =
        std::span<const std::uint64_t>(
            hashed.row_ids.data(), hashed.row_count);
    request.row_bytes = contract.row_bytes;
    request.table_manifest = &manifest.tensors[tensor_index];
    request.table_runtime = runtime;

    const auto handle = backend.start(request);
    if (handle == 0) {
        throw std::runtime_error(
            "PLE prefetch backend returned zero handle");
    }
    if (hashed.row_count != 0 &&
        contract.row_bytes >
            std::numeric_limits<std::uint64_t>::max() /
                hashed.row_count) {
        backend.cancel(handle);
        throw std::overflow_error(
            "PLE prefetch byte count overflow");
    }

    prefetch.handle = handle;
    prefetch.backend_identity = backend.implementation_identity();
    prefetch.input_token_id = input_token_id;
    prefetch.position = position;
    prefetch.lookup_bytes =
        static_cast<std::uint64_t>(hashed.row_count) *
        contract.row_bytes;
    prefetch.hash = hashed;
    prefetch.active = true;
}

PleHashResult finish_ple_lookup_prefetch(
    std::uint32_t input_token_id,
    std::uint64_t position,
    PleLookupBackend& backend,
    PleLookupPrefetchState& prefetch,
    std::span<std::byte> destination) {
    if (!prefetch.active ||
        prefetch.handle == 0 ||
        prefetch.backend_identity != backend.implementation_identity() ||
        prefetch.input_token_id != input_token_id ||
        prefetch.position != position ||
        prefetch.lookup_bytes > destination.size()) {
        throw std::invalid_argument(
            "PLE prefetch does not match active token/backend");
    }

    const auto handle = prefetch.handle;
    const auto hash = prefetch.hash;
    try {
        backend.finish(
            handle,
            destination.first(
                static_cast<std::size_t>(prefetch.lookup_bytes)));
    } catch (...) {
        backend.cancel(handle);
        prefetch = {};
        throw;
    }
    prefetch = {};
    return hash;
}

void cancel_ple_lookup_prefetch(
    PleLookupBackend& backend,
    PleLookupPrefetchState& prefetch) noexcept {
    if (prefetch.active && prefetch.handle != 0) {
        backend.cancel(prefetch.handle);
    }
    prefetch = {};
}

}  // namespace strix


namespace strix {
namespace {

constexpr std::uint64_t file_mapped_ple_backend_identity =
    0x504c45464d415002ULL;

std::uint64_t checked_add_lookup(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() -
            left) {
        throw std::overflow_error(
            "PLE lookup addition overflow");
    }
    return left + right;
}

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
        manifest.dimensions.size() != 2 ||
        manifest.dimensions[0] == 0 ||
        manifest.dimensions[1] == 0) {
        throw std::invalid_argument(
            "PLE lookup table manifest has invalid geometry");
    }
    if (manifest.ggml_type != 30 && manifest.ggml_type != 20) {
        throw std::invalid_argument(
            "file-mapped PLE lookup supports BF16 or IQ4_NL table rows");
    }
    const auto expected_row_bytes =
        checked_mul_lookup(
            manifest.dimensions[0],
            2ULL);
    if (request.row_bytes !=
        expected_row_bytes) {
        throw std::invalid_argument(
            "PLE lookup output row size does not match BF16 activation geometry");
    }
    std::uint64_t stored_row_bytes = expected_row_bytes;
    if (manifest.ggml_type == 20) {
        if (manifest.dimensions[0] % iq4_nl::block_elements != 0) {
            throw std::invalid_argument(
                "IQ4_NL PLE rows require complete 32-element blocks");
        }
        stored_row_bytes = checked_mul_lookup(
            manifest.dimensions[0] / iq4_nl::block_elements,
            iq4_nl::block_bytes);
    }
    if (runtime.storage !=
            RuntimeTensorStorage::file_mapped ||
        runtime.address == 0 ||
        runtime.bytes < manifest.byte_size) {
        throw std::invalid_argument(
            "file-mapped PLE backend requires a host-readable file mapping");
    }
    if (runtime.bytes > std::numeric_limits<std::uintptr_t>::max() - runtime.address) {
        throw std::overflow_error("PLE mapping address range overflows the host address space");
    }

    const auto minimum_bytes =
        checked_mul_lookup(
            manifest.dimensions[1],
            stored_row_bytes);
    if (runtime.bytes < minimum_bytes || manifest.byte_size < minimum_bytes) {
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
    pending_.stored_row_bytes = stored_row_bytes;
    pending_.ggml_type = manifest.ggml_type;
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
                pending_.stored_row_bytes);
        if (offset >
                pending_.table_bytes ||
            pending_.stored_row_bytes >
                pending_.table_bytes -
                    offset) {
            throw std::out_of_range(
                "PLE lookup row range exceeds mapped table");
        }
        auto* output = destination.data() + static_cast<std::size_t>(
            checked_mul_lookup(index, pending_.row_bytes));
        const auto* input = table + static_cast<std::size_t>(offset);
        if (pending_.ggml_type == 30) {
            std::memcpy(output, input, static_cast<std::size_t>(pending_.row_bytes));
            continue;
        }
        // Only selected IQ4_NL rows are decoded. Storage remains quantized;
        // the existing activation contract is BF16, rounded to nearest-even.
        const auto blocks = pending_.stored_row_bytes / iq4_nl::block_bytes;
        for (std::uint64_t block = 0; block < blocks; ++block) {
            const std::span<const std::byte, iq4_nl::block_bytes> packed(
                input + static_cast<std::size_t>(block * iq4_nl::block_bytes),
                iq4_nl::block_bytes);
            if (!std::isfinite(iq4_nl::fp16_bits_to_float(iq4_nl::scale_bits(packed)))) {
                throw std::invalid_argument("IQ4_NL PLE scale must be finite");
            }
            for (std::size_t element = 0; element < iq4_nl::block_elements; ++element) {
                const auto value = iq4_nl::dequantized_value(packed, element);
                const auto bits = std::bit_cast<std::uint32_t>(value);
                const auto bf16 = static_cast<std::uint16_t>(
                    (bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
                const auto column = block * iq4_nl::block_elements + element;
                std::memcpy(output + static_cast<std::size_t>(column * 2U), &bf16, sizeof(bf16));
            }
        }
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

PlannedPleLookupBackend::PlannedPleLookupBackend(
    SparseLookupIoSource& source,
    PlannedPleLookupPolicy policy)
    : source_(&source),
      policy_(policy) {
    if (policy_.max_inflight_ios == 0) {
        throw std::invalid_argument(
            "planned PLE lookup requires non-zero inflight depth");
    }
}

std::uint64_t
PlannedPleLookupBackend::implementation_identity() const noexcept {
    constexpr std::uint64_t base =
        0x504c45504c414e31ULL;
    return base ^
           (policy_.max_rows_per_io << 17U) ^
           (policy_.max_inflight_ios << 33U);
}

PleLookupHandle
PlannedPleLookupBackend::start(
    const PleLookupRequest& request) {
    if (pending_.active) {
        throw std::logic_error(
            "planned PLE backend already has an active lookup");
    }
    if (source_ == nullptr ||
        request.row_ids.empty() ||
        request.row_ids.size() >
            PleHashContract::maximum_heads ||
        request.row_bytes == 0 ||
        request.table_manifest == nullptr ||
        request.table_runtime == nullptr) {
        throw std::invalid_argument(
            "planned PLE lookup request is incomplete");
    }

    const auto& manifest =
        *request.table_manifest;
    const auto& runtime =
        *request.table_runtime;

    if (manifest.role !=
            TensorRole::ngram_table ||
        manifest.dimensions.size() < 2 ||
        manifest.dimensions[0] == 0 ||
        manifest.dimensions[1] == 0 ||
        manifest.ggml_type != 30) {
        throw std::invalid_argument(
            "planned PLE lookup requires a BF16 n-gram table");
    }

    const auto expected_row_bytes =
        checked_mul_lookup(
            manifest.dimensions[0],
            2ULL);
    if (request.row_bytes !=
        expected_row_bytes) {
        throw std::invalid_argument(
            "planned PLE row size does not match table geometry");
    }
    if (!runtime_tensor_file_backed(
            runtime.storage) ||
        runtime.bytes <
            manifest.byte_size) {
        throw std::invalid_argument(
            "planned PLE lookup requires a file-backed tensor binding");
    }

    const auto minimum_bytes =
        checked_mul_lookup(
            manifest.dimensions[1],
            request.row_bytes);
    if (manifest.byte_size <
        minimum_bytes) {
        throw std::invalid_argument(
            "planned PLE tensor is smaller than declared rows");
    }

    for (const auto row :
         request.row_ids) {
        if (row >=
            manifest.dimensions[1]) {
            throw std::out_of_range(
                "planned PLE row id exceeds table row count");
        }
    }

    SparseLookupRequest sparse_request;
    sparse_request.table_rows =
        manifest.dimensions[1];
    sparse_request.row_bytes =
        request.row_bytes;
    sparse_request.cache_capacity_rows = 0;
    sparse_request.max_rows_per_io =
        policy_.max_rows_per_io;
    sparse_request.max_inflight_ios =
        policy_.max_inflight_ios;
    sparse_request.row_ids =
        request.row_ids;

    auto plan =
        build_sparse_lookup_plan(
            sparse_request);
    for (auto& operation :
         plan.io_plan) {
        if (operation.byte_offset >
                manifest.byte_size ||
            operation.byte_count >
                manifest.byte_size -
                    operation.byte_offset) {
            throw std::out_of_range(
                "planned PLE I/O exceeds tensor range");
        }
        operation.byte_offset =
            checked_add_lookup(
                manifest.absolute_offset,
                operation.byte_offset);
    }

    if (next_handle_ == 0) {
        throw std::overflow_error(
            "planned PLE lookup handle space exhausted");
    }

    const auto handle =
        next_handle_++;
    auto* source =
        source_;
    auto future =
        std::async(
            std::launch::async,
            [plan, source]() mutable {
                return execute_sparse_lookup_io_plan(
                    plan,
                    *source);
            });

    pending_.handle =
        handle;
    pending_.row_bytes =
        request.row_bytes;
    pending_.row_count =
        static_cast<std::uint32_t>(
            request.row_ids.size());
    std::copy(
        request.row_ids.begin(),
        request.row_ids.end(),
        pending_.row_ids.begin());
    pending_.io_plan =
        std::move(plan.io_plan);
    pending_.future =
        std::move(future);
    pending_.active = true;
    return handle;
}

void PlannedPleLookupBackend::finish(
    PleLookupHandle handle,
    std::span<std::byte> destination) {
    if (!pending_.active ||
        handle == 0 ||
        handle != pending_.handle) {
        throw std::invalid_argument(
            "planned PLE lookup handle is not active");
    }

    const auto required =
        checked_mul_lookup(
            pending_.row_count,
            pending_.row_bytes);
    if (destination.size() <
        required) {
        throw std::invalid_argument(
            "planned PLE lookup destination is undersized");
    }

    auto current =
        std::move(pending_);
    pending_ = {};

    const auto execution =
        current.future.get();
    if (execution.operation_payloads.size() !=
        current.io_plan.size()) {
        throw std::logic_error(
            "planned PLE execution payload count mismatch");
    }

    for (std::uint32_t output_row = 0;
         output_row < current.row_count;
         ++output_row) {
        const auto row =
            current.row_ids[output_row];
        const std::byte* source_row = nullptr;

        for (std::size_t operation_index = 0;
             operation_index <
                current.io_plan.size();
             ++operation_index) {
            const auto& operation =
                current.io_plan[
                    operation_index];
            if (row <
                    operation.first_row ||
                row >=
                    operation.first_row +
                        operation.row_count) {
                continue;
            }

            const auto within_rows =
                row -
                operation.first_row;
            const auto within_bytes =
                checked_mul_lookup(
                    within_rows,
                    current.row_bytes);
            const auto& payload =
                execution.operation_payloads[
                    operation_index];
            if (within_bytes >
                    payload.size() ||
                current.row_bytes >
                    payload.size() -
                        static_cast<std::size_t>(
                            within_bytes)) {
                throw std::logic_error(
                    "planned PLE payload is missing requested row bytes");
            }

            source_row =
                payload.data() +
                static_cast<std::size_t>(
                    within_bytes);
            break;
        }

        if (source_row == nullptr) {
            throw std::logic_error(
                "planned PLE requested row has no owning I/O operation");
        }

        std::memcpy(
            destination.data() +
                static_cast<std::size_t>(
                    checked_mul_lookup(
                        output_row,
                        current.row_bytes)),
            source_row,
            static_cast<std::size_t>(
                current.row_bytes));
    }
}

void PlannedPleLookupBackend::cancel(
    PleLookupHandle handle) noexcept {
    if (!pending_.active ||
        handle == 0 ||
        handle != pending_.handle) {
        return;
    }

    auto cancelled =
        std::move(pending_);
    pending_ = {};
    // std::future from std::launch::async joins on destruction. The source
    // contract has no cancellation primitive yet, so cancel is fail-closed:
    // results are discarded and no exception escapes this noexcept boundary.
    static_cast<void>(cancelled);
}

}  // namespace strix
