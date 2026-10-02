#include "strix/compiled_execution.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/hip_launcher_catalog.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/hip_runtime_allocator.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/mapped_model_file.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_tensor_index.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/runtime_allocator.hpp"
#include "strix/serial_decode_reference.hpp"
#include "strix/serial_decode_replay.hpp"
#include "strix/serial_session.hpp"
#include "strix/session_state_layout.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void check_hip(
    hipError_t error,
    const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            hipGetErrorString(error));
    }
}

std::uint64_t parse_u64(
    std::string_view text,
    const char* name) {
    if (text.empty()) {
        throw std::invalid_argument(
            std::string(name) +
            " must not be empty");
    }
    std::uint64_t value = 0;
    for (const char character : text) {
        if (character < '0' ||
            character > '9') {
            throw std::invalid_argument(
                std::string(name) +
                " must be an unsigned integer");
        }
        const auto digit =
            static_cast<std::uint64_t>(
                character - '0');
        if (value >
            (std::numeric_limits<
                 std::uint64_t>::max() -
             digit) /
                10ULL) {
            throw std::overflow_error(
                std::string(name) +
                " overflows uint64");
        }
        value =
            value * 10ULL +
            digit;
    }
    return value;
}

double parse_double(
    std::string_view text,
    const char* name) {
    if (text.empty()) {
        throw std::invalid_argument(
            std::string(name) +
            " must not be empty");
    }
    std::string copy(text);
    char* end = nullptr;
    errno = 0;
    const auto value =
        std::strtod(
            copy.c_str(),
            &end);
    if (errno != 0 ||
        end == copy.c_str() ||
        *end != '\0' ||
        !std::isfinite(value)) {
        throw std::invalid_argument(
            std::string(name) +
            " must be a finite number");
    }
    return value;
}

strix::PersistentState all_persistent_state() {
    using strix::PersistentState;
    return
        PersistentState::attention_kv |
        PersistentState::recurrent_matrix |
        PersistentState::convolution_history |
        PersistentState::sparse_index |
        PersistentState::ngram_history |
        PersistentState::position_history;
}

struct Options {
    std::string model_path;
    std::vector<std::uint32_t> tokens;
    std::uint64_t context_tokens = 128;
    strix::HipModelResidencyMode residency =
        strix::HipModelResidencyMode::
            registered_file_mapping;
    std::uint64_t coalesce_gap_bytes = 4096;
    std::string reference_path;
    double max_logit_abs_error = 1e-3;
};

Options parse_options(
    int argc,
    char** argv) {
    if (argc < 2) {
        throw std::invalid_argument(
            "usage: strix-hip-decode-token MODEL.gguf [TOKEN_ID ...] "
            "[--reference FILE] [--max-logit-error X] "
            "[--context N] [--residency register|copy] [--coalesce-gap BYTES]");
    }

    Options options;
    options.model_path =
        argv[1];

    for (int index = 2;
         index < argc;
         ++index) {
        const std::string_view argument =
            argv[index];

        if (argument == "--reference") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    "--reference requires a fixture path");
            }
            options.reference_path =
                argv[++index];
            continue;
        }

        if (argument == "--max-logit-error") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    "--max-logit-error requires a value");
            }
            options.max_logit_abs_error =
                parse_double(
                    argv[++index],
                    "max-logit-error");
            if (options.max_logit_abs_error < 0.0) {
                throw std::invalid_argument(
                    "--max-logit-error must be non-negative");
            }
            continue;
        }

        if (argument == "--context") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    "--context requires a token count");
            }
            options.context_tokens =
                parse_u64(
                    argv[++index],
                    "context");
            continue;
        }

        if (argument == "--residency") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    "--residency requires register or copy");
            }
            const std::string_view value =
                argv[++index];
            if (value == "register") {
                options.residency =
                    strix::HipModelResidencyMode::
                        registered_file_mapping;
            } else if (value == "copy") {
                options.residency =
                    strix::HipModelResidencyMode::
                        resident_copy;
            } else {
                throw std::invalid_argument(
                    "--residency must be register or copy");
            }
            continue;
        }

        if (argument == "--coalesce-gap") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    "--coalesce-gap requires bytes");
            }
            options.coalesce_gap_bytes =
                parse_u64(
                    argv[++index],
                    "coalesce-gap");
            continue;
        }

        if (argument.starts_with("--")) {
            throw std::invalid_argument(
                "unknown option: " +
                std::string(argument));
        }

        const auto token =
            parse_u64(
                argument,
                "token id");
        if (token >
            std::numeric_limits<
                std::uint32_t>::max()) {
            throw std::out_of_range(
                "token id exceeds uint32");
        }
        options.tokens.push_back(
            static_cast<std::uint32_t>(
                token));
    }

    if (options.tokens.empty() &&
        options.reference_path.empty()) {
        throw std::invalid_argument(
            "at least one token id or --reference is required");
    }
    if (options.context_tokens == 0) {
        throw std::invalid_argument(
            "context must be non-zero");
    }
    return options;
}

struct RankedLogit {
    std::uint32_t token = 0;
    float logit = 0.0F;
};

std::vector<RankedLogit> top_logits(
    const std::vector<float>& logits,
    std::size_t count) {
    std::vector<std::uint32_t> order(
        logits.size());
    for (std::size_t index = 0;
         index < order.size();
         ++index) {
        order[index] =
            static_cast<std::uint32_t>(
                index);
    }

    count =
        std::min(
            count,
            order.size());
    std::partial_sort(
        order.begin(),
        order.begin() +
            static_cast<
                std::ptrdiff_t>(
                count),
        order.end(),
        [&](std::uint32_t left,
            std::uint32_t right) {
            if (logits[left] !=
                logits[right]) {
                return logits[left] >
                       logits[right];
            }
            return left < right;
        });

    std::vector<RankedLogit> result;
    result.reserve(count);
    for (std::size_t index = 0;
         index < count;
         ++index) {
        const auto token =
            order[index];
        result.push_back(
            RankedLogit{
                token,
                logits[token],
            });
    }
    return result;
}

}  // namespace

int main(
    int argc,
    char** argv) try {
    auto options =
        parse_options(
            argc,
            argv);

    const auto model =
        strix::ModelContract::
            qwen38_flash_next();

    std::optional<strix::SerialDecodeReference>
        reference;
    std::uint64_t required_reference_logit_values = 0;
    if (!options.reference_path.empty()) {
        reference =
            strix::read_serial_decode_reference(
                options.reference_path);
        strix::validate_serial_decode_reference_vocabulary(
            *reference,
            model.vocab_size);
        if (reference->steps.size() < 2) {
            throw std::invalid_argument(
                "serial replay qualification requires at least two reference steps");
        }
        for (const auto& step : reference->steps) {
            const auto probes =
                static_cast<std::uint64_t>(
                    step.logits.size());
            if (probes >
                std::numeric_limits<std::uint64_t>::max() -
                    required_reference_logit_values) {
                throw std::overflow_error(
                    "reference logit probe count overflow");
            }
            required_reference_logit_values += probes;
        }
        if (required_reference_logit_values == 0) {
            throw std::invalid_argument(
                "serial replay reference requires logit probes");
        }
        if (options.tokens.empty()) {
            options.tokens.reserve(
                reference->steps.size());
            for (const auto& step :
                 reference->steps) {
                options.tokens.push_back(
                    step.input_token);
            }
        } else {
            if (options.tokens.size() !=
                reference->steps.size()) {
                throw std::invalid_argument(
                    "CLI token sequence length does not match replay reference");
            }
            for (std::size_t index = 0;
                 index < options.tokens.size();
                 ++index) {
                if (options.tokens[index] !=
                    reference->steps[index].input_token) {
                    throw std::invalid_argument(
                        "CLI token sequence does not match replay reference");
                }
            }
        }
    }

    for (const auto token : options.tokens) {
        if (static_cast<std::uint64_t>(token) >=
            model.vocab_size) {
            throw std::invalid_argument(
                "input token is outside target model vocabulary");
        }
    }

    if (options.context_tokens >
        model.max_position_embeddings) {
        throw std::invalid_argument(
            "requested context exceeds model maximum");
    }
    if (options.tokens.size() >
        options.context_tokens) {
        throw std::invalid_argument(
            "input token sequence exceeds allocated context");
    }

    const auto info =
        strix::gguf::read_file(
            options.model_path);
    const auto manifest =
        strix::build_model_manifest(
            info);
    const auto tensor_index =
        strix::build_model_tensor_index(
            manifest);

    const auto execution =
        strix::build_serial_decode_execution_plan(
            model);
    const auto requirements =
        strix::build_serial_execution_tensor_requirements(
            execution);
    const auto tensor_access =
        strix::compile_execution_tensor_access_plan(
            execution,
            manifest,
            tensor_index,
            requirements);

    strix::ExecutionBindingRequest
        binding_request;
    binding_request.logical_rows = 1;
    binding_request.request_concurrency = 1;
    binding_request.minimum_qualification =
        strix::kernels::
            QualificationStatus::
                experimental;

    strix::kernels::KernelRegistry
        kernel_registry;
    strix::register_complete_hip_kernel_catalog(
        model,
        binding_request,
        {},
        kernel_registry);

    strix::ExecutionBufferRequest
        buffer_request;
    buffer_request.logical_rows = 1;
    buffer_request.activation_element_bytes =
        2;
    buffer_request.control_element_bytes =
        4;
    buffer_request.index_element_bytes =
        4;
    buffer_request.logit_element_bytes =
        4;
    buffer_request.mlp_intermediate_element_bytes =
        4;

    const auto compiled =
        strix::compile_serial_execution(
            model,
            kernel_registry,
            binding_request,
            buffer_request,
            256);

    strix::HipLauncherCatalog
        launcher_catalog(
            model,
            {
                strix::HipActivationStorage::
                    bf16,
            });
    strix::ExecutionLauncherRegistry
        launchers;
    launcher_catalog.
        register_for_compiled_execution(
            compiled,
            launchers);
    const auto coverage =
        launcher_catalog.audit(
            compiled,
            launchers);
    if (coverage.missing_steps != 0) {
        throw std::logic_error(
            "HIP launcher catalog is incomplete for compiled execution");
    }

    const auto launch_plan =
        strix::compile_launch_plan(
            compiled,
            launchers);
    strix::DirectLaunchDispatcher
        dispatcher(
            launch_plan);

    strix::SerialSessionStateLayoutRequest
        state_request;
    state_request.context_tokens =
        options.context_tokens;
    const auto state_layout =
        strix::build_serial_session_state_layout(
            model,
            state_request);

    strix::AllocationGenerationSource
        generations;
    auto mapped =
        strix::MappedModelFile::
            open_read_only(
                options.model_path,
                "model",
                generations);

    strix::HipModelResidencyPolicy
        residency_policy;
    residency_policy.mode =
        options.residency;
    residency_policy.coalesce_gap_bytes =
        options.coalesce_gap_bytes;
    residency_policy.
        promote_file_backed_candidates =
            false;

    auto model_residency =
        strix::HipModelResidency::create(
            mapped,
            manifest,
            generations,
            residency_policy);

    if (reference.has_value() &&
        reference->model_content_identity !=
            model_residency.binding().
                model_content_identity) {
        throw std::invalid_argument(
            "reference checkpoint SHA-256 does not match loaded model");
    }

    strix::HipRuntimeAllocator
        allocator(
            strix::HipHostAllocationMode::
                pageable_aligned);
    auto memory =
        strix::RuntimeExecutionMemoryOwner::
            allocate(
                compiled,
                state_layout,
                allocator,
                generations);

    strix::initialize_hip_fresh_runtime_memory(
        memory.scratch(),
        memory.state());

    strix::FileMappedPleLookupBackend
        ple_lookup;

    strix::ExecutionRuntimeResources
        resources;
    resources.scratch =
        &memory.scratch();
    resources.model_manifest =
        &manifest;
    resources.model_weights =
        &model_residency.binding();
    resources.tensor_access =
        &tensor_access;
    resources.session_state_layout =
        &state_layout;
    resources.session_state_memory =
        &memory.state();
    resources.ple_lookup =
        &ple_lookup;

    const auto identities =
        strix::build_runtime_execution_identities(
            compiled.execution_identity,
            compiled.arena,
            resources);

    strix::SerialSessionContext
        session;
    session.execution_identity =
        compiled.execution_identity;
    session.state_layout_identity =
        state_layout.layout_identity;
    session.runtime_compatibility_identity =
        identities.compatibility_identity;
    session.state.valid =
        all_persistent_state();

    const auto& lm_io =
        compiled.io.steps.back();
    if (lm_io.op !=
            strix::SerialOp::lm_head ||
        lm_io.writes.size() != 1) {
        throw std::logic_error(
            "compiled execution does not end in one logits write");
    }

    std::vector<float> logits(
        static_cast<std::size_t>(
            model.vocab_size));

    strix::SerialDecodeReplayObservation
        replay_observation;
    if (reference.has_value()) {
        replay_observation.model_content_identity =
            model_residency.binding().
                model_content_identity;
        replay_observation.compiled_execution_identity =
            compiled.execution_identity;
        replay_observation.runtime_compatibility_identity =
            identities.compatibility_identity;
        replay_observation.independent_reference_identity =
            reference->independent_reference_identity;
        replay_observation.input_tokens =
            options.tokens;
        replay_observation.reference_argmax_tokens.reserve(
            reference->steps.size());
        replay_observation.engine_argmax_tokens.reserve(
            reference->steps.size());
    }

    std::cout << '{' << '\n'
              << "  \"model_content_identity\": \""
              << strix::model_content_identity_hex(
                     model_residency.binding().model_content_identity)
              << "\"," << '\n'
              << "  \"execution_identity\": "
              << compiled.execution_identity
              << ',' << '\n'
              << "  \"launch_plan_identity\": "
              << launch_plan.plan_identity
              << ',' << '\n'
              << "  \"runtime_compatibility_identity\": "
              << identities.compatibility_identity
              << ',' << '\n'
              << "  \"context_tokens\": "
              << options.context_tokens
              << ',' << '\n'
              << "  \"residency\": \""
              << (options.residency ==
                          strix::HipModelResidencyMode::
                              registered_file_mapping
                      ? "register"
                      : "copy")
              << "\"," << '\n'
              << "  \"registered_bytes\": "
              << model_residency.registered_bytes()
              << ',' << '\n'
              << "  \"copied_bytes\": "
              << model_residency.copied_bytes()
              << ',' << '\n'
              << "  \"tokens\": [" << '\n';

    for (std::size_t input_index = 0;
         input_index <
            options.tokens.size();
         ++input_index) {
        const auto token =
            options.tokens[input_index];

        static_cast<void>(
            strix::execute_serial_token(
                compiled,
                session,
                resources,
                token,
                dispatcher));

        check_hip(
            hipDeviceSynchronize(),
            "hipDeviceSynchronize(serial token)");

        const auto logits_address =
            strix::resolve_execution_buffer(
                compiled.execution_identity,
                compiled.arena,
                memory.scratch(),
                lm_io.writes[0]);
        if (logits_address.bytes <
            logits.size() *
                sizeof(float)) {
            throw std::logic_error(
                "compiled logits allocation is undersized");
        }

        check_hip(
            hipMemcpy(
                logits.data(),
                reinterpret_cast<
                    const void*>(
                        logits_address.address),
                logits.size() *
                    sizeof(float),
                hipMemcpyDeviceToHost),
            "hipMemcpy(logits D2H)");

        strix::validate_serial_decode_engine_logits(
            logits,
            model.vocab_size);

        const auto ranked =
            top_logits(
                logits,
                5);
        if (ranked.empty()) {
            throw std::logic_error(
                "LM head produced no logits");
        }

        if (reference.has_value()) {
            const auto& expected =
                reference->steps[input_index];
            replay_observation.reference_argmax_tokens.push_back(
                expected.expected_argmax_token);
            replay_observation.engine_argmax_tokens.push_back(
                ranked.front().token);

            for (const auto& probe :
                 expected.logits) {
                if (probe.index >=
                    logits.size()) {
                    throw std::out_of_range(
                        "reference logit probe index exceeds model vocabulary");
                }
                const auto error =
                    std::abs(
                        static_cast<double>(
                            logits[probe.index]) -
                        static_cast<double>(
                            probe.value));
                replay_observation.maximum_logit_abs_error =
                    std::max(
                        replay_observation.maximum_logit_abs_error,
                        error);
                ++replay_observation.logit_values_compared;
            }
        }

        std::cout
            << "    {\"input_token\":"
            << token
            << ",\"position\":"
            << (session.next_position - 1ULL)
            << ",\"argmax_token\":"
            << ranked.front().token
            << ",\"argmax_logit\":"
            << ranked.front().logit
            << ",\"top5\":[";

        for (std::size_t rank = 0;
             rank < ranked.size();
             ++rank) {
            if (rank != 0) {
                std::cout << ',';
            }
            std::cout
                << "{\"token\":"
                << ranked[rank].token
                << ",\"logit\":"
                << ranked[rank].logit
                << '}';
        }
        std::cout << "]}";
        if (input_index + 1U !=
            options.tokens.size()) {
            std::cout << ',';
        }
        std::cout << '\n';
    }

    std::optional<strix::SerialDecodeReplayAssessment>
        replay_assessment;
    if (reference.has_value()) {
        strix::SerialDecodeReplayPolicy policy;
        policy.policy_identity =
            reference->independent_reference_identity;
        policy.max_logit_abs_error =
            options.max_logit_abs_error;
        policy.minimum_steps =
            static_cast<std::uint64_t>(
                reference->steps.size());
        policy.minimum_logit_values_compared =
            required_reference_logit_values;
        policy.require_exact_argmax = true;

        replay_assessment =
            strix::assess_serial_decode_replay(
                policy,
                replay_observation);
    }

    std::cout
        << "  ]," << '\n'
        << "  \"committed_tokens\": "
        << session.committed_tokens
        << ',' << '\n'
        << "  \"next_position\": "
        << session.next_position;

    if (replay_assessment.has_value()) {
        std::cout
            << ',' << '\n'
            << "  \"replay\": {" << '\n'
            << "    \"reference_id\": "
            << reference->independent_reference_identity
            << ',' << '\n'
            << "    \"logit_values_compared\": "
            << replay_observation.logit_values_compared
            << ',' << '\n'
            << "    \"max_logit_abs_error\": "
            << replay_observation.maximum_logit_abs_error
            << ',' << '\n'
            << "    \"argmax_match\": "
            << (replay_assessment->argmax_match
                    ? "true"
                    : "false")
            << ',' << '\n'
            << "    \"passed\": "
            << (replay_assessment->passes_declared_policy
                    ? "true"
                    : "false")
            << '\n'
            << "  }";
    }

    std::cout
        << '\n'
        << '}' << '\n';

    return replay_assessment.has_value() &&
                   !replay_assessment->passes_declared_policy
               ? 2
               : 0;
} catch (const std::exception& error) {
    std::cerr
        << "strix-hip-decode-token: "
        << error.what() << '\n';
    return 1;
}
