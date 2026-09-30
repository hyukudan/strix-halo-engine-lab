#include "strix/session_state_layout.hpp"

#include "strix/memory_plan.hpp"

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

    strix::SerialSessionStateLayoutRequest request;
    request.context_tokens = 32'768;
    request.device_alignment = 256;
    request.host_alignment = 64;

    const auto layout =
        strix::build_serial_session_state_layout(
            model, request);

    expect(layout.layout_identity != 0,
           "session-state layout has a stable non-zero identity");
    expect(layout.allocations.size() == 7,
           "serial state layout exposes seven explicit components");

    const auto* kv =
        strix::find_session_state_allocation(
            layout,
            strix::SessionStateComponent::attention_kv);
    expect(kv != nullptr &&
               kv->bytes ==
                   strix::base_kv_bytes_per_token(model) *
                       request.context_tokens &&
               kv->storage ==
                   strix::SessionStateStorage::device,
           "attention KV receives exact context-sized device storage");

    const auto* ple =
        strix::find_session_state_allocation(
            layout,
            strix::SessionStateComponent::ple_convolution);
    expect(ple != nullptr &&
               ple->bytes == 180ULL * 1024ULL &&
               !ple->provisional,
           "PLE convolution state is an exact device allocation");

    const auto* sparse =
        strix::find_session_state_allocation(
            layout,
            strix::SessionStateComponent::sparse_index);
    expect(sparse != nullptr &&
               !sparse->provisional &&
               sparse->bytes ==
                   strix::provisional_indexer_bytes_per_token(model) *
                       request.context_tokens,
           "QSA raw-key cache is an exact context-sized device allocation");

    const auto* ngram =
        strix::find_session_state_allocation(
            layout,
            strix::SessionStateComponent::ngram_history);
    const auto* position =
        strix::find_session_state_allocation(
            layout,
            strix::SessionStateComponent::position_history);
    expect(ngram != nullptr &&
               position != nullptr &&
               ngram->storage ==
                   strix::SessionStateStorage::host &&
               position->storage ==
                   strix::SessionStateStorage::host,
           "small resume histories remain explicit host state");

    strix::MemoryPlanRequest memory_request;
    memory_request.context_tokens =
        request.context_tokens;
    memory_request.sessions = 1;
    memory_request.include_mtp_kv = false;
    memory_request.include_indexer = true;
    const auto memory =
        strix::build_memory_plan(
            model, memory_request);

    const auto exact_unaligned =
        memory.base_kv +
        memory.indexer +
        memory.gdn_state +
        memory.ple_conv_state +
        memory.ngram_history +
        memory.position_history;

    expect(layout.device_total_bytes +
               layout.host_total_bytes >=
               exact_unaligned,
           "aligned session layout covers memory-plan state bytes");
    expect(layout.device_total_bytes %
                   request.device_alignment ==
               0 &&
               layout.host_total_bytes %
                   request.host_alignment ==
               0,
           "session state pools honor independent host/device alignment");

    auto no_sparse = request;
    no_sparse.include_sparse_index_candidate = false;
    const auto without_sparse =
        strix::build_serial_session_state_layout(
            model, no_sparse);
    expect(
        strix::find_session_state_allocation(
            without_sparse,
            strix::SessionStateComponent::sparse_index) ==
            nullptr,
        "session layout can omit QSA raw-key state");
    expect(without_sparse.layout_identity !=
               layout.layout_identity,
           "sparse-state policy changes layout identity");

    auto invalid = request;
    invalid.context_tokens =
        model.max_position_embeddings + 1ULL;
    bool context_rejected = false;
    try {
        static_cast<void>(
            strix::build_serial_session_state_layout(
                model, invalid));
    } catch (const std::invalid_argument&) {
        context_rejected = true;
    }
    expect(context_rejected,
           "session state layout rejects context beyond model maximum");

    if (failures == 0) {
        std::cout << "session state layout tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
