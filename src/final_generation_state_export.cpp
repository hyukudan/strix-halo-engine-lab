#include "strix/final_generation_state_export.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/session_snapshot.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <exception>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <vector>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strix {
namespace {
constexpr std::uint64_t manifest_limit = 1024U * 1024U;
constexpr std::size_t chunk_bytes = 65536U;
constexpr std::array names{"attention-kv", "recurrent-matrix", "gdn-convolution", "ple-convolution",
                           "sparse-index", "ngram-history", "position-history"};
void require(bool value, const char* message) { if (!value) throw std::invalid_argument(message); }
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) throw std::overflow_error("final state export size overflow");
    return a + b;
}
std::filesystem::path destination(const std::filesystem::path& path) {
    require(!path.empty(), "final state export destination is empty");
    const auto absolute = std::filesystem::absolute(path).lexically_normal();
    require(absolute.has_filename() && absolute.filename() != "." && absolute.filename() != "..",
            "final state export destination is invalid");
    const auto parent = std::filesystem::canonical(absolute.parent_path());
    require(std::filesystem::is_directory(parent), "final state export parent is not a directory");
    return parent / absolute.filename();
}
void absent(const std::filesystem::path& path) {
    // symlink_status rejects a dangling symlink occupying this name too.
    require(!std::filesystem::exists(std::filesystem::symlink_status(path)), "final state export path already exists");
}
std::filesystem::path staging(const std::filesystem::path& target) {
    return target.parent_path() / (target.filename().string() + ".partial");
}
void publish(const std::filesystem::path& from, const std::filesystem::path& to) {
#if defined(__linux__) && defined(SYS_renameat2)
    constexpr unsigned rename_noreplace = 1U;
    if (syscall(SYS_renameat2, AT_FDCWD, from.c_str(), AT_FDCWD, to.c_str(), rename_noreplace) != 0)
        throw std::system_error(errno, std::generic_category(), "atomic final state no-replace publication");
#elif defined(_WIN32)
    if (!MoveFileW(from.c_str(), to.c_str()))
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "atomic final state no-replace publication");
#else
    (void)from; (void)to;
    throw std::runtime_error("atomic no-replace final state directory publication is unsupported on this platform");
#endif
}
std::string quote(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    std::string out = "\"";
    for (const auto c : value) {
        const auto b = static_cast<unsigned char>(c);
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (b < 0x20U) { out += "\\u00"; out += hex[b >> 4U]; out += hex[b & 15U]; }
        else out += c;
    }
    return out + '"';
}
bool hex(std::string_view value, std::size_t size) {
    return value.size() == size && std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
template<class Range> void values(std::ostream& stream, const Range& range) {
    stream << '['; bool first = true;
    for (const auto value : range) { if (!first) stream << ','; first = false; stream << value; }
    stream << ']';
}
void write(const std::filesystem::path& path, std::string_view text) {
    std::ofstream file(path, std::ios::binary | std::ios::out);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.write(text.data(), static_cast<std::streamsize>(text.size())); file.close();
}
void drain(const std::function<void()>& operation) noexcept {
    // A throwing drain is never a recoverable proof of ownership safety.
    try { operation(); } catch (...) { std::terminate(); }
}
std::array<const SessionStateAllocation*, 7> allocations(const SerialSessionStateLayout& layout) {
    require(layout.layout_identity && layout.context_tokens && layout.allocations.size() == names.size(),
            "final state export requires seven actual components");
    std::array<const SessionStateAllocation*, 7> out{};
    for (const auto& allocation : layout.allocations) {
        const auto index = static_cast<std::size_t>(allocation.component);
        require(index < out.size() && !out[index] && allocation.bytes,
                "final state export component missing, duplicate or empty");
        require(allocation.storage == SessionStateStorage::device || allocation.storage == SessionStateStorage::host,
                "final state export storage invalid");
        out[index] = &allocation;
    }
    for (const auto* allocation : out) require(allocation != nullptr, "final state export component missing");
    return out;
}
void validate_result(const SerialGreedyRequest& request, const SerialGreedyRequestResult& result,
                     const SerialSessionContext& session) {
    require(!request.prefix_tokens.empty() && request.max_new_tokens && request.vocab_size &&
            !result.generated_tokens.empty() && result.generated_tokens.size() <= request.max_new_tokens,
            "final state export requires a completed nonempty greedy request");
    const auto consumed = add(request.prefix_tokens.size(), result.generated_tokens.size() - 1U);
    require(result.steps_executed == consumed && result.final_position == consumed &&
            session.next_position == consumed && session.committed_tokens == consumed && consumed <= request.context_capacity,
            "final state export consumed frontier differs from actual request/session");
    require(session.health == SerialSessionHealth::ready && static_cast<std::uint32_t>(session.state.valid) == 63U &&
            arithmetic_lineage_valid(session.arithmetic_lineage, consumed), "final state export session is unhealthy or incomplete");
    for (auto value : request.prefix_tokens) require(value < request.vocab_size, "final state prefix token outside vocabulary");
    for (auto value : result.generated_tokens) require(value < request.vocab_size, "final state generated token outside vocabulary");
    require(request.eos_token_id < request.vocab_size &&
            std::find(result.generated_tokens.begin(), result.generated_tokens.end() - 1, request.eos_token_id) == result.generated_tokens.end() - 1,
            "final state export has an earlier EOS");
    const bool eos = result.generated_tokens.back() == request.eos_token_id;
    if (result.stop_reason == SerialRequestStopReason::eos) require(eos, "final state EOS stop mismatch");
    else {
        require(!eos, "final state export ignored EOS");
        if (result.stop_reason == SerialRequestStopReason::context_capacity)
            require(consumed == request.context_capacity, "final state context stop mismatch");
        else require(result.stop_reason == SerialRequestStopReason::max_new_tokens &&
                     result.generated_tokens.size() == request.max_new_tokens && consumed < request.context_capacity,
                     "final state maximum token stop mismatch");
    }
}
} // namespace
void validate_final_generation_state_destination(const std::filesystem::path& path) {
    const auto target = destination(path); absent(target); absent(staging(target));
}
void validate_final_generation_state_budget(const SerialSessionStateLayout& layout, FinalGenerationStatePolicy policy) {
    auto bytes = manifest_limit;
    for (const auto* allocation : allocations(layout)) bytes = add(bytes, allocation->bytes);
    require(bytes <= policy.maximum_bytes, "final state export byte budget is insufficient");
}
void export_final_generation_state(const std::filesystem::path& path, FinalGenerationStatePolicy policy,
    const ModelContract& model, const SerialSessionStateLayout& layout, const SessionStateMemoryBinding& binding,
    const SerialSessionContext& session, const SerialGreedyRequest& request, const SerialGreedyRequestResult& result,
    const FinalGenerationStateMetadata& metadata, std::shared_ptr<const void> owner,
    const RawBaselineStateReader& reader, const std::function<void()>& require_binding,
    const std::function<void()>& quiesce_or_fail_stop, const std::function<void()>& before_publication) {
    require(owner && reader && require_binding && quiesce_or_fail_stop && before_publication,
            "final state export requires actual owner, reader, guards and drain");
    // Retain this bounded copy destination and owner through every read failure
    // until the catch proves original quiescence, before either can be freed.
    std::vector<std::byte> chunk;
    try {
        chunk.resize(chunk_bytes);
        validate_final_generation_state_destination(path); validate_final_generation_state_budget(layout, policy);
        validate_result(request, result, session); validate_session_state_memory_binding(layout, binding);
        require(session.state_layout_identity == layout.layout_identity && request.context_capacity == layout.context_tokens,
                "final state export layout/session mismatch");
        require(hex(metadata.model_sha256,64) && hex(metadata.model_content_identity,64) && hex(metadata.executable_sha256,64) &&
                hex(metadata.gpu_identity,32) && metadata.gpu_identity != std::string(32,'0') &&
                !metadata.numerical_identity.empty() && metadata.numerical_identity.size() <= 512U &&
                !metadata.output_mode.empty() && metadata.output_mode.size()<=128U &&
                !metadata.gpu_architecture.empty() && metadata.gpu_architecture.size()<=128U &&
                metadata.kernel_plan_identity && metadata.launch_plan_identity && metadata.runtime_binding_identity &&
                metadata.hip_runtime_version && metadata.hip_driver_version,
                "final state export provenance is incomplete");
        // Bound variable-length JSON before constructing it. 22 bytes cover two
        // appearances of every uint32 token including separators; 384 covers
        // each interval's six uint64 values, enum and field names.
        const auto token_count=add(request.prefix_tokens.size(),result.generated_tokens.size());
        require(token_count<=(manifest_limit-32768U)/22U,"final state token metadata exceeds reserved budget");
        const auto token_metadata=token_count*22U;
        require(session.arithmetic_lineage.intervals.size()<=(manifest_limit-32768U-token_metadata)/384U,
                "final state lineage metadata exceeds reserved budget");
        const auto snapshot = build_serial_session_snapshot_plan(model, layout,
            {session.execution_identity, session.runtime_compatibility_identity, session.next_position,
             session.committed_tokens, session.state, 64});
        require(snapshot.ranges.size() == 7U, "final state snapshot is incomplete");
        const auto ordered = allocations(layout);
        for (std::size_t i=0; i<ordered.size(); ++i) {
            const auto address=resolve_session_state(layout,binding,ordered[i]->component);
            for (std::size_t j=0; j<i; ++j)
                if (ordered[i]->storage==ordered[j]->storage)
                    require(!(ordered[i]->offset<add(ordered[j]->offset,ordered[j]->bytes) &&
                              ordered[j]->offset<add(ordered[i]->offset,ordered[i]->bytes)),
                            "final state component ranges overlap");
            require(address.bytes==ordered[i]->bytes,"final state component range changed");
        }
        drain(quiesce_or_fail_stop); require_binding();
        const auto target = destination(path), temporary = staging(target);
        require(std::filesystem::create_directory(temporary), "final state export staging already exists");
        std::filesystem::permissions(temporary,std::filesystem::perms::owner_all,std::filesystem::perm_options::replace);
        std::ostringstream manifest;
        manifest << "{\"schema\":\"strix-final-generation-state-v1\",\"timing_invalid\":true,\"logits_exported\":false"
                 << ",\"model_sha256\":" << quote(metadata.model_sha256)
                 << ",\"model_content_identity\":" << quote(metadata.model_content_identity)
                 << ",\"executable_sha256\":" << quote(metadata.executable_sha256)
                 << ",\"numerical_identity\":" << quote(metadata.numerical_identity)
                 << ",\"output_mode\":" << quote(metadata.output_mode)
                 << ",\"gpu_identity\":" << quote(metadata.gpu_identity)
                 << ",\"gpu_architecture\":" << quote(metadata.gpu_architecture)
                 << ",\"hip_runtime_version\":" << metadata.hip_runtime_version << ",\"hip_driver_version\":" << metadata.hip_driver_version
                 << ",\"execution_identity\":" << session.execution_identity << ",\"runtime_compatibility_identity\":" << session.runtime_compatibility_identity
                 << ",\"runtime_binding_identity\":" << metadata.runtime_binding_identity << ",\"kernel_plan_identity\":" << metadata.kernel_plan_identity
                 << ",\"launch_plan_identity\":" << metadata.launch_plan_identity << ",\"state_layout_identity\":" << layout.layout_identity
                 << ",\"snapshot_plan_identity\":" << snapshot.plan_identity
                 << ",\"state_binding\":{\"device_base\":" << binding.device_base << ",\"device_bytes\":" << binding.device_bytes
                 << ",\"device_generation\":" << binding.device_allocation_generation << ",\"host_base\":" << binding.host_base
                 << ",\"host_bytes\":" << binding.host_bytes << ",\"host_generation\":" << binding.host_allocation_generation << '}'
                 << ",\"prepared_eager\":" << (metadata.prepared_eager ? "true" : "false")
                 << ",\"shared_overlap\":" << (metadata.shared_overlap ? "true" : "false")
                 << ",\"batched_moe_activation\":" << (metadata.batched_moe_activation ? "true" : "false")
                 << ",\"next_position\":" << session.next_position << ",\"committed_tokens\":" << session.committed_tokens
                 << ",\"steps_executed\":" << result.steps_executed << ",\"state_health\":\"ready\",\"state_valid\":63,\"state_generations\":";
        values(manifest,session.state.generations);
        manifest << ",\"prefix_tokens\":"; values(manifest,request.prefix_tokens);
        manifest << ",\"generated_tokens\":"; values(manifest,result.generated_tokens);
        manifest << ",\"consumed_input_tokens\":[";
        bool first = true;
        const auto token = [&](std::uint32_t value) { if (!first) manifest << ','; first = false; manifest << value; };
        for (auto value : request.prefix_tokens) token(value);
        for (auto i = result.generated_tokens.begin(); i != result.generated_tokens.end() - 1; ++i) token(*i);
        manifest << "],\"unconsumed_emitted_token\":" << result.generated_tokens.back()
                 << ",\"stop_reason\":" << quote(result.stop_reason == SerialRequestStopReason::eos ? "eos" :
                    result.stop_reason == SerialRequestStopReason::context_capacity ? "context_capacity" : "max_new_tokens")
                 << ",\"context_capacity\":" << request.context_capacity << ",\"maximum_generated_tokens\":" << request.max_new_tokens
                 << ",\"vocabulary_size\":" << request.vocab_size
                 << ",\"eos_token\":" << request.eos_token_id << ",\"arithmetic_lineage\":{\"history_identity\":"
                 << session.arithmetic_lineage.history_identity << ",\"tainted\":false,\"intervals\":[";
        for (std::size_t i=0; i<session.arithmetic_lineage.intervals.size(); ++i) {
            const auto& interval=session.arithmetic_lineage.intervals[i]; if (i) manifest << ',';
            manifest << "{\"begin\":" << interval.begin << ",\"end\":" << interval.end << ",\"producer\":" << static_cast<unsigned>(interval.producer)
                     << ",\"numerical_plan_identity\":" << interval.numerical_plan_identity << ",\"reduction_plan_identity\":" << interval.reduction_plan_identity
                     << ",\"state_update_identity\":" << interval.state_update_identity << '}';
        }
        manifest << "]},\"components\":[";
        for (std::size_t i=0; i<ordered.size(); ++i) {
            const auto& allocation=*ordered[i];
            const auto filename=std::string(names[i])+".raw";
            std::ofstream file(temporary/filename,std::ios::binary|std::ios::out);
            file.exceptions(std::ios::badbit|std::ios::failbit);
            for (std::uint64_t offset=0; offset<allocation.bytes;) {
                const auto count=static_cast<std::size_t>(std::min<std::uint64_t>(chunk.size(),allocation.bytes-offset));
                require_binding(); reader(allocation,offset,std::span{chunk}.first(count)); require_binding();
                file.write(reinterpret_cast<const char*>(chunk.data()),static_cast<std::streamsize>(count)); offset+=count;
            }
            file.close();
            const auto hash=model_content_identity_hex(sha256_file_content(temporary/filename));
            if (i) manifest << ',';
            manifest << "{\"component\":" << quote(names[i]) << ",\"state_flag\":" << static_cast<unsigned>(allocation.state_flag)
                     << ",\"storage\":" << quote(allocation.storage==SessionStateStorage::device ? "device" : "host")
                     << ",\"layout_offset\":" << allocation.offset << ",\"provisional\":" << (allocation.provisional ? "true" : "false")
                     << ",\"path\":" << quote(filename) << ",\"bytes\":" << allocation.bytes << ",\"sha256\":" << quote(hash) << '}';
        }
        manifest << "]}\n";
        const auto text=manifest.str(); require(text.size()<=manifest_limit,"final state manifest exceeds reserved budget");
        write(temporary/"manifest.json",text);
        before_publication(); require_binding();
        // No generic rename fallback: an existing destination is never replaced.
        publish(temporary,target);
    } catch (...) {
        drain(quiesce_or_fail_stop);
        throw;
    }
}
} // namespace strix
