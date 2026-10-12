#include "strix/raw_baseline_recording.hpp"
#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {
using namespace strix;
int failures = 0;
void expect(bool good, const char* message) { if (!good) { ++failures; std::cerr << message << '\n'; } }
template<class Function> bool rejects(Function&& function) { try { function(); } catch (const std::exception&) { return true; } return false; }
struct Private {
    std::filesystem::path parent;
    Private() {
        static unsigned sequence = 0;
        parent = std::filesystem::temp_directory_path() / ("strix-raw-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + '-' + std::to_string(++sequence));
        std::filesystem::create_directory(parent);
    }
    ~Private() { std::error_code error; std::filesystem::remove_all(parent, error); }
    std::filesystem::path output() const { return parent / "recording"; }
};
std::string read(const std::filesystem::path& path) { std::ifstream input(path, std::ios::binary); return {std::istreambuf_iterator<char>{input}, {}}; }
SerialSessionStateLayout layout(std::uint64_t context = 4) {
    SerialSessionStateLayout result; result.layout_identity = 55; result.context_tokens = context; result.device_total_bytes = 40; result.host_total_bytes = 16;
    for (unsigned n = 0; n < 7; ++n) result.allocations.push_back({static_cast<SessionStateComponent>(n), static_cast<PersistentState>(1U << (n < 3 ? n : n - 1)),
        n >= 5 ? SessionStateStorage::host : SessionStateStorage::device, static_cast<std::uint64_t>(n >= 5 ? n - 5 : n) * 8, 8, false});
    return result;
}
RawBaselineMetadata metadata(bool greedy = false) {
    RawBaselineMetadata result;
    result.model_sha256 = std::string(64, '1'); result.model_content_identity = std::string(64, '2'); result.executable_sha256 = std::string(64, '3');
    result.gpu_identity = "injected-gpu"; result.gpu_architecture = "fixture-only";
    result.execution_identity = 11; result.kernel_plan_identity = 12; result.runtime_compatibility_identity = 13; result.vocabulary_size = 3;
    result.supplied_tokens = greedy ? std::vector<std::uint32_t>{1} : std::vector<std::uint32_t>{1, 2};
    if (greedy) { result.mode = RawBaselineMode::greedy; result.maximum_generated_tokens = 2; result.eos_token = 2; }
    return result;
}
RawBaselinePolicy policy(RawBaselineStatePolicy state_policy = RawBaselineStatePolicy::final) { return {1024 * 1024, 2, 4, state_policy}; }
SerialSessionContext session(std::uint64_t next = 0) {
    SerialSessionContext result; result.execution_identity = 11; result.state_layout_identity = 55; result.runtime_compatibility_identity = 13;
    result.next_position = result.committed_tokens = next; result.state.valid = static_cast<PersistentState>(63); result.state.generations.fill(next); return result;
}
struct CopyState { unsigned calls = 0; std::size_t max_chunk = 0; std::byte* exposed = nullptr; bool fail = false; };
RawBaselineStateReader reader(const std::shared_ptr<CopyState>& source) {
    return [source](const SessionStateAllocation& allocation, std::uint64_t offset, std::span<std::byte> destination) {
        ++source->calls; source->max_chunk = std::max(source->max_chunk, destination.size());
        for (std::size_t n = 0; n < destination.size(); ++n) destination[n] = static_cast<std::byte>((static_cast<unsigned>(allocation.component) * 17 + offset + n) & 255U);
        if (source->fail) { source->exposed = destination.data(); throw std::runtime_error("injected unknown copy outcome"); }
    };
}
void write_teacher(const std::filesystem::path& output, RawBaselineStatePolicy state_policy = RawBaselineStatePolicy::final) {
    auto copy = std::make_shared<CopyState>(); auto transfer = reader(copy);
    RawBaselineRecording recording(output, metadata(), policy(state_policy), layout());
    recording.capture_initial(session(), transfer);
    const std::array first{3.0F, 3.0F, -1.0F}, second{-2.0F, 3.0F, 1.0F};
    recording.record_token(1, 0, first, session(1), transfer, 2);
    recording.record_token(2, 1, second, session(2), transfer);
    recording.finalize(session(2), transfer);
    expect(copy->max_chunk == 4 && copy->calls == (state_policy == RawBaselineStatePolicy::final ? 28U : 42U), "initial/final or each copies use bounded actual allocation chunks");
}
void complete_raw_and_greedy() {
    Private private_files; write_teacher(private_files.output());
    const auto output = private_files.output(); const auto receipt = read(output / "receipt.json"), rows = read(output / "steps.jsonl"), states = read(output / "states.jsonl");
    expect(!std::filesystem::exists(output / "INCOMPLETE") && receipt.find("\"complete\":true") != std::string::npos && receipt.find("\"qualified\":false") != std::string::npos && receipt.find("\"evidence_kind\":\"injected-fixture\"") != std::string::npos, "completed fixture keeps evidence separate from native qualification");
    expect(receipt.find("\"performance_evidence\":false") != std::string::npos && receipt.find("\"generated_output_tokens\":[]") != std::string::npos, "teacher-forced data invents neither generated output nor performance proof");
    expect(receipt.find("\"activation_storage\":\"bf16\"") != std::string::npos, "actual activation storage is distinct from full FP32 output dtype");
    expect(read(output / "step-0.f32le") == std::string("\x00\x00\x40\x40\x00\x00\x40\x40\x00\x00\x80\xbf", 12), "complete FP32 logits preserve exact little-endian bits and vocabulary order");
    expect(read(output / "history.u32le") == std::string("\x01\x00\x00\x00\x02\x00\x00\x00", 8), "history records exact fed tokens little endian");
    expect(rows.find("\"argmax_token\":0") != std::string::npos && rows.find("\"teacher_forced_target\":2") != std::string::npos && rows.find("\"teacher_forced_target\":null") != std::string::npos, "first-index ties and explicit teacher-forced targets differ from argmax");
    for (const char* component : {"attention_kv", "recurrent_matrix", "gdn_convolution", "ple_convolution", "sparse_index", "ngram_history", "position_history"}) expect(states.find(component) != std::string::npos, "all seven components covering six state classes are recorded");
    expect(states.find("\"kind\":\"initial\"") != std::string::npos && states.find("\"kind\":\"final\"") != std::string::npos && states.find("\"next_position\":2") != std::string::npos, "initial and final actual session positions are explicit");
    const auto expected_hash = model_content_identity_hex(sha256_file_content(output / "step-0.f32le")); expect(rows.find(expected_hash) != std::string::npos, "artifact hash independently matches serialized complete logits");
    Private each; write_teacher(each.output(), RawBaselineStatePolicy::each); expect(read(each.output() / "states.jsonl").find("post-1") != std::string::npos, "each policy records final executed token state");
    Private generated; auto copy = std::make_shared<CopyState>(); auto transfer = reader(copy);
    RawBaselineRecording recording(generated.output(), metadata(true), policy(), layout()); recording.capture_initial(session(), transfer);
    const std::array logits{-1.0F, 5.0F, 2.0F}; recording.record_token(1, 0, logits, session(1), transfer); recording.record_token(1, 1, logits, session(2), transfer);
    const std::array<std::uint32_t, 2> outputs{1, 1}; recording.finalize(session(2), transfer, outputs, "max_new_tokens");
    expect(read(generated.output() / "receipt.json").find("\"generated_output_tokens\":[1,1]") != std::string::npos && read(generated.output() / "steps.jsonl").find("generation_fed") != std::string::npos, "actual emitted greedy outputs are independently distinguished from fed history");
}
void rejects_before_creation() {
    for (unsigned mutation = 0; mutation < 11; ++mutation) {
        Private p; auto m = metadata(); auto bounds = policy(); auto state_layout = layout();
        switch (mutation) {
        case 0: bounds.state_policy = static_cast<RawBaselineStatePolicy>(99); break;
        case 1: m.mode = static_cast<RawBaselineMode>(99); break;
        case 2: m.evidence = static_cast<RawBaselineEvidence>(99); break;
        case 3: bounds.maximum_bytes = 8; break;
        case 4: bounds.maximum_steps = std::numeric_limits<std::uint64_t>::max(); break;
        case 5: state_layout.allocations.pop_back(); break;
        case 6: state_layout.allocations[1].offset = 0; break;
        case 7: m.evidence = RawBaselineEvidence::native_eager; break;
        case 8: bounds.chunk_bytes = 3; break;
        case 9: state_layout.allocations[1].bytes = std::numeric_limits<std::uint64_t>::max(); break;
        case 10: m.activation_storage = static_cast<RawBaselineActivationStorage>(99); break;
        }
        expect(rejects([&] { RawBaselineRecording rejected(p.output(), m, bounds, state_layout); }) && !std::filesystem::exists(p.output()), "invalid enum/native metadata/geometry/budget rejected before any artifact creation");
    }
    Private p; std::filesystem::create_directory(p.parent / ".git"); expect(rejects([&] { validate_raw_baseline_output_directory(p.output()); }), "Git ancestor rejected");
    expect(rejects([&] { validate_raw_baseline_output_directory("relative-recording"); }), "relative output rejected");
    Private existing; std::filesystem::create_directory(existing.output()); expect(rejects([&] { RawBaselineRecording rejected(existing.output(), metadata(), policy(), layout()); }), "existing recording never overwritten");
}
void failed_tokens_never_finalize() {
    for (unsigned mutation = 0; mutation < 6; ++mutation) {
        Private p; auto copy = std::make_shared<CopyState>(); auto transfer = reader(copy); RawBaselineRecording recording(p.output(), metadata(), policy(), layout()); recording.capture_initial(session(), transfer);
        std::array logits{1.0F, 2.0F, 3.0F}; auto state = session(1); std::uint64_t position = 0; std::optional<std::uint32_t> target = 2;
        switch (mutation) { case 0: logits[1] = std::numeric_limits<float>::infinity(); break; case 1: logits[0] = std::numeric_limits<float>::quiet_NaN(); break;
        case 2: state.state.valid = PersistentState::attention_kv; break; case 3: ++state.runtime_compatibility_identity; break; case 4: ++position; break; case 5: target = 1; break; }
        expect(rejects([&] { recording.record_token(1, position, logits, state, transfer, target); }), "nonfinite, stale state/history and invented teacher target rejected");
        expect(rejects([&] { recording.finalize(state, transfer); }) && !std::filesystem::exists(p.output() / "receipt.json"), "failed token cannot publish success");
    }
    Private changed; auto copy = std::make_shared<CopyState>(); auto transfer = reader(copy); RawBaselineRecording recording(changed.output(), metadata(), policy(), layout()); recording.capture_initial(session(), transfer);
    const std::array logits{1.0F, 2.0F, 3.0F}; recording.record_token(1, 0, logits, session(1), transfer, 2); recording.record_token(2, 1, logits, session(2), transfer);
    { std::ofstream mutate(changed.output() / "step-0.f32le", std::ios::binary); mutate << "altered"; }
    expect(rejects([&] { recording.finalize(session(2), transfer); }) && !std::filesystem::exists(changed.output() / "receipt.json"), "artifact drift prevents final publication");
}
void uncertain_copy_and_publication() {
    Private p; auto source = std::make_shared<CopyState>(); source->fail = true; std::weak_ptr<CopyState> retained = source; auto transfer = reader(source);
    {
        RawBaselineRecording recording(p.output(), metadata(), policy(), layout());
        expect(rejects([&] { recording.capture_initial(session(), transfer); }), "unknown partial copy is failed, not silently completed");
        expect(rejects([&] { recording.finalize(session(), transfer); }) && !std::filesystem::exists(p.output() / "receipt.json"), "unknown copy cannot create final receipt");
    }
    transfer = {}; source.reset(); const auto held = retained.lock(); expect(held && held->exposed && *held->exposed == std::byte{0}, "strong captured source owner and destination survive failed recorder destruction");
    Private binding; auto copy = std::make_shared<CopyState>(); auto reader_ok = reader(copy); RawBaselineRecording recorder(binding.output(), metadata(), policy(), layout()); recorder.capture_initial(session(), reader_ok);
    const std::array logits{1.0F, 2.0F, 3.0F}; recorder.record_token(1, 0, logits, session(1), reader_ok, 2); recorder.record_token(2, 1, logits, session(2), reader_ok);
    expect(rejects([&] { recorder.finalize(session(2), reader_ok, {}, "raw_input_exhausted", [] { throw std::runtime_error("observed native binding changed"); }); }) && !std::filesystem::exists(binding.output() / "receipt.json") && std::filesystem::exists(binding.output() / "INCOMPLETE"), "late native binding rejection leaves no success publisher");
    Private marker; RawBaselineRecording publication(marker.output(), metadata(), policy(), layout()); publication.capture_initial(session(), reader_ok);
    publication.record_token(1, 0, logits, session(1), reader_ok, 2); publication.record_token(2, 1, logits, session(2), reader_ok);
    expect(rejects([&] { publication.finalize(session(2), reader_ok, {}, "raw_input_exhausted", [&] {
        std::filesystem::remove(marker.output() / "INCOMPLETE"); std::filesystem::create_directory(marker.output() / "INCOMPLETE");
        std::ofstream obstruction(marker.output() / "INCOMPLETE/blocked"); obstruction << "injected publication failure";
    }); }) && !std::filesystem::exists(marker.output() / "receipt.json"), "failed marker removal cannot leave a complete final receipt");
}
void exact_greedy_stop_boundaries() {
    Private eos; auto copy = std::make_shared<CopyState>(); auto transfer = reader(copy); RawBaselineRecording recording(eos.output(), metadata(true), policy(), layout()); recording.capture_initial(session(), transfer);
    const std::array logits{1.0F, 2.0F, 3.0F}; recording.record_token(1, 0, logits, session(1), transfer);
    expect(rejects([&] { recording.record_token(2, 1, logits, session(2), transfer); }) && !std::filesystem::exists(eos.output() / "receipt.json"), "greedy cannot feed a token after first emitted EOS");
    Private capacity; auto m = metadata(true); m.maximum_generated_tokens = 1; auto bounds = policy(); bounds.maximum_steps = 1; RawBaselineRecording capped(capacity.output(), m, bounds, layout(1)); capped.capture_initial(session(), transfer);
    const std::array non_eos{3.0F, 2.0F, 1.0F}; capped.record_token(1, 0, non_eos, session(1), transfer); const std::array<std::uint32_t, 1> output{0};
    expect(rejects([&] { capped.finalize(session(1), transfer, output, "max_new_tokens"); }), "context-capacity stop wins over coincident max-new-tokens");
    Private correct; RawBaselineRecording good(correct.output(), m, bounds, layout(1)); good.capture_initial(session(), transfer); good.record_token(1, 0, non_eos, session(1), transfer); good.finalize(session(1), transfer, output, "context_capacity");
    expect(std::filesystem::exists(correct.output() / "receipt.json"), "exact context-capacity boundary can finalize without inventing extra execution");
    Private first_eos; RawBaselineRecording eos_good(first_eos.output(), m, bounds, layout(1)); eos_good.capture_initial(session(), transfer); eos_good.record_token(1, 0, logits, session(1), transfer);
    const std::array<std::uint32_t, 1> eos_output{2}; eos_good.finalize(session(1), transfer, eos_output, "eos");
    expect(std::filesystem::exists(first_eos.output() / "receipt.json"), "first emitted EOS wins over context and maximum token boundaries");
}
}
int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--write-example") { write_teacher(std::filesystem::path(argv[2])); return failures ? 1 : 0; }
        complete_raw_and_greedy(); rejects_before_creation(); failed_tokens_never_finalize(); uncertain_copy_and_publication(); exact_greedy_stop_boundaries();
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    std::cout << "RAW recording injected stream/state/token/publication tests; no GPU execution\n";
    return failures ? 1 : 0;
}
