#include "strix/device_allocation_error.hpp"
#include "strix/serial_request.hpp"
#include "strix/serve/api.hpp"
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/decode_form.hpp"
#include "strix/serve/prefill_chunk_fallback.hpp"
#include "strix/serve/serial_generation_cli.hpp"
#include "strix/session/embed_rows.hpp"
#include "strix/session/prompt_windows.hpp"
#include "strix/spec/block_drafter.hpp"
#include "strix/token_sampling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <new>
#include <optional>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

// The native runtime header and the CUDA weight uploader are the two production call sites of the chunk retry: they
// compile here against the host runtime stub (tests/stubs/cuda_status), which injects a refused allocation at the
// exact runtime call. Keep these last, after the headers the runtime header itself includes: the macro only selects the native section of the runtime header.
#include "strix/device/declaration.hpp"
#include "strix/session/step_replay.hpp"
#include "core/weight_load_queue.hpp"
#define __CUDACC__ 1
#include "cuda/core/weight_uploader.hpp"
#include "strix/device/runtime.hpp"

namespace {

using namespace strix::serve;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr std::array scores{0.0F, 2.0F, -100.0F, 1.0F};
unsigned loads = 0;
std::vector<std::uint32_t> inputs;
unsigned tunes = 0, seeds = 0;
CliSpeculation loaded_mode = CliSpeculation::off;
std::string cost_device;  // non-empty: the fixture persists its verify walls (virtual clock, 1 ms per reading)
double virtual_ms = 0.0;

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

class Decoder final : public SerialGenerationDecoder, public strix::spec::VerifyTarget {
public:
    explicit Decoder(unsigned rows) : rows_(rows) {}

    std::uint32_t prefill(std::span<const std::uint32_t> ids, std::span<float> logits) override {
        inputs.assign(ids.begin(), ids.end());
        if (!logits.empty()) std::ranges::copy(scores, logits.begin());
        next_ = ids.size() >= 4 ? 1U : 2U;
        return 1;
    }
    std::uint32_t decode(std::uint32_t id, std::span<float> logits) override {
        inputs.push_back(id);
        // The next row picks EOS under both greedy and sampling-v1.
        if (!logits.empty()) std::ranges::copy(std::array{-100.0F, -100.0F, 2.0F, -100.0F}, logits.begin());
        return next_;
    }

    std::uint32_t max_verify_rows(CliSpeculation) const override { return rows_; }
    strix::spec::VerifyTarget* verify_target() override { return this; }
    std::string verify_cost_device() const override { return cost_device; }
    void tune_speculation(strix::spec::BlockConfig& config) const override {
        ++tunes;
        config.arbiter.windows = {1, 2, 4, 8};  // engine windows: the width check reads the widest, not the count
        if (!cost_device.empty()) config.clock_ms = [] { return virtual_ms += 1.0; };
    }
    void seed_speculation(strix::spec::BlockDrafter& drafter) const override {
        ++seeds;
        for (const std::uint32_t row : {1U, 2U, 4U, 8U}) drafter.seed_verify_ms(row, 10.0 + double(row));
    }
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        check(tokens.size() <= rows_, "verifier width bounded by admission");
        staged_.assign(tokens.begin(), tokens.end());
        greedy.assign(tokens.size(), next_);
    }
    void commit(std::uint32_t keep) override {
        inputs.insert(inputs.end(), staged_.begin(), staged_.begin() + keep);
        staged_.clear();
    }

private:
    unsigned rows_, next_{};
    std::vector<std::uint32_t> staged_;
};

const SerialGenerationFamily family{"fixture", "Fixture", "CHECKPOINT", 9, 4, 64, 2,
    [](const SerialGenerationLoad& load) -> std::unique_ptr<SerialGenerationDecoder> {
        ++loads;
        loaded_mode = load.speculation;
        check(load.checkpoint == "checkpoint" && load.prefill_chunk == 0 && !load.packed_decode, "family binding");
        check(load.positions >= 2 && load.positions <= 64, "context binding");
        return std::make_unique<Decoder>(std::min(8U, load.verify_rows));
    }, 0U, false, false, nullptr, false, nullptr, true};

// Image input of --messages: a byte-level tokenizer with the control "<img>" (id 256), a renderer that writes
// "ab" + one "<img>" per normalized image part + "c", an adapter whose image is two rows of width 4, and a decoder
// that records the provided rows it was given.
std::string media_utf8(std::uint32_t cp) {
    std::string out;
    if (cp < 0x80U) out += static_cast<char>(cp);
    else { out += static_cast<char>(0xC0U | (cp >> 6)); out += static_cast<char>(0x80U | (cp & 0x3FU)); }
    return out;
}
std::shared_ptr<const BpeTokenizer> media_tokenizer(const std::string&) {
    std::vector<std::string> tokens;
    std::uint32_t extra = 0;
    for (std::uint32_t b = 0; b < 256U; ++b) {
        const bool direct = (b >= 0x21U && b <= 0x7EU) || (b >= 0xA1U && b <= 0xACU) || (b >= 0xAEU);
        tokens.push_back(media_utf8(direct ? b : 256U + extra++));
    }
    std::vector<std::int32_t> types(256, 1);
    tokens.emplace_back("<img>");
    types.push_back(3);
    return BpeTokenizer::from_parts(std::move(tokens), std::move(types), {}, "qwen35");
}
std::string media_render(std::string_view body) {
    std::string out = "ab";
    for (auto at = body.find("\"media\""); at != std::string_view::npos; at = body.find("\"media\"", at + 1)) out += "<img>";
    return out + "c";
}
std::string text_render(std::string_view) { return "ab"; }

std::vector<std::uint32_t> embedded_inputs;
std::shared_ptr<const strix::session::EmbedTable> embedded_table;
unsigned media_opens = 0;

class MediaDecoder final : public SerialGenerationDecoder {
public:
    std::uint32_t prefill(std::span<const std::uint32_t> ids, std::span<float>) override {
        embedded_inputs.assign(ids.begin(), ids.end());
        embedded_table.reset();
        return 1;
    }
    std::uint32_t prefill_embedded(std::span<const std::uint32_t> ids,
                                   const std::shared_ptr<const strix::session::EmbedTable>& table,
                                   std::span<float>) override {
        embedded_inputs.assign(ids.begin(), ids.end());
        embedded_table = table;
        return 1;
    }
    std::uint32_t decode(std::uint32_t, std::span<float>) override { return 2; }
};

class FakeImages final : public ApiMediaAdapter {
public:
    std::string identity() const override { return "fake-tower"; }
    std::string placeholder() const override { return "<img>"; }
    std::uint32_t max_rows() const override { return 2; }
    ApiMediaSpan encode(const media::MediaItem& item, std::uint32_t budget) override {
        check(budget >= 2 && item.width == 1 && item.height == 1, "image item and row budget");
        strix::session::EmbedRows rows;
        rows.digest[0] = 7;
        rows.rows = 2;
        rows.width = 4;
        rows.real_token_id = 256;
        rows.data.assign(8, 0x3F80);
        ApiMediaSpan span;
        span.lease = interner_.intern(std::move(rows));
        span.layout = span.lease.ids();
        return span;
    }
private:
    strix::session::EmbedInterner interner_{4};
};
class FakeMediaInput final : public SerialMediaInput {
public:
    ApiMediaAdapter& adapter() override { return images_; }
private:
    FakeImages images_;
};

SerialGenerationFamily media_family() {
    SerialGenerationFamily f{"fixture", "Fixture", "CHECKPOINT", 300, 4, 64, 2,
        [](const SerialGenerationLoad&) -> std::unique_ptr<SerialGenerationDecoder> {
            return std::make_unique<MediaDecoder>();
        }, 0U, false, false, media_tokenizer, false, text_render, true};
    f.media = [](const std::string&) -> std::unique_ptr<SerialMediaInput> {
        ++media_opens;
        return std::make_unique<FakeMediaInput>();
    };
    f.render_media_messages = media_render;
    return f;
}

struct Outcome { int code; std::string out, error; };

Outcome run(std::initializer_list<std::string> args, const SerialGenerationFamily& with = family) {
    std::vector<std::string> owned{"fixture", "checkpoint"};
    owned.insert(owned.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (auto& arg : owned) argv.push_back(arg.data());
    std::ostringstream out, error;
    const int code = run_serial_generation_cli(static_cast<int>(argv.size()), argv.data(), with, out, error);
    return {code, out.str(), error.str()};
}

}  // namespace

void media_messages(const std::function<std::pair<int, std::string>(const std::string&)>& run_messages) {
    const auto dir = std::filesystem::temp_directory_path();
    const auto write = [&](const char* name, const std::string& text) {
        const auto path = (dir / name).string();
        std::ofstream(path, std::ios::binary) << text;
        return path;
    };
    // 1x1 PNG as a data URL in an OpenAI image_url part.
    const std::string png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
    const auto image = write("strix-serial-media.json",
        R"({"messages":[{"role":"user","content":[{"type":"text","text":"q"},{"type":"image_url","image_url":{"url":")" +
        png + R"("}}]}]})");
    const auto [code, out] = run_messages(image);
    check(code == 0 && out.starts_with("next_token 1\n"), "image prompt runs");
    check(media_opens == 1 && embedded_table != nullptr, "image prompt opens the adapter and passes a table");
    check(embedded_inputs.size() == 5 && embedded_inputs[0] == 'a' && embedded_inputs[1] == 'b' &&
          strix::session::is_embed_id(embedded_inputs[2]) && strix::session::is_embed_id(embedded_inputs[3]) &&
          strix::session::embed_row(embedded_inputs[3]) == 1 && embedded_inputs[4] == 'c',
          "placeholder expands to one virtual id per row");
    check(embedded_table && embedded_table->find(strix::session::embed_handle(embedded_inputs[2])) != nullptr,
          "the table resolves the virtual ids");
    // A body without images keeps the plain renderer and opens nothing.
    const auto text = write("strix-serial-text.json", R"([{"role":"user","content":"q"}])");
    const auto [text_code, text_out] = run_messages(text);
    check(text_code == 0 && media_opens == 1 && embedded_table == nullptr &&
          embedded_inputs == std::vector<std::uint32_t>{'a', 'b'}, "text body is unchanged");
    std::filesystem::remove(image);
    std::filesystem::remove(text);
}

// Implicit prefill chunk: one retry at the fallback rows, only after an allocation failure.
static void chunk_fallback_cases() {
    namespace sv = strix::serve;
    static_assert(sv::resolve_prefill_rows(std::nullopt, 8192U, 0U) == 8192U);
    static_assert(sv::resolve_prefill_rows(4096U, 8192U, 0U) == 4096U);      // explicit wins
    static_assert(sv::resolve_prefill_rows(std::nullopt, 8192U, 300U) == 300U);  // the bound caps
    static_assert(sv::resolve_prefill_rows(512U, 8192U, 300U) == 300U);
    static_assert(sv::admit_prefill_chunk(std::nullopt, 8192U, 2048U, 5000U).chunk == 5000U);
    static_assert(sv::admit_prefill_chunk(std::nullopt, 8192U, 2048U, 5000U).fallback == 2048U);
    static_assert(sv::admit_prefill_chunk(std::nullopt, 8192U, 2048U, 1000U).fallback == 1000U);
    static_assert(sv::admit_prefill_chunk(4096U, 8192U, 2048U, 9000U).fallback == 0U);  // explicit: no retry
    static_assert(sv::admit_prefill_chunk(4096U, 8192U, 2048U, 9000U).explicit_rows);
    static_assert(sv::admit_prefill_chunk(std::nullopt, 8192U, 0U, 9000U).fallback == 0U);  // unqualified backend
    std::ostringstream log;
    std::vector<unsigned> tried;
    // The device holds at most `capacity` rows: larger loads throw `what`.
    const auto loader = [&](unsigned capacity, const char* what = "reserve: out of memory", bool typed = true) {
        return [&tried, capacity, what, typed](std::uint32_t rows) {
            tried.push_back(rows);
            if (rows > capacity) {
                if (typed) throw strix::DeviceAllocationError(what, 2);
                throw std::runtime_error(what);
            }
            return rows;
        };
    };
    const auto throws = [&](bool explicit_chunk, std::uint32_t chunk, std::uint32_t fallback, auto load) {
        try {
            (void)sv::load_with_chunk_fallback(chunk, explicit_chunk, fallback, log, load);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    check(sv::load_with_chunk_fallback(8192U, false, 2048U, log, loader(8192U)) == 8192U && tried.size() == 1U &&
          log.str().empty(), "chunk that fits is kept without a line");
    tried.clear();
    check(sv::load_with_chunk_fallback(8192U, false, 2048U, log, loader(4096U)) == 2048U &&
          tried == std::vector<unsigned>{8192U, 2048U} &&
          log.str() == "strix-prefill-chunk 8192 -> 2048 (device memory)\n", "implicit chunk falls back once");
    tried.clear();
    log.str({});
    check(throws(true, 8192U, 2048U, loader(4096U)) && tried.size() == 1U && log.str().empty(),
          "explicit chunk is never changed");
    tried.clear();
    check(throws(false, 8192U, 2048U, loader(4096U, "open checkpoint: no such file", false)) && tried.size() == 1U &&
          log.str().empty(), "a non-memory failure is not retried");
    tried.clear();
    check(throws(false, 8192U, 2048U, loader(4096U, "invalid checkpoint: out of memory in metadata key", false)) &&
          tried.size() == 1U && log.str().empty(), "an untyped failure whose text says out of memory is not retried");
    tried.clear();
    check(throws(false, 8192U, 2048U, loader(4096U, "invalid checkpoint: out of memory in metadata key", false)) &&
          tried.size() == 1U, "message text alone never selects the retry");
    tried.clear();
    check(throws(false, 8192U, 2048U, loader(1024U)) && tried.size() == 2U, "failure of the retry propagates");
    tried.clear();
    check(throws(false, 1000U, 2048U, loader(512U)) && tried.size() == 1U, "chunk at or below the fallback");
    tried.clear();
    check(throws(false, 8192U, 0U, loader(4096U)) && tried.size() == 1U, "no fallback configured");
    // The qualification gate loads through the shared admission entry: bound = prefix positions.
    tried.clear();
    log.str({});
    check(sv::load_with_prefill_admission(std::nullopt, 8192U, 2048U, 5000U, log, loader(4096U)) == 2048U &&
          tried == std::vector<unsigned>{5000U, 2048U} && !log.str().empty(), "gate: implicit chunk retries once");
    tried.clear();
    check(throws(true, 5000U, 2048U, loader(4096U)) && sv::load_with_prefill_admission(4096U, 8192U, 2048U, 9000U, log,
          loader(4096U)) == 4096U, "gate: explicit chunk is used as given");
    tried.clear();
    bool host_failure_escaped = false;
    try {
        (void)sv::load_with_chunk_fallback(8192U, false, 2048U, log, [&](std::uint32_t rows) -> std::uint32_t {
            tried.push_back(rows);
            throw std::bad_alloc();
        });
    } catch (const std::bad_alloc&) {
        host_failure_escaped = true;
    }
    check(host_failure_escaped && tried.size() == 1U, "a host bad_alloc is not retried at fewer device rows");
}

// The shared CLI entry point: the implicit chunk (family default 6 rows, fallback 2) retries once after a typed
// device allocation failure; an explicit --prefill-chunk, or a failure that only says "out of memory", never does.
static void chunk_fallback_cli_cases() {
    std::vector<std::uint32_t> loaded_rows;
    bool typed = true;
    SerialGenerationFamily pressure = family;
    pressure.default_prefill_chunk = 6U;
    pressure.prefill_chunk_fallback = 2U;
    pressure.load = [&](const SerialGenerationLoad& load) -> std::unique_ptr<SerialGenerationDecoder> {
        loaded_rows.push_back(load.prefill_chunk);
        if (load.prefill_chunk > 3U) {
            if (typed) throw strix::DeviceAllocationError("reserve: out of memory", 2);
            throw std::runtime_error("invalid checkpoint: out of memory in metadata key");
        }
        return std::make_unique<Decoder>(std::min(8U, load.verify_rows));
    };
    const auto implicit = run({"8", "4", "1", "2", "3", "5"}, pressure);
    check(implicit.code == 0 && loaded_rows == std::vector<std::uint32_t>{6U, 2U}, "implicit chunk retries once");
    loaded_rows.clear();
    const auto fits = run({"8", "4", "1", "2", "3", "5", "--prefill-chunk", "3"}, pressure);
    check(fits.code == 0 && loaded_rows == std::vector<std::uint32_t>{3U}, "explicit chunk is used as given");
    loaded_rows.clear();
    const auto refused = run({"8", "4", "1", "2", "3", "5", "--prefill-chunk", "6"}, pressure);
    check(refused.code != 0 && loaded_rows == std::vector<std::uint32_t>{6U}, "explicit chunk is never retried");
    loaded_rows.clear();
    // A backend without a qualified fallback (family data zero): the implicit default has one attempt only.
    pressure.prefill_chunk_fallback = 0U;
    const auto unqualified = run({"8", "4", "1", "2", "3", "5"}, pressure);
    check(unqualified.code != 0 && loaded_rows == std::vector<std::uint32_t>{6U}, "no fallback datum: no retry route");
    pressure.prefill_chunk_fallback = 2U;
    loaded_rows.clear();
    // The prompt bounds the chunk (a 3-row prompt loads 3 rows) and the fallback.
    const auto bounded = run({"8", "4", "1"}, pressure);
    check(bounded.code == 0 && loaded_rows == std::vector<std::uint32_t>{3U}, "prompt bounds the implicit chunk");
    loaded_rows.clear();
    typed = false;
    const auto untyped = run({"8", "4", "1", "2", "3", "5"}, pressure);
    check(untyped.code != 0 && loaded_rows == std::vector<std::uint32_t>{6U}, "untyped failure is not retried");

    // The same out-of-memory runtime status, injected into the runtime stub at the two ACTUAL allocation call sites.
    // A pinned host staging allocation (the CUDA weight uploader) fails as the generic std::runtime_error: never
    // retried, one load. A device row buffer (device::native::Buffer, the device-allocation boundary) fails as the
    // typed DeviceAllocationError: retried once at the fallback rows, two loads.
    namespace stub = cuda_status_stub;
    const auto refused_as = [](auto&& allocate, bool typed_error) {
        try {
            allocate();
        } catch (const strix::DeviceAllocationError& error) {
            return typed_error && error.status == static_cast<int>(cudaErrorMemoryAllocation);
        } catch (const std::runtime_error& error) {
            return !typed_error && std::string(error.what()).find("out of memory") != std::string::npos;
        }
        return false;
    };
    const auto pinned_staging = [] {
        strix::cuda::core::WeightUploader uploader;
        (void)uploader.allocate(4096U);
    };
    const auto device_rows = [] { strix::device::native::Buffer<std::uint8_t> rows(4096U); };
    stub::malloc_host_status = cudaErrorMemoryAllocation;
    check(refused_as(pinned_staging, false) && stub::malloc_host_calls == 1U, "pinned staging refusal is a plain runtime error");
    stub::malloc_host_status = cudaSuccess;
    stub::malloc_status = cudaErrorMemoryAllocation;
    check(refused_as(device_rows, true) && stub::malloc_calls == 1U, "device row refusal is a DeviceAllocationError");
    stub::malloc_status = cudaSuccess;

    std::function<void()> site = pinned_staging;
    pressure.load = [&](const SerialGenerationLoad& load) -> std::unique_ptr<SerialGenerationDecoder> {
        loaded_rows.push_back(load.prefill_chunk);
        if (load.prefill_chunk > 3U) site();
        return std::make_unique<Decoder>(std::min(8U, load.verify_rows));
    };
    stub::malloc_host_status = cudaErrorMemoryAllocation;
    loaded_rows.clear();
    const auto staging = run({"8", "4", "1", "2", "3", "5"}, pressure);
    check(staging.code != 0 && loaded_rows == std::vector<std::uint32_t>{6U}, "pinned staging refusal loads once");
    stub::malloc_host_status = cudaSuccess;
    stub::malloc_status = cudaErrorMemoryAllocation;
    site = device_rows;
    loaded_rows.clear();
    const auto rows = run({"8", "4", "1", "2", "3", "5"}, pressure);
    check(rows.code == 0 && loaded_rows == std::vector<std::uint32_t>{6U, 2U}, "device row refusal loads twice");
    stub::malloc_status = cudaSuccess;
}

// The one decode-form resolver (strix/serve/decode_form.hpp) against the expression it replaced in the command line:
// the same outcome for every flag pair and either family default, and the same refusal of both flags.
static void decode_form_cases() {
    for (const bool family_default : {false, true})
        for (const bool packed : {false, true})
            for (const bool grouped : {false, true}) {
                if (packed && grouped) {
                    bool refused = false;
                    try {
                        (void)resolve_decode_form(packed, grouped, family_default);
                    } catch (const std::invalid_argument&) {
                        refused = true;
                    }
                    check(refused, "both decode-form flags are refused");
                    continue;
                }
                const bool before = packed || (family_default && !grouped);
                check(resolve_decode_form(packed, grouped, family_default) == before, "decode form equals the old rule");
            }
    static_assert(resolve_decode_form(false, false, true) && !resolve_decode_form(false, true, true));
    static_assert(resolve_decode_form(true, false, false) && !resolve_decode_form(false, false, false));
}

int main() try {
    decode_form_cases();
    chunk_fallback_cases();
    chunk_fallback_cli_cases();
    {
        const auto family_with_media = media_family();
        media_messages([&](const std::string& path) {
            const auto outcome = run({"--messages", path}, family_with_media);
            if (outcome.code != 0) std::cerr << outcome.error;
            return std::pair{outcome.code, outcome.out};
        });
    }
    const auto greedy = run({"8", "4", "--generate", "4"});
    check(greedy.code == 0 && greedy.out.starts_with("next_token 1\ngenerated 1 2\n"), "greedy IDs and EOS");
    check(greedy.out.find(" decode_tokens 1 ") != std::string::npos, "count decode calls only");
    check(inputs == std::vector<std::uint32_t>{8, 4, 1}, "consume the selected ID once, leaving EOS pending");

    const auto probe = run({"8", "4"});
    check(probe.code == 0 && probe.out.starts_with("next_token 1\ntiming ") && inputs.size() == 2,
          "zero generation still selects the prompt head");
    const auto one = run({"8", "4", "--generate", "1"});
    check(one.code == 0 && one.out.starts_with("next_token 1\ngenerated 1\n") && inputs.size() == 2,
          "leave the last budgeted token pending");

    std::uint64_t seed = 0;
    for (; seed < 100; ++seed) {
        strix::TokenSampler sampler({1.0F, 2, 1.0F, 0.0F, seed}, strix::TokenSamplerOptions{});
        if (sampler.decide(scores, 2, nullptr).token == 3) break;
    }
    check(seed < 100, "fixture's shared sampler must select a non-greedy ID");
    const auto sampled = run({"8", "4", "--generate", "4", "--top-k", "2", "--seed", std::to_string(seed)});
    check(sampled.code == 0 && sampled.out.starts_with("next_token 3\ngenerated 3 2\n") && inputs.back() == 3,
          "top-k uses shared sampling-v1 and feeds its selected token back");
    const auto disabled = run({"8", "4", "--top-k", "2", "--temperature", "0", "--seed", std::to_string(seed)});
    check(disabled.code == 0 && disabled.out.starts_with("next_token 1\n"), "explicit zero temperature stays greedy");

    for (const auto flags : {std::array{"--top-p", "0.8"}, std::array{"--min-p", "0.3"}}) {
        const bool top_p = std::string_view(flags[0]) == "--top-p";
        const auto policy = strix::TokenSamplingPolicy{1.0F, 0, top_p ? 0.8F : 1.0F, top_p ? 0.0F : 0.3F, 0};
        std::uint64_t draw_seed = 0;
        for (; draw_seed < 100; ++draw_seed) {
            auto draw_policy = policy;
            draw_policy.seed = draw_seed;
            strix::TokenSampler sampler(draw_policy, strix::TokenSamplerOptions{});
            strix::TokenSampler unfiltered({1.0F, 0, 1.0F, 0.0F, draw_seed}, strix::TokenSamplerOptions{});
            if (sampler.decide(scores, 2, nullptr).token == 3 &&
                unfiltered.decide(scores, 2, nullptr).token != 3) break;
        }
        check(draw_seed < 100, "filtered fixture selects a non-greedy ID");
        const auto filtered = run({"8", "4", "--generate", "4", flags[0], flags[1], "--seed", std::to_string(draw_seed)});
        check(filtered.code == 0 && filtered.out.starts_with("next_token 3\ngenerated 3 2\n") && inputs.back() == 3,
              "top-p and min-p independently enable the shared sampler");
        const auto greedy_filtered = run({"8", "4", flags[0], flags[1], "--temperature", "0"});
        check(greedy_filtered.code == 0 && greedy_filtered.out.starts_with("next_token 1\n"),
              "explicit zero temperature overrides either filter");
    }

    const auto repeated = [](std::string mode) {
        return run({"1", "1", "1", "1", "1", "1", "1", "1", "--generate", "16",
                    "--temperature", "0", "--speculation", mode});
    };
    const auto plain = repeated("off");
    const auto plain_inputs = inputs;
    const auto lookup = repeated("lookup");
    const auto generated_end = plain.out.find("timing ");
    check(plain.code == 0 && lookup.code == 0 && generated_end != std::string::npos &&
          lookup.out.starts_with(plain.out.substr(0, generated_end)) && inputs == plain_inputs,
          "forced lookup matches complete greedy IDs and the committed prefix");
    const auto receipt = lookup.out.find(" acceptance ");
    check(receipt != std::string::npos, "lookup receipt present");
    std::istringstream metrics(lookup.out.substr(receipt));
    std::uint64_t accepted{}, drafted{}, passes{};
    std::string label;
    char slash{};
    double mean{};
    metrics >> label >> accepted >> slash >> drafted >> label >> passes >> label >> mean;
    check(metrics && slash == '/' && drafted > 0 && accepted == drafted && passes > 0 &&
          std::abs(mean - static_cast<double>(accepted) / static_cast<double>(passes)) < 1e-5,
          "report actual accepted and drafted counts and mean per proposal");
    tunes = seeds = 0;
    const auto automatic = repeated("auto");
    check(tunes == 1 && seeds == 1 && loaded_mode == CliSpeculation::automatic,
          "auto: the decoder sees the mode at load and tunes and seeds the shared speculator once");
    check(automatic.code == 0 && automatic.out.starts_with(plain.out.substr(0, generated_end)) &&
          inputs == plain_inputs && automatic.out.find("strix-spec-stats ") != std::string::npos,
          "greedy auto runs the shared verifier and matches serial IDs and commit");
    tunes = seeds = 0;
    const auto diagnostic = repeated("verify");
    check(diagnostic.code == 0 && diagnostic.out.starts_with(plain.out.substr(0, generated_end)) && tunes == 0 &&
          seeds == 0 && loaded_mode == CliSpeculation::verify,
          "the verify diagnostic keeps its equal prices (no engine tuning or seeds)");
    (void)repeated("off");
    check(loaded_mode == CliSpeculation::off, "plain decode loads without a speculation mode");
    auto auto_family = family;
    auto_family.default_speculation = CliSpeculation::automatic;
    const auto by_default = run({"1", "1", "1", "1", "1", "1", "1", "1", "--generate", "16"}, auto_family);
    check(by_default.code == 0 && by_default.out.find("speculation mode auto ") != std::string::npos &&
          by_default.out.starts_with(plain.out.substr(0, generated_end)) && inputs == plain_inputs,
          "the family default speculation applies when --speculation is absent");
    const auto forced_off = run({"1", "1", "1", "1", "1", "1", "1", "1", "--generate", "16", "--speculation", "off"},
                                auto_family);
    check(forced_off.code == 0 && forced_off.out.find("speculation mode") == std::string::npos,
          "explicit --speculation off overrides the family default");
    auto sampled_family = auto_family;
    sampled_family.sampled_speculation = true;
    const auto sampled_fallback = run({"8", "4", "--generate", "4", "--temperature", "1"}, sampled_family);
    check(sampled_fallback.code == 0 && sampled_fallback.out.find("strix-spec-stats ") == std::string::npos,
          "sampled auto falls back to serial when the target cannot decide sampled rows");
    check(run({"8", "4", "--generate", "4", "--speculation", "verify", "--temperature", "1"}, sampled_family)
              .error.find("cannot verify sampled rows") != std::string::npos,
          "sampled forced verify refuses a target without sampled rows");
    const auto miss = run({"8", "4", "--generate", "4", "--speculation", "lookup"});
    check(miss.code == 0 && miss.out.starts_with("next_token 1\ngenerated 1 2\n") &&
          miss.out.find(" acceptance 0/0 draft_passes 0 mean_accepted_length 0") != std::string::npos,
          "lookup with no match preserves the EOS continuation");
    const auto sampled_auto = run({"8", "4", "--generate", "4", "--speculation", "auto", "--temperature", "1"});
    check(sampled_auto.code == 0 && sampled_auto.out.find("strix-spec-stats ") == std::string::npos,
          "sampled auto stays on the plain route");

    const auto previous = loads;
    check(run({"9", "--generate", "1"}).code == 1, "input vocabulary bounds");
    check(run({"1", "--generate", "64"}).code == 1, "context bounds");
    check(run({"1", "--prefill-chunk", "1"}).code == 1, "reject unsupported native flag");
    check(run({"1", "--temperature", "nan"}).code == 1, "reject nonfinite sampling policy");
    check(run({"1", "--top-p", "0"}).code == 1 && run({"1", "--min-p", "1"}).code == 1 &&
          run({"1", "--top-p", "0.8junk"}).code == 1 && run({"1", "--min-p"}).code == 1,
          "reject invalid or missing filter values");
    check(run({"1", "--generate", "4", "--speculation", "lookup", "--temperature", "1"}).error.find(
              "--speculation lookup requires greedy decoding") != std::string::npos,
          "reject sampled lookup before load");
    check(loads == previous, "reject invalid requests before load");

    unsigned decodes = 0;
    const auto full = strix::execute_serial_token_continuation({1, 8, 4, 8, 10, 2},
        [&](std::uint32_t, std::uint64_t) { ++decodes; return 2U; });
    check(full.stop_reason == strix::SerialRequestStopReason::context_capacity && full.final_position == 8 &&
          full.generated_tokens == std::vector<std::uint32_t>{1} && decodes == 0,
          "context boundary publishes the pending token without target work");

    constexpr std::array prompt{1U, 2U, 3U, 4U, 5U};
    std::vector<unsigned> ordered;
    std::vector<std::size_t> sizes;
    strix::session::prompt_windows::append_chunks(std::span<const unsigned>(prompt), 2,
        [&](std::span<const unsigned> chunk, bool last) {
            sizes.push_back(chunk.size());
            ordered.insert(ordered.end(), chunk.begin(), chunk.end());
            check(last == (ordered.size() == prompt.size()), "head on final chunk only");
        });
    check(sizes == std::vector<std::size_t>{2, 2, 1} && std::ranges::equal(ordered, prompt), "ordered chunk tail");
    // Tail absorption (opt-in by divisor): the remainder after full chunks when it is at most chunk / divisor.
    static_assert(strix::serve::prefill_tail_rows(8236U, 8192U, 8U) == 44U);
    static_assert(strix::serve::prefill_tail_rows(8236U, 8192U, 0U) == 0U);
    static_assert(strix::serve::prefill_tail_rows(8192U, 8192U, 8U) == 0U);
    static_assert(strix::serve::prefill_tail_rows(4309U, 4309U, 8U) == 0U);
    static_assert(strix::serve::prefill_tail_rows(68000U, 8192U, 8U) == 0U);   // 2,464-row remainder > 1,024
    static_assert(strix::serve::prefill_tail_rows(33792U, 8192U, 8U) == 1024U);
    static_assert(strix::serve::prefill_tail_rows(33793U, 8192U, 8U) == 0U);
    static_assert(strix::serve::prefill_tail_rows(16384U, 8192U, 8U) == 0U);

    // Persisted verify walls: a device identity opts in; the first run records, the next seeds; ids never change.
    {
        namespace fs = std::filesystem;
        std::random_device rd;
        const auto dir = fs::temp_directory_path() / ("strix-serial-verify-costs-" + std::to_string(rd()));
        set_env("STRIX_COST_TABLE", nullptr);
        set_env("STRIX_COST_TABLE_DIR", dir.string().c_str());
        const auto ids = [](const Outcome& o) { return o.out.substr(0, o.out.find("timing ")); };
        const auto line = [](const Outcome& o) {
            const auto at = o.out.find("speculation cost-table ");
            return at == std::string::npos ? std::string{} : o.out.substr(at, o.out.find('\n', at) - at);
        };
        const auto plain_ids = ids(repeated("off"));
        const auto unbound = repeated("auto");
        check(unbound.code == 0 && line(unbound).empty(), "no device identity: nothing persists");
        cost_device = "fixture-device";
        const auto first = repeated("auto");
        check(first.code == 0 && ids(first) == plain_ids, "first persisted run: plain ids");
        check(line(first).starts_with("speculation cost-table missing seeded 0 recorded ") &&
              line(first).find(" stored 1 ") != std::string::npos, "first run records and stores its walls");
        const auto second = repeated("auto");
        check(second.code == 0 && ids(second) == plain_ids, "seeded run: plain ids");
        check(line(second).starts_with("speculation cost-table loaded seeded ") &&
              !line(second).starts_with("speculation cost-table loaded seeded 0 "), "next run seeds the stored walls");
        const auto priced_equal = repeated("verify");
        check(priced_equal.code == 0 && line(priced_equal).empty(), "verify diagnostic keeps equal prices");
        set_env("STRIX_COST_TABLE", "0");
        const auto off = repeated("auto");
        check(off.code == 0 && line(off).empty() && ids(off) == plain_ids, "STRIX_COST_TABLE=0 switches it off");
        set_env("STRIX_COST_TABLE", nullptr);
        set_env("STRIX_COST_TABLE_DIR", nullptr);
        cost_device.clear();
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
