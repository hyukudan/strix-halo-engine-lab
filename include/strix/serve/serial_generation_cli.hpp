#pragma once

#include "strix/token_sampling.hpp"

#include <functional>
#include <vector>

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace strix::session {
class EmbedTable;
}

namespace strix::spec {
class BlockDrafter;
struct BlockConfig;
class ChainDrafter;
class VerifyTarget;
}

namespace strix::serve {

class BpeTokenizer;
class ApiMediaAdapter;

enum class CliSpeculation { off, automatic, verify, lookup };

// Bind one positional-checkpoint CLI family. The decoder consumes all prompt
// IDs at prefill, then one emitted ID per decode. Each call returns checked
// first-max and also fills the full output row when the span is nonempty.
class SerialGenerationDecoder {
public:
    virtual ~SerialGenerationDecoder() = default;
    virtual std::uint32_t prefill(std::span<const std::uint32_t>, std::span<float>) = 0;
    // Prefill of a prompt that holds virtual ids (session/embed_rows.hpp) resolved by `table`: the rows of those
    // ids come from the table instead of the embedding table. A family without provided rows refuses a table;
    // a null table is the plain prefill.
    virtual std::uint32_t prefill_embedded(std::span<const std::uint32_t> ids,
                                           const std::shared_ptr<const session::EmbedTable>& table,
                                           std::span<float> logits);
    virtual std::uint32_t decode(std::uint32_t, std::span<float>) = 0;
    virtual spec::VerifyTarget* verify_target() { return nullptr; }
    // Includes the anchor. Zero declines the requested mode.
    virtual std::uint32_t max_verify_rows(CliSpeculation) const { return 0; }
    // Optional, request-owned drafter. Called after prefill for greedy auto.
    virtual spec::ChainDrafter* chain_drafter(std::uint32_t, std::uint32_t) { return nullptr; }
    // Optional engine data for the shared speculator (auto and lookup): arbiter windows, the copy + chain hybrid,
    // chain cost seeds. Called before the request's mode settings (block width, forced lookup, diagnostic prices),
    // which win. Never changes emitted ids.
    virtual void tune_speculation(spec::BlockConfig&) const {}
    // Optional engine cost seeds (verify walls per row, chain survival), after construction. Skipped by the
    // `verify` diagnostic, which prices every row equally.
    virtual void seed_speculation(spec::BlockDrafter&) const {}
    // Device identity of the verify target (backend, device name, link): with one, auto and lookup persist their
    // measured verify walls per (executable, device, model, policy) through the shared cost table and seed the next
    // run from them (strix/spec/verify_cost_table.hpp; off: STRIX_COST_TABLE=0). Empty (default): nothing persists.
    [[nodiscard]] virtual std::string verify_cost_device() const { return {}; }
};

struct SerialGenerationLoad {
    const std::string& checkpoint;
    std::uint32_t positions;
    std::uint32_t prefill_chunk;
    bool packed_decode;
    // Maximum useful target width for a speculative request (greedy, or sampled
    // when the family declares sampled_speculation), or zero.
    // Backends cap this to their verifier's supported widths.
    std::uint32_t verify_rows = 0;
    // The request's speculation mode (off when verify_rows is zero): a decoder loads its drafter only when it may run.
    CliSpeculation speculation = CliSpeculation::off;
    // Rows the final prefill chunk takes beyond prefill_chunk (prompt_windows::append_chunks tail_rows); the decoder's
    // chunk capacity is prefill_chunk + prefill_tail. Zero unless the family opts in (prefill_tail_divisor).
    std::uint32_t prefill_tail = 0;
};

// Image input of --messages: owns the family's media adapter (e.g. vision::ApiMediaBridge over its VisionAdapter,
// encoder backend, weights and interner) for one run.
class SerialMediaInput {
public:
    virtual ~SerialMediaInput() = default;
    [[nodiscard]] virtual ApiMediaAdapter& adapter() = 0;
};

struct SerialGenerationFamily {
    std::string_view executable, diagnostic_name, checkpoint_argument;
    std::uint32_t input_vocabulary, output_vocabulary, max_context, eos;
    std::function<std::unique_ptr<SerialGenerationDecoder>(const SerialGenerationLoad&)> load;
    // A family opts into the native chunk control with a positive default and
    // into raw text by binding its tokenizer asset validator.
    std::uint32_t default_prefill_chunk = 0;
    bool supports_packed_decode = false;
    bool report_load_timing = false;
    std::shared_ptr<const BpeTokenizer> (*load_tokenizer)(const std::string&) = nullptr;
    bool packed_decode_default = false;
    std::string (*render_messages)(std::string_view) = nullptr;
    bool supports_speculation = false;
    // The family's verify target decides sampled rows (VerifyTarget::supports_row_rule(true)):
    // speculation then also runs with temperature, emitting the ids of serial sampled decode.
    bool sampled_speculation = false;
    // Speculation mode when --speculation is absent (family data). automatic is lossless (the ids of plain
    // decode, greedy or sampled) and falls back to serial steps when drafting does not pay.
    CliSpeculation default_speculation = CliSpeculation::off;
    // Optional image input of --messages (both set or neither). media opens the family's adapter for the
    // checkpoint; render_media_messages renders a body whose image parts the shared media layer has normalized
    // ({"type": "image", "media": k}), writing the adapter's placeholder once per part. Without them, a body
    // with images is rendered by render_messages, which refuses image parts.
    std::function<std::unique_ptr<SerialMediaInput>(const std::string& checkpoint)> media = {};
    std::string (*render_media_messages)(std::string_view) = nullptr;
    // Opt-in tail absorption for a family whose prefill bits do not depend on chunk boundaries: a prompt longer than
    // its chunk ends with one chunk that also takes a remainder of up to chunk / prefill_tail_divisor rows instead of
    // a short extra chunk (prefill_tail_rows). Zero = greedy chunks.
    std::uint32_t prefill_tail_divisor = 0;
};

// Tail rows for a prompt of prompt_size rows in chunks of `chunk` (already clamped to the prompt): the remainder after
// the full chunks when it is at most chunk / divisor, else 0. The decoder's chunk capacity grows only by an absorbed tail.
[[nodiscard]] constexpr std::uint32_t prefill_tail_rows(std::uint32_t prompt_size, std::uint32_t chunk,
                                                        std::uint32_t divisor) {
    if (divisor == 0U || chunk == 0U || prompt_size <= chunk) return 0U;
    const auto remainder = prompt_size % chunk;
    return remainder != 0U && remainder <= chunk / divisor ? remainder : 0U;
}

struct SerialGenerationOptions {
    std::vector<std::uint32_t> prompt;
    const char* text{};
    const char* messages{};
    const char* tokenizer{};
    const char* tokenize_file{};
    const char* logits{};
    std::uint32_t generate{}, prefill_chunk{};
    bool packed_decode{};
    bool generate_requested{}, sampling_requested{};
    TokenSamplingPolicy sampling;
    CliSpeculation speculation = CliSpeculation::off;
};

using SerialCliOption = std::function<bool(std::string_view, const std::function<const char*()>&)>;

SerialGenerationOptions parse_serial_generation_cli(int argc, char** argv, const SerialGenerationFamily&,
                                                    const SerialCliOption& extra = {});
void run_serial_generation(const std::string& checkpoint, SerialGenerationOptions,
                           const SerialGenerationFamily&, std::ostream&,
                           const std::function<void()>& on_loaded = {});

// Owns common flags, context admission, sampling, stream output and timing.
int run_serial_generation_cli(int argc, char** argv, const SerialGenerationFamily& family,
                              std::ostream& out, std::ostream& error);

}  // namespace strix::serve
