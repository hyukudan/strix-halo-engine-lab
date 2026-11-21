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

namespace strix::spec {
class VerifyTarget;
}

namespace strix::serve {

class BpeTokenizer;

enum class CliSpeculation { off, automatic, verify };

// Bind one positional-checkpoint CLI family. The decoder consumes all prompt
// IDs at prefill, then one emitted ID per decode. Each call returns checked
// first-max and also fills the full output row when the span is nonempty.
class SerialGenerationDecoder {
public:
    virtual ~SerialGenerationDecoder() = default;
    virtual std::uint32_t prefill(std::span<const std::uint32_t>, std::span<float>) = 0;
    virtual std::uint32_t decode(std::uint32_t, std::span<float>) = 0;
    virtual spec::VerifyTarget* verify_target() { return nullptr; }
    // One cost per consecutive verify width, including the plain row. An
    // empty table declines speculation for the request's target path.
    virtual std::span<const double> verify_costs(CliSpeculation) const { return {}; }
};

struct SerialGenerationLoad {
    const std::string& checkpoint;
    std::uint32_t positions;
    std::uint32_t prefill_chunk;
    bool packed_decode;
};

struct SerialGenerationFamily {
    std::string_view executable, diagnostic_name, checkpoint_argument;
    std::uint32_t input_vocabulary, output_vocabulary, max_context, eos;
    std::unique_ptr<SerialGenerationDecoder> (*load)(const SerialGenerationLoad&);
    // A family opts into the native chunk control with a positive default and
    // into raw text by binding its tokenizer asset validator.
    std::uint32_t default_prefill_chunk = 0;
    bool supports_packed_decode = false;
    bool report_load_timing = false;
    std::shared_ptr<const BpeTokenizer> (*load_tokenizer)(const std::string&) = nullptr;
    bool packed_decode_default = false;
    std::string (*render_messages)(std::string_view) = nullptr;
    bool supports_speculation = false;
};

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
