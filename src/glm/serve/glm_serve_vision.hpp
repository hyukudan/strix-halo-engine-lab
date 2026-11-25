#pragma once
// Image input of strix-glm5-serve, the host half (no device code):
//   VisionStack   rank 0: the projector file, the GLM adapter on the shared FP32 host tower, the embed cache and the
//                 shared ApiMediaBridge the API front doors call (one image part -> its interned rows);
//   PromptEmbeds  every rank: the table of the prompt in prefill, the pair embed frame in both directions and the
//                 resolution of a prompt chunk's virtual ids into vocabulary ids plus the engine's splice.
#include "glm/vision/glm_vision.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/pair/embed_frame.hpp"
#include "strix/serve/media.hpp"
#include "strix/session/embed_rows.hpp"
#include "strix/session/embed_splice.hpp"
#include "strix/vision/api_media.hpp"
#include "strix/vision/embed_cache.hpp"
#include "strix/vision/encoder.hpp"
#include "strix/vision/gguf_weights.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::glm::serve {

inline constexpr const char* kImageBeginText = "<|begin_of_image|>";
inline constexpr const char* kImageText = "<|image|>";
inline constexpr const char* kImageEndText = "<|end_of_image|>";

// Members are declared in dependency order: the bridge refers to everything above it.
struct VisionStack {
    vision::GgufWeights source;
    VisionConfig model;
    std::unique_ptr<VisionAdapter> adapter;
    vision::EncoderWeights weights;
    std::unique_ptr<vision::ReferenceEncoder> encoder;
    std::unique_ptr<vision::EmbedCache> cache;
    session::EmbedInterner interner;
    std::shared_ptr<vision::ApiMediaBridge> bridge;
    explicit VisionStack(vision::GgufWeights&& s) : source(std::move(s)) {}
};

// encode(text) -> ids with special tokens parsed (the server's tokenizer). Throws std::invalid_argument /
// std::runtime_error naming what the file or the tokenizer lacks.
[[nodiscard]] inline std::unique_ptr<VisionStack> make_vision_stack(
    const std::filesystem::path& file, const std::function<std::vector<std::uint32_t>(const std::string&)>& encode,
    const strix::serve::media::Limits& limits) {
    // Header first: schema (names, shapes, dtypes), the language-model match and the projector byte budget are judged
    // from the bounded header; only then are the file's bytes read (at exactly the length the header was judged at)
    // and hashed.
    const gguf::FileInfo header = gguf::read_file(file, vision_gguf_limits());
    const VisionConfig model = validate_vision(header);
    require_vision_matches_language(model, kLanguageHidden, kLanguageVocab, kImageToken);
    vision::HostBudget budget;  // max_bytes keeps the independent whole-file budget (4 GiB)
    budget.exact_bytes = check_projector_bytes(header);
    // The file opened for the load is judged by the same schema, language match and byte layout from its own header
    // before any of its bytes are allocated or read (a same-length replacement is refused without a full read).
    budget.admit = [exact = budget.exact_bytes](const gguf::FileInfo& opened) {
        require_vision_matches_language(validate_vision(opened), kLanguageHidden, kLanguageVocab, kImageToken);
        if (check_projector_bytes(opened) != exact) throw std::invalid_argument("GLM vision: the projector file changed while it was read");
    };
    auto st = std::make_unique<VisionStack>(vision::GgufWeights{file, vision_gguf_limits(), budget});
    // The bytes actually loaded (and hashed below) are judged again from their own tensor table: a same-length
    // replacement between the header read and the load must pass the same schema, language match and byte budget, and
    // the stack's configuration is the loaded file's.
    st->model = validate_vision(st->source.info());
    require_vision_matches_language(st->model, kLanguageHidden, kLanguageVocab, kImageToken);
    if (check_projector_bytes(st->source.info()) != budget.exact_bytes)
        throw std::invalid_argument("GLM vision: the projector file changed while it was read");
    const auto sha = strix::sha256_bytes_fast(std::span<const std::byte>(st->source.bytes()));
    // The template text of an image must be the three vocabulary tokens the adapter was written for.
    if (encode(kImageText) != std::vector<std::uint32_t>{kImageToken} ||
        encode(kImageBeginText) != std::vector<std::uint32_t>{kImageBeginToken} ||
        encode(kImageEndText) != std::vector<std::uint32_t>{kImageEndToken})
        throw std::runtime_error("--vision: the tokenizer does not hold the image tokens of this model");
    st->adapter = std::make_unique<VisionAdapter>(st->model, strix::model_content_identity_hex(sha), kImageToken);
    st->adapter->bind(st->source, st->weights);
    st->encoder = std::make_unique<vision::ReferenceEncoder>(st->adapter->config(), st->weights);
    st->cache = std::make_unique<vision::EmbedCache>(vision::EmbedCache::budget_from_env());
    vision::ApiMediaOptions o;
    o.placeholder = kImageText;
    o.part_prefix = kImageBeginText;
    o.part_suffix = kImageEndText;
    o.max_rows = st->adapter->resize().max_rows;
    o.lm_width = kLanguageHidden;
    o.limits.max_bytes = limits.max_image_bytes;
    o.limits.max_pixels = limits.max_pixels;
    st->bridge = std::make_shared<vision::ApiMediaBridge>(*st->adapter, *st->encoder, st->weights, st->cache.get(), st->interner, std::move(o));
    return st;
}

class PromptEmbeds {
public:
    explicit PromptEmbeds(pair::EmbedFrameCrc crc = nullptr) : crc_(crc) {}

    // The width of every provided row: the language model's hidden size (bounds a peer's embed frame before it is read).
    [[nodiscard]] static constexpr std::uint32_t row_width() noexcept { return kLanguageHidden; }

    // The lead: the table of the request whose prompt is in prefill (GenerationJob::embeds).
    void bind(std::shared_ptr<const session::EmbedTable> table) { table_ = std::move(table); }
    // The lead: the frame of a chunk's spans; only spans the peer does not hold yet carry their rows.
    [[nodiscard]] std::vector<std::byte> frame(std::span<const std::uint32_t> ids) {
        if (!table_) throw std::logic_error("embed frame: no table is bound");
        return pair::encode_embed_frame(*table_, ids, ledger_, crc_);
    }
    // The follower: re-interns the frame's rows, binds the resulting table and rewrites ids to its handles.
    void received(std::span<const std::byte> frame, std::vector<std::uint32_t>& ids) {
        if (!receiver_) receiver_.emplace();
        auto got = receiver_->receive(frame, ids, crc_);
        table_ = std::move(got.table);
        ids = std::move(got.ids);
    }
    // A prompt chunk: vocabulary ids and the splice of its provided rows (empty for text; nothing is looked up then).
    // A chunk may start inside a span; rows of a span appear in order.
    [[nodiscard]] session::ResolvedEmbedRun resolve(std::span<const std::uint32_t> tokens) const {
        session::EmbedResolveRules rules;
        rules.width = kLanguageHidden;
        return session::resolve_embed_run(tokens, table_, rules);
    }
    [[nodiscard]] static bool holds_virtual_ids(std::span<const std::uint32_t> tokens) noexcept {
        for (const std::uint32_t id : tokens)
            if (session::is_embed_id(id)) return true;
        return false;
    }

private:
    pair::EmbedFrameCrc crc_;
    std::shared_ptr<const session::EmbedTable> table_;
    pair::EmbedFrameLedger ledger_;                    // what the follower holds (lead)
    std::optional<pair::EmbedFrameReceiver> receiver_; // built by the first image prompt (follower)
};

}  // namespace strix::glm::serve
