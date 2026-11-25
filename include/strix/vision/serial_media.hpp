#pragma once
// Image input of one serial CLI run (serve::SerialMediaInput) for any family with a vision::VisionAdapter: the
// checkpoint's tower tensors opened through weight::TensorLookup (safetensors), bound by the adapter, an encoder
// backend made by the caller (device tower or the host reference), an interner and the shared ApiMediaBridge.
// The family supplies only its adapter, its tower tensor schema, the backend and its placeholder text.
//
// Header only; link strix_vision_api_media and the weight library.
#include "strix/serve/serial_generation_cli.hpp"
#include "strix/session/embed_rows.hpp"
#include "strix/vision/adapter.hpp"
#include "strix/vision/api_media.hpp"
#include "strix/vision/lookup_weights.hpp"
#include "strix/weight/tensor_access.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::vision {

struct SerialMediaSpec {
    std::string shard;                         // safetensors file holding the tower tensors
    std::vector<weight::TensorSpec> tensors;   // the tower inventory (exact)
    std::string schema_version = "safetensors@1";
    std::string ignored_prefix;                // tensors of the language model in the same file
    std::string placeholder;                   // template text of the image token (one id)
    std::uint32_t lm_width = 0;
    unsigned axes = 0;                         // 0: the LM position of a row is its prompt index
    std::shared_ptr<const weight::Source> shard_source;  // already-open shard, when part of a confined asset set
};

class SerialMedia final : public serve::SerialMediaInput {
public:
    using BackendFactory = std::function<std::unique_ptr<EncoderBackend>(const EncoderConfig&, const EncoderWeights&)>;

    SerialMedia(std::unique_ptr<VisionAdapter> adapter, SerialMediaSpec spec, const BackendFactory& make_backend)
        : spec_(std::move(spec)), adapter_(std::move(adapter)), source_(lookup_) {
        if (!adapter_ || !make_backend) throw std::invalid_argument("serial media: adapter and backend are required");
        const weight::TensorSchema schema{weight::Container::safetensors, spec_.schema_version, spec_.tensors,
                                          spec_.ignored_prefix};
        const auto status = spec_.shard_source ? lookup_.open_safetensors(std::move(spec_.shard_source), schema)
                                                : lookup_.open_safetensors(spec_.shard, schema);
        if (!status)
            throw std::runtime_error("serial media: tower tensors: " + status.detail);
        adapter_->bind(source_, weights_);
        backend_ = make_backend(adapter_->config(), weights_);
        if (!backend_) throw std::runtime_error("serial media: no encoder backend");
        ApiMediaOptions options;
        options.placeholder = spec_.placeholder;
        options.max_rows = adapter_->resize().max_rows;
        options.lm_width = spec_.lm_width;
        options.axes = spec_.axes;
        bridge_.emplace(*adapter_, *backend_, weights_, nullptr, interner_, std::move(options));
    }

    [[nodiscard]] serve::ApiMediaAdapter& adapter() override { return *bridge_; }
    [[nodiscard]] const ApiMediaBridge& bridge() const { return *bridge_; }

private:
    SerialMediaSpec spec_;
    std::unique_ptr<VisionAdapter> adapter_;
    weight::TensorLookup lookup_;
    // Owns the FP32 copy of every tensor the adapter read: the bound weights (and the adapter's own tensors, e.g. a
    // projector or class row) are views into it and are read again at every encode (positions, the host reference,
    // a lazily built device panel), so it lives as long as they do. Declared before weights_ and after lookup_.
    LookupWeights source_;
    EncoderWeights weights_;
    std::unique_ptr<EncoderBackend> backend_;
    session::EmbedInterner interner_;
    std::optional<ApiMediaBridge> bridge_;
};

}  // namespace strix::vision
