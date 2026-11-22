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
};

class SerialMedia final : public serve::SerialMediaInput {
public:
    using BackendFactory = std::function<std::unique_ptr<EncoderBackend>(const EncoderConfig&, const EncoderWeights&)>;

    SerialMedia(std::unique_ptr<VisionAdapter> adapter, SerialMediaSpec spec, const BackendFactory& make_backend)
        : spec_(std::move(spec)), adapter_(std::move(adapter)) {
        if (!adapter_ || !make_backend) throw std::invalid_argument("serial media: adapter and backend are required");
        const weight::TensorSchema schema{weight::Container::safetensors, spec_.schema_version, spec_.tensors,
                                          spec_.ignored_prefix};
        if (const auto status = lookup_.open_safetensors(spec_.shard, schema); !status)
            throw std::runtime_error("serial media: tower tensors: " + status.detail);
        LookupWeights source(lookup_);
        adapter_->bind(source, weights_);
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
    weight::TensorLookup lookup_;  // the bound weights view its storage
    EncoderWeights weights_;
    std::unique_ptr<EncoderBackend> backend_;
    session::EmbedInterner interner_;
    std::optional<ApiMediaBridge> bridge_;
};

}  // namespace strix::vision
