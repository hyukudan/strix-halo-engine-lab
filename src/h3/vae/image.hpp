#pragma once
// Image VAE: shared scalar-policy CNN/ViT operators with image geometry and posterior policy.
#include <functional>
#include <memory>
#include "h3/block/safetensors.hpp"
#include "h3/vae/precision.hpp"

namespace strix::h3::vae {
// Canonical raw RGB tile assembly (same halo/blend/crop policy as ImageVae).
// The callback receives the y-major tile index and pixel H,W.
std::vector<float> assemble_image_tiles(int height, int width,
    const std::function<std::vector<float>(std::size_t,int,int)>& tile);
class ImageVae {
public:
    using TileFn = std::function<std::vector<float>(const std::vector<float>&, int, int)>;
    using TileEval = std::function<std::vector<float>(std::size_t)>;
    // Optional ownership/gather policy. Sizes are raw FP32 RGB element counts.
    // Evaluate only owned indices, then return every tile in canonical order.
    using TileDispatch = std::function<std::vector<std::vector<float>>(const std::vector<std::size_t>&, const TileEval&)>;
    explicit ImageVae(const std::string& weights);
    // Normalized latent [24][H/16][W/16] and RGB planar pixels [3][H][W] in [0,1].
    std::vector<float> encode(const std::vector<float>& pixels, int height, int width, const TileFn& moments = {});
    std::vector<float> decode(const std::vector<float>& latent, int height, int width, const TileFn& decoder = {},
                              const TileDispatch& dispatch = {});
    const HeadT<float>& head() const { return head_; }
    static std::vector<float> posterior(const std::vector<float>& moments, const std::vector<float>& noise,
                                       const std::vector<float>& mean, const std::vector<float>& sd);
private:
    block::SafeTensors st_;
    HeadT<float> head_;
    std::vector<LayerT<float>> layers_;
    int rope_axes_=3;
    std::unique_ptr<enc::WeightsT<float>> encoder_;
    void load_encoder();
};
}
