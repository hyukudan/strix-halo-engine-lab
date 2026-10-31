// Device MTP sessions: shared immutable DeviceModel, reusable drafters (no per-request state besides one hidden row).
#include "device.hpp"
#include "session.hpp"

#include <cstdlib>
#include <mutex>
#include <string_view>

namespace strix::gemma4::eagle::mtp {
namespace {
bool enabled(const char* name) {
    const auto* v = std::getenv(name);
    return !v || std::string_view(v) != "0";
}
struct Assets {
    const Weights& weights;
    Tables tables;
    gpu::Buffer storage;
    LinearView embedding;
    std::pair<std::uint32_t, std::uint32_t> layers;
    std::shared_ptr<const DeviceModel> model;
    std::vector<std::unique_ptr<Drafter>> spare;
    std::mutex mutex;
    bool resident = false;
    Assets(const Weights& w, Tables t, LinearView host, engine::Backend* backend)
        : weights(w), tables(t), embedding(host), layers(shared_kv_layers(backend->config())) {
        if (enabled("STRIX_G4_DRAFT_HEAD_SHARE")) {
            const auto view = backend->resident_embedding(host);
            if (view.data) { embedding = view; resident = true; }
        }
    }
    std::unique_ptr<Drafter> acquire() {
        std::lock_guard lock(mutex);
        if (!resident && !storage.size()) {
            storage.resize(std::size_t(embedding.outputs) * embedding.row_bytes);
            storage.upload(embedding.data, storage.size());
            embedding.data = storage.as<std::uint8_t>();
        }
        if (!model) model = std::make_shared<DeviceModel>(weights, tables, embedding);
        if (!spare.empty()) {
            auto d = std::move(spare.back());
            spare.pop_back();
            return d;
        }
        return std::make_unique<DeviceDrafter>(model);
    }
    void release(std::unique_ptr<Drafter> d) noexcept {
        try {
            std::lock_guard lock(mutex);
            if (spare.size() < 8) spare.push_back(std::move(d));
        } catch (...) {}
    }
};
} // namespace

engine::DraftFactory device_drafts(const Weights& weights, Tables tables, LinearView host_embedding, engine::Backend* backend,
                                   bool prepare) {
    if (!backend) throw std::invalid_argument("mtp device drafts need the device backend (target KV views)");
    auto assets = std::make_shared<Assets>(weights, tables, host_embedding, backend);
    if (prepare) assets->release(assets->acquire());
    return [assets] {
        return std::make_unique<Session>(assets->acquire(), assets->layers,
                                         [assets](std::unique_ptr<Drafter> d) { assets->release(std::move(d)); });
    };
}

} // namespace strix::gemma4::eagle::mtp
