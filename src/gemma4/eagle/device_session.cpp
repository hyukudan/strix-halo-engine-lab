// Device e4b-eagle sessions: shared immutable model, reusable per-request drafters (mirrors engine DraftAssets).
#include "device.hpp"
#include "session.hpp"

#include <cstdlib>
#include <mutex>
#include <string_view>

namespace strix::gemma4::eagle {
namespace {
bool enabled(const char* name) {
    const auto* v = std::getenv(name);
    return !v || std::string_view(v) != "0";
}
struct Assets {
    const Weights& weights;
    RopeView rope;
    gpu::Buffer storage;
    LinearView embedding;
    std::shared_ptr<const DeviceModel> model;
    std::vector<std::unique_ptr<Drafter>> spare;
    std::mutex mutex;
    bool resident = false, reuse = enabled("STRIX_G4_DRAFT_REUSE");
    Assets(const Weights& w, RopeView r, LinearView host, engine::Backend* backend) : weights(w), rope(r), embedding(host) {
        if (backend && enabled("STRIX_G4_DRAFT_HEAD_SHARE")) {
            const auto view = backend->resident_embedding(host);
            if (view.data) { embedding = view; resident = true; }
        }
    }
    void load() {
        if (!resident && !storage.size()) {
            storage.resize(std::size_t(embedding.outputs) * embedding.row_bytes);
            try { storage.upload(embedding.data, storage.size()); }
            catch (...) { storage.resize(0); throw; }
            embedding.data = storage.as<std::uint8_t>();
        }
        if (!model) model = std::make_shared<DeviceModel>(weights, rope, embedding);
    }
    std::unique_ptr<Drafter> acquire() {
        std::lock_guard lock(mutex);
        load();
        if (reuse && !spare.empty()) {
            auto result = std::move(spare.back());
            spare.pop_back();
            result->reset();
            return result;
        }
        return std::make_unique<DeviceDrafter>(model);
    }
    void release(std::unique_ptr<Drafter> drafter) noexcept {
        auto* device = dynamic_cast<DeviceDrafter*>(drafter.get());
        if (!reuse || !device || !device->reusable()) return;
        try {
            std::lock_guard lock(mutex);
            if (spare.size() < 8) spare.push_back(std::move(drafter));
        } catch (...) {} // dropping a spare never changes outputs
    }
};
} // namespace

engine::DraftFactory device_drafts(const Weights& weights, RopeView rope, LinearView host_embedding, engine::Backend* backend,
                                   bool prepare) {
    auto assets = std::make_shared<Assets>(weights, rope, host_embedding, backend);
    if (prepare) assets->release(assets->acquire());
    return [assets] {
        return std::make_unique<Session>(assets->acquire(), [assets](std::unique_ptr<Drafter> d) { assets->release(std::move(d)); });
    };
}

} // namespace strix::gemma4::eagle
