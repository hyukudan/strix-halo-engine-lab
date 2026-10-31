#pragma once

// e4b-eagle on strict g4-v1 device primitives (gfx1151 wave32). Bit-identical to HostDrafter by construction:
// linears/norm epilogues/GeGLU/attention are the target's own kernels; the five eagle kernels (tap norms, input
// embed+norms, Q/K/V head norms+RoPE+KV write, ring publish, select) use the same scalar recipe and lane order.
//
// Per committed-row stage (n <= 8 rows, K/V only): 7 launches (6 for combine=sum) + 3 small H2D copies, one sync.
// Publish: 1 launch + 2 D2D copies, one sync.
// draft(k): depth 1 = 13 launches (q, heads, 4x attention, o, epilogue, gate_up, down, epilogue, head, select),
//           depth >= 2 = 16 launches (+ input, in_proj, input norm, fused qkv; 15 for sum). All k depths are
//           enqueued without host synchronization; one download of the k proposals/probabilities at the end.
#include "eagle.hpp"
#include "gemma4/kernels/ops.hpp"

#include <memory>

namespace strix::gemma4::eagle {

// Immutable resident coefficients, fused QKV, RoPE table and d2t, built once by the loaded-model owner.
// target_embedding must be a DEVICE view (e.g. DeviceTarget::resident_embedding) that outlives this object.
class DeviceModel {
public:
    DeviceModel(const Weights& weights, RopeView rope, LinearView target_embedding);
    ~DeviceModel();
    DeviceModel(const DeviceModel&) = delete;
    DeviceModel& operator=(const DeviceModel&) = delete;
    [[nodiscard]] const Config& config() const;
    [[nodiscard]] std::size_t resident_bytes() const;
    // Packed weight bytes read by one chained draft depth / by depth 1 / by a stage of one row (logical traffic).
    [[nodiscard]] std::size_t step_weight_bytes() const;
    [[nodiscard]] std::size_t first_depth_weight_bytes() const;
    [[nodiscard]] std::size_t stage_weight_bytes() const;
    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
    friend class DeviceDrafter;
};

class DeviceDrafter final : public Drafter {
public:
    explicit DeviceDrafter(std::shared_ptr<const DeviceModel> model, bool shared_weights = true);
    ~DeviceDrafter() override;
    DeviceDrafter(const DeviceDrafter&) = delete;
    DeviceDrafter& operator=(const DeviceDrafter&) = delete;
    [[nodiscard]] const Config& config() const override;
    [[nodiscard]] std::uint64_t frontier() const override;
    [[nodiscard]] const Timings& timings() const override; // GPU event times
    void reset() override;
    void stage(const Rows& rows) override;
    void publish() override;   // a failure here poisons the instance (never reused)
    void discard() noexcept override;
    Draft draft(std::uint32_t anchor, std::uint64_t position, std::uint32_t k) override;
    [[nodiscard]] bool reusable() const noexcept;
    [[nodiscard]] const DeviceModel& model() const;
    [[nodiscard]] std::vector<float> key(std::uint64_t position) const;
    [[nodiscard]] std::vector<float> value(std::uint64_t position) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strix::gemma4::eagle
