#pragma once
// MiniMax DSpark drafter: the EXL3 block linears (5 layers x q, k, v, o, gate, up, down) and the fc of an EXL3 requant of
// the draft (dspark-exl3-K*), for DeviceDrafter's linear hooks.
//
// Two implementations of the same bits per row:
//   lib   strix_gemv (<= 8 rows) / strix_gemm (more rows) of the strix-trellis library, one call per linear (the
//         attach_draft_exl3 path of 9bc47241);
//   opt   strix::minimax::moe::opt::enqueue_linears_scheduled on raw device copies (strix_gemv bits per row, batch invariant,
//         the measured per-shape schedule regardless of the engine's process-wide STRIX_MMX_LINEAR_V pin;
//         mmx_moe.hpp), the linears that share an input in one launch pair: q + k + v, gate + up (k + v in appends).
// Rows > 8 (32-row context appends: k, v, fc) and the fc always take the library, so their bits never change.
// One instance serves one stream at a time (its workspaces are shared by consecutive calls; stream order serialises them).
#include "mm_common.hpp"
#include "minimax/moe/mmx_moe.hpp"

#include <strix_trellis.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace strix::minimax::engine {

class DraftLinears {
public:
    // Kinds = DeviceDrafter::LinearKind: 0..6 = q, k, v, o, gate, up, down of a layer, 7 = fc (layer ignored).
    static constexpr unsigned kLayers = 5, kKinds = 7, kKindFc = 7, kBlock = kLayers * kKinds;
    enum class Mode { lib, opt, load };  // load = both copies resident, switch with set_opt()
    // Mode from STRIX_MM_DRAFT_LIN: unset / "opt" -> opt, "lib" -> lib, "load" -> load (opt on).
    static Mode mode_from_env();
    DraftLinears(const std::string& dir, Mode mode);
    ~DraftLinears();
    DraftLinears(const DraftLinears&) = delete;
    DraftLinears& operator=(const DraftLinears&) = delete;

    [[nodiscard]] bool has_fc() const noexcept { return has_fc_; }
    [[nodiscard]] Mode mode() const noexcept { return mode_; }
    [[nodiscard]] bool opt() const noexcept { return opt_on_; }
    void set_opt(bool on);  // only with Mode::load (A/B in one process)
    [[nodiscard]] std::uint64_t device_bytes() const noexcept { return device_bytes_; }
    [[nodiscard]] unsigned input(unsigned layer, unsigned kind) const { return dims_[slot(layer, kind)].cols; }
    [[nodiscard]] unsigned output(unsigned layer, unsigned kind) const { return dims_[slot(layer, kind)].rows; }
    [[nodiscard]] unsigned bits(unsigned layer, unsigned kind) const { return dims_[slot(layer, kind)].bits; }

    // out[r][0..output) for r < rows. Stream-ordered, no synchronisation.
    void run(unsigned layer, unsigned kind, const float* x, std::size_t rows, float* out, hipStream_t stream);
    // Block kinds (0..6) of one layer that read the same x (count <= 4): one call per kind on lib, one enqueue_linears on opt.
    void run_group(unsigned layer, const std::uint32_t* kinds, unsigned count, const float* x, std::size_t rows, float* const* outs,
                   hipStream_t stream);
    // One call of every shape the drafter uses (the first library call of a context pays its setup), synchronises.
    void warm_up();

private:
    struct Dim {
        unsigned rows = 0, cols = 0, bits = 0;  // outputs, inputs
    };
    static constexpr unsigned kFcSlot = kBlock;  // storage slot of the fc (block linears: layer * 7 + kind)
    unsigned slot(unsigned layer, unsigned kind) const;
    bool lib_has(unsigned s) const { return s < lib_.size() && lib_[s] != nullptr; }
    void lib_mm(unsigned s, const float* x, float* out, std::size_t rows, hipStream_t stream);
    void release() noexcept;
    detail::PackReader pack_;
    Mode mode_;
    bool opt_on_ = false, has_fc_ = false;
    strix_context ctx_ = nullptr;
    std::vector<strix_weights> lib_;        // [36] by slot, null where not prepared
    std::vector<moe::opt::Exl3> raw_;       // [35] by slot, empty in Mode::lib
    std::vector<Dim> dims_;                 // [36] by slot
    std::vector<void*> allocations_;
    void* lib_ws_ = nullptr;
    std::uint64_t lib_ws_bytes_ = 0;
    void* opt_ws_ = nullptr;
    float *sx_ = nullptr, *sy_ = nullptr;  // [32][max input] zeros, [32][max output]: workspace queries, warm-up
    std::uint64_t device_bytes_ = 0;
};

}  // namespace strix::minimax::engine
