#pragma once

#include "inkling2_mini_resident_route.hpp"
#include "inkling2_mini_weight_options.hpp"
#include "strix/serve/cohort_decode.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace strix::inkling2 {

struct MiniServeOptions {
    std::string checkpoint;
    std::string tokenizer;
    std::string devices_path;
    std::string device_id;
    std::string device_profile;
    unsigned context = 8192;
    unsigned prefill_rows = 128;
    unsigned slots = 0;
    std::uint64_t kv_bytes = 0;
    MiniWeightOptions weight_options;
    profile::Name tier = profile::Name::standard;
    // Resolved operand class of the FAST W4A4 expert lever (mini_moe_w4a4_class; bf16 = off): the served prefill runs
    // the class the profile line reports, as the CLI and the gate do.
    moe_core::grouped_w4a4::Activations moe_w4a4 = moe_core::grouped_w4a4::Activations::bf16;
    bool packed_decode = false;       // resolved decode expert form (mini_serve_packed_decode)
    bool packed_decode_requested = false;  // --packed-decode / --grouped-decode given
};

// Cohort width the served decode form is chosen for: the configured slots, or every prefill row without --slots
// (make_mini_device_cohort sizes the cohort the same way).
inline unsigned mini_serve_cohort_rows(const MiniServeOptions& opt) { return opt.slots ? opt.slots : opt.prefill_rows; }

// HIP serving decodes experts on the CLI's packed form (kMiniPackedDecodeDefault) below kMiniGroupedDecodeMinSlots
// cohort rows, and on the grouped tile kernel from there on. A server keeps its form for its whole life, so a lane's
// tokens never depend on the cohort size; a packed server shares the CLI's EXACT identity. CUDA serving keeps the
// grouped tile kernel until its cohort decode is measured (the CUDA packed form runs one launch per lane).
// --packed-decode / --grouped-decode override either backend.
inline bool mini_serve_packed_decode(std::string_view backend, std::optional<bool> requested, unsigned cohort_rows) {
    if (requested) return *requested;
    return backend == "hip" && kMiniPackedDecodeDefault && cohort_rows < kMiniGroupedDecodeMinSlots;
}

// Route line value of the choice: the form and why ("requested", "backend", "slots<N" or "slots>=N").
inline std::string mini_serve_decode_form(const MiniServeOptions& opt, std::string_view backend) {
    std::string form = opt.packed_decode ? "packed" : "grouped";
    form += " moe_decode_choice ";
    if (opt.packed_decode_requested) return form + "requested";
    if (backend != "hip") return form + "backend";
    return form + (opt.packed_decode ? "slots<" : "slots>=") + std::to_string(kMiniGroupedDecodeMinSlots);
}

struct MiniServeCohort {
    std::unique_ptr<serve::CohortBackend> backend;
    weight::WeightLoadPlan weights;
};

MiniServeCohort make_mini_hip_cohort(const MiniServeOptions&);
MiniServeCohort make_mini_cuda_cohort(const MiniServeOptions&);

} // namespace strix::inkling2
