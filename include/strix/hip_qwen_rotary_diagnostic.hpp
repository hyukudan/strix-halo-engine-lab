#pragma once
#include "strix/model_contract.hpp"
#include <vector>

namespace strix::qwen_rotary_diagnostic {
// Linked only by BUILD_TESTING. These fixtures execute the actual production
// kernels, including their reduction and BF16 publication operations.
enum class Site { query, key, index_query, dynamic_query, known_rows, grouped,
    wave, batch_grouped, batch_wave, pooled, batch_pooled, score, known_score,
    mtp_norm, mtp_score, count };
std::vector<float> run(Site, const ModelContract&, std::uint64_t position, bool frozen_native = false);
void publication_probes(const ModelContract&);
void rejection_probes(const ModelContract&);
void multirow_probes(const ModelContract&);
void device_admission_probes(const ModelContract&);
void fail_next_retirement(unsigned step);
void fail_next_reader(unsigned step);
void wait_retirements();
void lifetime_probes(std::shared_ptr<const HipQwenRotaryTable>);
} // namespace strix::qwen_rotary_diagnostic
