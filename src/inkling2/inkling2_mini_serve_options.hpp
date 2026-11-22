#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace strix::serve { class CohortBackend; }

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
};

std::unique_ptr<serve::CohortBackend> make_mini_hip_cohort(const MiniServeOptions&);
std::unique_ptr<serve::CohortBackend> make_mini_cuda_cohort(const MiniServeOptions&);

} // namespace strix::inkling2
