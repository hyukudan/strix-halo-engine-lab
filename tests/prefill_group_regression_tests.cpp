// Host regressions for group admission; no device, model or HIP runtime required.
#include "strix/deepseek41_prefill.hpp"
#include "strix/serve/prefill_group.hpp"
#include <iostream>
#include <limits>
#include <vector>

using namespace strix::serve;
namespace {
int failures = 0;
void check(bool ok, const char* name) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << name << '\n'; }
}
std::vector<GroupCandidate> prompts(std::uint32_t rows, std::uint32_t route = 0U) {
    return {{0U, rows, 0U, route}, {1U, rows, 1U, route}, {2U, rows, 2U, route}};
}
std::size_t planned(const std::vector<GroupCandidate>& c, PrefillGroupCapacity cap) {
    return plan_prefill_group(c, {}, {}, std::numeric_limits<double>::infinity(), cap).size();
}
}
int main() {
    PrefillGroupCapacity cap;
    cap.rows = 32U;
    check(planned(prompts(20U), cap) == 1U, "32-row device: two 20-row prompts stay solo");
    cap.rows = 40U;
    check(planned(prompts(20U), cap) == 2U, "total row boundary admits two");
    cap = {};
    cap.members = 2U;
    check(planned(prompts(20U), cap) == 2U, "allocated member count");
    cap = {};
    cap.member_rows = 19U;
    check(planned(prompts(20U), cap) == 1U, "member storage bound");
    cap = {};
    cap.window_rows = 8U;
    cap.windows = 5U;
    check(planned(prompts(20U), cap) == 1U, "partial windows count per member");
    cap.windows = 6U;
    check(planned(prompts(20U), cap) == 2U, "six windows admit two partial members");
    cap = {};
    cap.window_rows = 8U;
    cap.split_class_mask = 1U;
    cap.local_rows = 23U;
    check(planned(prompts(20U, 1U), cap) == 1U, "owner restarts: 12+12 local rows exceed 23");
    cap.local_rows = 24U;
    check(planned(prompts(20U, 1U), cap) == 2U, "both rank bounds: 24 local rows fit");
    check(planned(prompts(20U), cap) == 1U, "unsplit members each consume all local rows");
    check(cap.local_count(17U, 1U) == 9U && cap.window_count(17U) == 3U,
          "partial terminal window retains rank zero ownership");
    cap = {};
    cap.rejected_class_mask = 4U;
    check(planned(prompts(20U, 4U), cap) == 1U, "matching dense classes still stay solo");
    cap.members = 0U;
    check(planned(prompts(20U), cap) == 1U, "unsupported options retain solo");

    using strix::deepseek41::DevicePrefill;
    strix::deepseek41::PrefillOptions options;
    check(DevicePrefill::group_options_supported(options), "exact options supported");
    options.bf16_attention = true;
    check(!DevicePrefill::group_options_supported(options), "BF16 group declined");
    options = {}; options.fp16_dense = true;
    check(!DevicePrefill::group_options_supported(options), "FP16 group declined");
    options = {}; options.fast_dense = true;
    check(!DevicePrefill::group_options_supported(options), "FAST dense group declined");
    options = {}; options.fast_attention_flash = true;
    check(!DevicePrefill::group_options_supported(options), "flash group declined");
    options = {}; options.fast_attention = true;
    check(!DevicePrefill::group_options_supported(options), "FAST attention group declined");
    options = {}; options.fast_moe = true;
    check(!DevicePrefill::group_options_supported(options), "FAST MoE group declined");
    options = {}; options.trace = true;
    check(!DevicePrefill::group_options_supported(options), "trace group declined");
    options = {}; options.quality_head_rows = 2U;
    check(!DevicePrefill::group_options_supported(options), "multirow quality head declined");

    PrefillGroupPolicy policy;
    policy.short_rows = 0U;
    check(!policy.groups() && policy.storage_rows(4096U) == 0U, "zero threshold disables allocation");
    policy.short_rows = 1U;
    check(!policy.groups() && policy.storage_rows(4096U) == 0U, "exclusive threshold one has no members");
    policy.short_rows = 64U;
    check(policy.storage_rows(32U) == 32U, "member allocation bounded by context");
    check(policy.storage_rows(4096U) == 63U, "exclusive threshold is distinct from storage");
    check(policy.describe().find("prefill_group: auto") != std::string::npos &&
          policy.describe().find("device eligibility/capacity pending") != std::string::npos,
          "inspection reports requested policy without claiming device admission");
    bool invalid = false;
    try {
        (void)PrefillGroupPolicy::from_env([](const char* key) {
            return std::string(key) == "STRIX_SERVE_PREFILL_GROUP" ? std::string("bad") : std::string();
        });
    } catch (const std::invalid_argument&) { invalid = true; }
    check(invalid, "bad grouping policy rejected before inspection exits");

    struct Waiting { unsigned rows; std::uint32_t overtakes; };
    std::vector<Waiting> waiting{{100U, 0U}};
    for (unsigned i = 0; i < 7U; ++i) waiting.push_back({20U, 0U});
    const auto size = [](const Waiting& q) { return q.rows; };
    const auto skips = [](Waiting& q) -> std::uint32_t& { return q.overtakes; };
    const auto first = select_waiting(waiting, true, size, skips);
    check(first == 1U, "short-first head overtakes the long owner");
    std::vector<std::uint64_t> selected{1U, 2U, 3U, 4U, 5U, 6U, 7U};
    limit_prefill_overtakes(selected, waiting, skips);
    check(selected.size() == 7U && waiting[0].overtakes == 7U, "all seven real overtakes charged");
    for (std::size_t i = 1; i < waiting.size(); ++i) waiting[i].overtakes = 0U;
    check(select_waiting(waiting, true, size, skips) == 1U, "eighth short selection admitted");
    selected = {1U, 2U, 3U, 4U, 5U, 6U, 7U};
    limit_prefill_overtakes(selected, waiting, skips);
    check(selected.size() == 1U && waiting[0].overtakes == 8U, "companions cannot cross starvation bound");
    check(select_waiting(waiting, true, size, skips) == 0U, "protected long owner selected next");
    check(waiting[1].overtakes == 0U, "selected group owners are not charged");
    if (failures != 0) return 1;
    std::cout << "prefill group regressions passed\n";
}
