#include "../apps/http_slot_capacity.hpp"
#include "../apps/hip_decode_options.hpp"
#include <iostream>
using namespace strix::decode_cli;
int main() {
    int failures = 0;
    const auto expect = [&](bool ok) { if (!ok) ++failures; };
    for (const auto n : {1U, 2U, 8U, 9U, 16U}) {
        const auto bytes = 100U + (n - 1U) * 10U + n * 2U;
        expect(resolve_http_slot_capacity(10, 2, bytes, bytes, 100).slots == n);
        expect(resolve_http_slot_capacity(10, 2, bytes - 1U, bytes, 100).slots == (n > 1U ? n - 1U : 1U));
    }
    expect(resolve_http_slot_capacity(10, 2, 1000, 200, 100).slots == 9U);
    expect(resolve_http_slot_capacity(10, 2, 1000, 1000, 100, 4).slots == 4U);
    expect(resolve_http_slot_capacity(10, 2, {}, 1000, 100).slots == 1U);
    expect(resolve_http_slot_capacity(10, 2, 1000, {}, 100).slots == 1U);
    expect(resolve_http_slot_capacity(UINT64_MAX, 1, UINT64_MAX, UINT64_MAX, 0).slots == 1U);
    expect(resolve_http_slot_capacity(1, 0, UINT64_MAX, UINT64_MAX, 0).slots == UINT32_MAX);
    expect(resolve_http_slot_capacity(10, 2, 1000, 1000, 100, 1000).slots == 75U);
    expect(resolve_http_slot_capacity(10, 2, 1000, 1000, 100, 75).slots == 75U);
    expect(resolve_http_slot_capacity(10, 2, 1, 1, 100).slots == 1U);
    expect(http_eager_slots(25) == 4U);
    expect(http_eager_slots(2) == 2U);
    expect(http_eager_slots(25, 0) == 25U);
    expect(http_eager_slots(25, 1) == 1U);
    for (const bool queued : {false, true})
        for (const bool memory : {false, true})
            for (const auto active : {3U, 4U})
                for (const auto capacity : {4U, 25U})
                    expect(http_slot_should_grow(queued, active, 4, capacity, memory) ==
                        (queued && memory && active == 4U && capacity == 25U));
    for (const bool queued : {false, true})
        for (const auto active : {3U, 4U})
            expect(http_slot_should_grow(queued, active, 4, 25, true, true) == (active == 4U));
    expect(!http_slot_should_grow(false, 4, 4, 4, true, true));
    expect(!http_slot_should_grow(false, 4, 4, 25, false, true));
    expect(http_slot_memory_fits(10, 2, 112, 112, 100));
    expect(!http_slot_memory_fits(10, 2, 111, 112, 100));
    expect(!http_slot_memory_fits(10, 2, 112, {}, 100));
    expect(!http_slot_memory_fits(10, 2, {}, 112, 100));
    expect(!http_slot_memory_fits(10, 2, 112, 111, 100));
    expect(http_slot_memory_fits(10, 0, 110, 110, 100));
    expect(!http_slot_memory_fits(10, 0, 109, 110, 100));
    expect(!http_slot_memory_fits(UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX, 1));
    const auto parse = [](std::vector<std::string> args) {
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        return parse_options(static_cast<int>(argv.size()), argv.data());
    };
    auto parsed = parse({"decoder", "model.gguf", "1"});
    expect(!parsed.http_batch_explicit);
    parsed = parse({"decoder", "model.gguf", "1", "--http-batch", "0"});
    expect(parsed.http_batch_explicit && !parsed.http_batch_cap);
    parsed = parse({"decoder", "model.gguf", "1", "--http-batch", "16"});
    expect(parsed.http_batch_explicit && parsed.http_batch_cap == 16U);
    Options o;
    o.http_serve = true;
    o.mtp.window_prefix = o.mtp.draft_chain = true;
    o.mtp.window_rows_set = {2U, 4U, 8U};
    resolve_http_route(o); expect(o.http_batch == 1U && !o.http_batch_explicit);
    o.http_batch_explicit = true; o.http_batch_cap = 9;
    resolve_http_route(o); expect(o.http_batch == 9U);
    o.http_batch_cap = 0; resolve_http_route(o); expect(o.http_batch == 0U);
    o.http_batch_explicit = false; o.mtp.copy_wide_rows = {16};
    resolve_http_route(o); expect(o.http_batch == 0U);
    o.http_batch_explicit = true; o.http_batch_cap = 2;
    bool rejected = false;
    try { resolve_http_route(o); } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected);
    o.http_batch_explicit = false; o.mtp.copy_wide_rows.clear(); o.mtp.draft_chain = false;
    resolve_http_route(o); expect(o.http_batch == 0U);
    o.mtp.draft_chain = true; o.rtx_state_import = true;
    resolve_http_route(o); expect(o.http_batch == 0U && o.http_slot_reason == "single-only state import");
    o.rtx_state_import = false; o.http_sampling = "sample";
    o.http_batch_explicit = true; o.http_batch_cap = 1U;
    resolve_http_route(o); expect(o.http_batch == 1U);
    std::cout << "HTTP slot tests: failures=" << failures << '\n';
    return failures ? 1 : 0;
}
