#include "strix/compact_stage_images.hpp"
#include "strix/execution_arena.hpp"
#include "strix/indexed_attention_workspace.hpp"
#include "strix/q8_projection_organization.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void check(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
}
int main() {
    using namespace strix;
    check(compact_stage_images_requested(nullptr), "default on");
    check(!compact_stage_images_requested("0"), "explicit off");
    for (const char* value : {"", "1", "false"})
        check(compact_stage_images_requested(value), "nonzero policy enables");
    for (const char* value : {"0", "00", "0x0"})
        check(!compact_stage_images_requested(value), "zero-leading values disable consistently");
    // Every BF16 bit pattern: both zero signs, subnormals, infinities, signaling
    // and quiet NaNs, both signs and every payload. No floating arithmetic.
    for (unsigned upper=0;upper<65536U;++upper) {
        const auto word=static_cast<std::uint16_t>(upper);
        check(widen_compact_bits(compact_rounded_bits(widen_compact_bits(word))) == (upper<<16U), "all BF16 bits round-trip");
        for (unsigned lower : {0U,1U,0x7fffU,0x8000U,0xffffU}) {
            const auto raw=(upper<<16U)|lower;
            // Exactly the incumbent integer RNE, including its wrap/payload
            // behavior for NaNs. Compaction must not substitute canonical NaNs.
            const auto rounded=(raw+0x7fffU+((raw>>16U)&1U))&0xffff0000U;
            check(widen_compact_bits(compact_rounded_bits(rounded))==rounded, "producer rounding bits survive storage");
        }
    }
    const auto model=ModelContract::qwen38_flash_next();
    const auto fp32=build_indexed_attention_workspace_layout(model,256,false,1,true);
    const auto compact=build_indexed_attention_workspace_layout(model,256,false,1,true,true);
    check(compact.compact_attention_output && !fp32.compact_attention_output, "explicit physical layout policy");
    check(compact.attention_output_bytes*2U==fp32.attention_output_bytes, "half-size image");
    check(compact.attention_output_offset==fp32.attention_output_offset &&
        compact.attention_scores_offset==fp32.attention_scores_offset && compact.q_gate_offset==fp32.q_gate_offset,
        "raw projections and scores keep their FP32 layout");
    auto invalid=model;invalid.bytes_fp32=1U;
    bool bad_container=false;
    try { (void)build_indexed_attention_workspace_layout(invalid,256,false,1,true,true); }
    catch(const std::invalid_argument&) { bad_container=true; }
    check(bad_container,"compact layout refuses non-FP32 containers");
    const auto savings=fp32.total_bytes-compact.total_bytes;
    check(savings==12288U, "pinned per-row savings");
    check(savings*32768U==402653184U,"32K logical output-image bytes saved");
    ExecutionArenaPlan serial;serial.alignment=256;serial.total_bytes=fp32.total_bytes;
    ExecutionArenaAllocation a{};a.bytes=fp32.total_bytes;
    a.use.buffer.buffer_class=ExecutionBufferClass::indexed_attention_workspace;
    serial.allocations.push_back(a);
    for (unsigned rows : {40U,8192U}) {
        const auto old=build_known_prefill_arena(serial,256,rows);
        const auto small=build_known_prefill_arena(serial,256,rows,0,false,true,savings);
        check(old.physical.total_bytes-small.physical.total_bytes==savings*rows, "arena removes actual storage");
        check(small.row_view.back().allocations[0].offset+compact.total_bytes==small.physical.total_bytes,
            "last row ends at the compact arena boundary");
        check(small.row_stride[0]==compact.total_bytes && small.row_view[0].allocations[0].bytes==compact.total_bytes,
            "physical price and row view agree");
    }
    bool rejected=false;
    try { (void)build_known_prefill_arena(serial,256,40,0,false,true,UINT64_MAX); }
    catch (const std::exception&) { rejected=true; }
    check(rejected,"invalid savings cannot underflow");
    using namespace q8_projection_organization;
    const View input{0x10000U,6144U*2U}, weights{0x100000U,6144U/32U*34U*256U}, output{0x400000U,512U};
    validate_fp32_bf16_views(6144,256,input,weights,output,true);
    rejected=false;
    try { validate_fp32_bf16_views(6144,256,input,weights,output); }
    catch (const std::invalid_argument&) { rejected=true; }
    check(rejected,"legacy validator retains FP32 extent checks");
    rejected=false;
    try { validate_fp32_bf16_views(6144,256,input,weights,{input.address,512U},true); }
    catch (const std::invalid_argument&) { rejected=true; }
    check(rejected,"compact validator retains alias checks");
    std::cout << "compact stage-image parsing, special bits and layouts passed\n";
}
