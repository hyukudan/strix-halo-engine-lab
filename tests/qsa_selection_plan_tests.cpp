#include "strix/qsa_selection_plan.hpp"
#include "strix/session_state_cache.hpp"
#include "strix/model_content_hash.hpp"
#include <cstring>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    expect(strix::build_qsa_hierarchical_selection_layout(2048).scratch_bytes == 0 &&
        strix::build_qsa_hierarchical_selection_layout(2051).scratch_bytes == 0 &&
        strix::build_qsa_hierarchical_selection_layout(2052).scratch_bytes == 4096 &&
        strix::build_qsa_hierarchical_selection_layout(131072).scratch_bytes == 131072,
        "hierarchical scratch follows actual complete-block buckets including tails");
    for (const auto invalid : {0ULL, 131073ULL, ~0ULL}) {
        bool rejected = false;
        try { (void)strix::build_qsa_hierarchical_selection_layout(invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "hierarchical capacity bounds reject zero and overflow without allocating");
    }
    const auto plan =
        strix::build_qsa_compressed_selection_plan(
            2048,
            4);
    expect(plan.block_topk == 512 &&
               plan.final_width == 2051,
           "QSA selection allocates 512 compressed blocks plus partial-group tail capacity");

    std::vector<std::int64_t> blocks(
        static_cast<std::size_t>(
            plan.block_topk),
        -1);
    blocks[0] = 0;
    blocks[1] = 2;

    const auto expanded =
        strix::expand_qsa_block_selection_reference(
            plan,
            blocks,
            10,
            12);

    expect(expanded.size() == 2051,
           "QSA expansion preserves fixed token-budget-plus-tail width");
    expect(
        std::vector<std::int64_t>(
            expanded.begin(),
            expanded.begin() + 9) ==
            std::vector<std::int64_t>({
                0, 1, 2, 3,
                8, 9, 10, 11,
                8}),
        "QSA expansion preserves selected blocks then appends visible partial-group tail");

    const std::vector<double> group{
        1.0, 5.0,
        3.0, 7.0,
        5.0, 9.0,
        7.0, 11.0};
    expect(
        strix::average_pool_qsa_key_group_reference(
            group,
            4,
            2) ==
            std::vector<double>({4.0, 8.0}),
        "QSA compressed key reference averages complete groups in FP64 host oracle");

    // CPU lifecycle oracle: only raw rows are serialized; derived complete-block
    // keys must rebuild on a fresh restore and clamp on a reused parked owner.
    // This exercises host semantics, not device FP32 qualification.
    {
        using namespace strix;
        const auto heap=[](std::uint64_t n) {return PrefixCacheBuffer{
            std::shared_ptr<std::byte>(new std::byte[n],std::default_delete<std::byte[]>()),n};};
        const auto hash=[](std::string_view s) {return sha256_bytes(std::as_bytes(std::span(s.data(),s.size())));};
        std::vector<double> raw(64);
        for(std::size_t i=0;i<raw.size();++i) raw[i]=double(i%13)-6.0;
        SessionCheckpoint saved; saved.compatibility={hash("model"),hash("engine"),"cpu-pool-session",false};
        saved.frontier.consumed=13; saved.frontier.accepted=13; saved.frontier.tokens.resize(13,1);
        auto backing=heap(13*2*sizeof(double)); std::memcpy(backing.data.get(),raw.data(),backing.bytes);
        saved.sections.push_back({"raw-index-rows",backing});
        const auto envelope=encode_session_checkpoint(saved,heap);
        PrefixStateCache parked({1U<<20,{},0},heap);
        expect(parked.park_session("pool-on",envelope,1),"pool-on session parks raw state");
        const auto resumed=decode_session_checkpoint(parked.resume_session("pool-on"),saved.compatibility);
        std::vector<double> restored(raw.size());
        std::memcpy(restored.data(),resumed.sections[0].payload.data.get(),resumed.sections[0].payload.bytes);
        // Extend raw state after restore; a complete group straddles the saved boundary.
        std::copy(raw.begin()+26,raw.end(),restored.begin()+26);
        const auto cold_pool=[&](std::size_t blocks_count) {
            std::vector<double> out;
            for(std::size_t block=0;block<blocks_count;++block) {
                const auto key=average_pool_qsa_key_group_reference(std::span(raw).subspan(block*8,8),4,2);
                out.insert(out.end(),key.begin(),key.end());
            }
            return out;
        };
        for(bool fresh:{true,false}) {
            std::vector<double> pool(16,-999.0); std::size_t valid=fresh?0:8;
            if(!fresh) {
                const auto retained=cold_pool(3);
                std::copy(retained.begin(),retained.end(),pool.begin());
            }
            for(std::size_t position=13;position<32;++position) {
                valid=std::min(valid,position/4);
                const auto complete=(position+1)/4;
                while(valid<complete) {
                    const auto pooled_group=average_pool_qsa_key_group_reference(
                        std::span(restored).subspan(valid*8,8),4,2);
                    std::copy(pooled_group.begin(),pooled_group.end(),pool.begin()+valid*2); ++valid;
                }
                const auto cold=cold_pool(complete);
                expect(std::memcmp(pool.data(),cold.data(),cold.size()*sizeof(double))==0,
                    "session save/restore/park derived pool byte equality");
            }
        }
    }

    bool invalid = false;
    try {
        static_cast<void>(
            strix::build_qsa_compressed_selection_plan(
                2048,
                3));
    } catch (const std::invalid_argument&) {
        invalid = true;
    }
    expect(invalid,
           "QSA selection rejects non-divisible token budget");

    if (failures == 0) {
        std::cout << "QSA selection plan tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
