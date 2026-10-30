#include "strix/qsa_committed_pool.hpp"
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void require(bool ok,const char* what) {if(!ok){std::fprintf(stderr,"FAIL: %s\n",what);std::exit(1);}}
template<class F> void refuses(F f) {bool threw=false;try{f();}catch(const std::exception&){threw=true;}require(threw,"invalid admission refused");}
// The oracle is raw row history, not a second frontier implementation. Every
// reused complete key must carry the four exact current source versions.
struct Oracle {
    strix::MtpQsaPoolPlan plan=strix::mtp_qsa_pool_plan(256);
    strix::QsaCommittedFrontier frontier;
    std::array<std::uint64_t,256> raw{};
    std::array<std::array<std::uint64_t,4>,64> pooled{};
    std::uint64_t version=0;
    void replace(std::uint64_t first,std::uint32_t rows) {
        for(std::uint64_t r=first;r<first+rows;++r) raw[r]=++version;
    }
    void check(std::uint32_t blocks) const {
        require(frontier.committed<=frontier.raw_end/4,"committed frontier inside canonical prefix");
        require(frontier.provisional<=frontier.raw_end/4+plan.provisional_blocks,"bounded provisional suffix");
        for(std::uint32_t b=0;b<blocks;++b)for(unsigned r=0;r<4;++r)
            require(pooled[b][r]==raw[4U*b+r],"leased block equals fresh raw history");
    }
    void build(std::uint32_t blocks) {
        const auto prior=frontier.committed;
        for(auto b=prior;b<blocks;++b)for(unsigned r=0;r<4;++r)pooled[b][r]=raw[4U*b+r];
        frontier.built(blocks,plan);check(blocks);
    }
    void write(std::uint64_t first,std::uint32_t rows,bool certified=true) {
        frontier.before_rewrite(first,rows,plan,certified);replace(first,rows);
        build(static_cast<std::uint32_t>((first+rows)/4));
        // Each physical row sees only its own complete block prefix and tail.
        for(std::uint64_t v=first+1;v<=first+rows;++v)check(static_cast<std::uint32_t>(v/4));
    }
};
}
int main() {
    using namespace strix;
    for(std::uint32_t u=0;u<65536;++u) {
        const auto bits=u<<16U;const auto value=std::bit_cast<float>(bits);
        require((std::bit_cast<std::uint32_t>(value)>>16U)==u,"exhaustive BF16 pack/widen words including NaN payloads");
    }
    for(const auto capacity:{32768U,131072U,262144U,393216U,524288U,1048576U,32771U,3U,1U}) {
        const auto p=mtp_qsa_pool_plan(capacity);
        require(p.entries==13 && p.blocks==capacity/4,"thirteen exact floor-capacity entries");
        require(p.entry_stride%256==0 && p.key_bytes==13ULL*(capacity/4)*256,"packed stride/payload; no sixteen-slot or FP32 mirror");
        require(p.scratch_bytes==8ULL*p.maxima_stride*4,"one shared eight-row scratch envelope");
        require(p.reserved_bytes==p.key_bytes+p.scratch_bytes+p.metadata_bytes,"one exact service reservation");
        const auto budget=p.reserved_bytes+1000;
        require(mtp_qsa_pool_fits(p,budget,300,400,300),"one reservation across three window owners");
        require(!mtp_qsa_pool_fits(p,budget-1,300,400,300),"one-byte admission boundary");
        require(!mtp_qsa_pool_fits(p,p.reserved_bytes-1,0,0,0),"insufficient memory declines before allocation");
        require(2*p.reserved_bytes==2*(p.key_bytes+p.scratch_bytes+p.metadata_bytes),"separate HTTP slot reservations");
    }
    require(mtp_qsa_pool_plan(32768).key_bytes==27262976,"32K payload");
    require(mtp_qsa_pool_plan(131072).key_bytes==109051904,"131K payload");
    require(mtp_qsa_pool_plan(262144).key_bytes==218103808,"262K payload");
    require(mtp_qsa_pool_plan(393216).key_bytes==327155712,"393K payload");
    require(mtp_qsa_pool_plan(1048576).key_bytes==872415232,"1M payload");
    refuses([]{(void)mtp_qsa_pool_plan(0);});refuses([]{(void)mtp_qsa_pool_plan(UINT64_MAX);});
    refuses([]{(void)mtp_qsa_pool_plan(256,10);});refuses([]{(void)mtp_qsa_pool_plan(256,9,UINT64_MAX);});
    for(const auto rows:{2U,4U,8U})for(unsigned residue=0;residue<4;++residue)for(unsigned accepted=0;accepted<=rows;++accepted) {
        Oracle o;const std::uint64_t start=100+residue;
        o.replace(0,static_cast<std::uint32_t>(start));o.frontier.raw_end=start;o.build(static_cast<std::uint32_t>(start/4));
        o.write(start,rows);o.frontier.commit(start+accepted);o.check(o.frontier.committed);
        // R8 idle-owner hazard: serial fallback rewrites the rejected block,
        // then a different R2 owner starts beyond it. All residues are covered.
        for(std::uint64_t pos=start+accepted;pos<start+rows+4;++pos) {
            o.write(pos,1);o.frontier.commit(pos+1);o.check(o.frontier.committed);
        }
        o.write(start+rows+4,2);o.frontier.commit(start+rows+5);o.check(o.frontier.committed);
        // Restore an interior block, including mid-block accepted/EOS/budget frontiers.
        for(unsigned replay=0;replay<3;++replay) {
            const auto changed=start+accepted;
            o.frontier.invalidate(changed);o.replace(changed,4);o.frontier.commit(changed+4);
            o.write(changed+4,4);o.frontier.commit(changed+8);o.check(o.frontier.committed);
        }
        // Full import, cancellation at every store/build/score/commit boundary,
        // and prompt replacement reuse the allocation but never its validity.
        for(unsigned phase=0;phase<4;++phase) {
            o.frontier.invalidate(0);o.replace(0,120);o.frontier.commit(120);
            o.frontier.before_rewrite(120,8,o.plan,true);o.replace(120,8);
            if(phase>0)o.build(32);
            if(phase>1)o.check(32);
            if(phase>2)o.frontier.commit(125);
            o.frontier.invalidate(0);require(o.frontier.committed==0 && o.frontier.provisional==0,"cancel clears speculative and committed validity");
            o.replace(0,100);o.frontier.commit(100);o.write(100,2);o.frontier.commit(102);o.check(o.frontier.committed);
        }
    }
    // Predictor A is independent from target A+1. Equal accepted token IDs do
    // not certify proposed raw keys: authoritative teacher/fast replay replaces them.
    for(unsigned residue=0;residue<4;++residue) {
        Oracle draft,target;const std::uint64_t base=100+residue;
        draft.replace(0,static_cast<std::uint32_t>(base));draft.frontier.raw_end=base;draft.build(static_cast<std::uint32_t>(base/4));
        target.replace(0,static_cast<std::uint32_t>(base+1));target.frontier.raw_end=base+1;target.build(static_cast<std::uint32_t>((base+1)/4));
        for(unsigned i=0;i<8;++i)draft.write(base+i,1,false);
        require(draft.frontier.raw_end==base && target.frontier.raw_end==base+1,"target/predictor frontiers independent");
        draft.frontier.invalidate(base);draft.frontier.commit(base);
        for(unsigned i=0;i<8;++i) {draft.write(base+i,1);draft.frontier.commit(base+i+1);draft.check(draft.frontier.committed);}
        require(draft.raw[base]!=target.raw[base],"teacher versions deliberately differ despite equal tokens");
        // First-row reuse only after the existing exact source/teacher-pair proof:
        draft.frontier.invalidate(base);draft.frontier.commit(base+1);draft.write(base+1,1);draft.frontier.commit(base+2);
    }
    Oracle bounded;bounded.frontier.raw_end=0;
    refuses([&]{bounded.frontier.before_rewrite(0,10,bounded.plan,true);});
    refuses([&]{bounded.frontier.built(4,bounded.plan);});
    std::puts("PASS: packed words, single-owner bytes, causal frontiers, acceptance/rejection, replay/restore/import/cancel and prompt epochs");
}
