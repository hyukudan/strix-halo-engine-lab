#include "../apps/qwen_tiled_prompt_plan.hpp"
#include <iostream>
#include <memory>
#include <set>
using namespace strix;
using namespace strix::decode_cli;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
// Adapter lets the frozen old algorithm retain its backend/owner selection expressions.
using KnownWindowOutputPolicy=QwenPlanOutput;
struct Backend { QwenPlanOutput policy; QwenPlanOutput output_policy() const {return policy;} };
struct Owner {};
struct Incumbent {
    using Phase=QwenPlanPhase;
    struct Tail {std::uint32_t rows;std::shared_ptr<Backend> backend;std::shared_ptr<Owner> owner;};
    struct Tiled {std::uint32_t rows;std::shared_ptr<Backend> blocks;std::shared_ptr<Owner> block_owner;
        std::vector<Tail> tails;std::vector<std::uint32_t> ladder_tail_rows;} tiled_;
    struct Block {Backend* backend;Owner* owner;std::uint32_t rows;bool capture,final;};
    QwenTiledPlanConfig config;
    Phase phase_=Phase::blocks;std::size_t tail_=0;int balanced_=-1;
    std::uint64_t pos_=0,n_=0;bool small_=false,adopted_=false;
    Incumbent(const QwenTiledPlanConfig& c,std::uint64_t n):config(c),n_(n) {
        tiled_.rows=c.rows;tiled_.blocks=std::make_shared<Backend>(Backend{c.block.policy});
        tiled_.block_owner=std::make_shared<Owner>();tiled_.ladder_tail_rows=c.ladder;
        for(const auto& t:c.tails) tiled_.tails.push_back({t.rows,std::make_shared<Backend>(Backend{t.policy}),std::make_shared<Owner>()});
    }
    std::optional<Block> next() {
#include "qwen_tiled_incumbent.inc"
    }
};
std::uint64_t shapes=0,units=0;
void compare(const QwenTiledPlanConfig& c,std::uint64_t n,bool small=false) {
    Incumbent old(c,n);old.small_=small;QwenPlanCursor cursor;std::uint64_t pos=0;
    for(;;) {
        std::optional<Incumbent::Block> a;std::optional<QwenPlanUnit> b;
        std::string ae,be;
        try {a=old.next();} catch(const std::logic_error& e) {ae=e.what();}
        try {b=qwen_tiled_next(c,cursor,pos,n,small);} catch(const std::logic_error& e) {be=e.what();}
        require(ae==be,"missing owner/error differs");
        if(!ae.empty()) break;
        require(bool(a)==bool(b),"block/serial decision differs");
        require(cursor==QwenPlanCursor{old.phase_,old.tail_,old.balanced_},"continuation cursor differs");
        if(!a) break;
        require(b->begin==pos && b->end==pos+a->rows && b->capture==a->capture && b->final==a->final,"unit differs");
        const auto* backend=b->t_owner?old.tiled_.blocks.get():old.tiled_.tails[b->tail].backend.get();
        require(backend==a->backend,"owner/profile selection differs");
        ++units;pos=b->end;old.pos_=pos;
        if(pos==n) break;
    }
    ++shapes;
}
}
int main() {
try {
    for(const std::uint32_t t:{1024U,2048U,4096U,8192U})
    for(unsigned mask=0;mask<16;++mask)
    for(const std::uint32_t sched:{0U,4U,12U,16U,20U})
    for(bool start:{false,true})
    for(const auto policy:{QwenPlanOutput::state_only,QwenPlanOutput::terminal,QwenPlanOutput::all_rows}) {
        QwenTiledPlanConfig c;c.rows=t;c.block={t,policy};c.schedule=sched;c.tiled_start=start;
        unsigned bit=0;for(const auto r:{1024U,256U,64U,16U}) {if(mask&(1U<<bit)) {c.tails.push_back({r,policy});c.ladder.push_back(r);}++bit;}
        // Exhaust all residues through two T boundaries and cadence edges 4T/8T/16T.
        for(std::uint64_t n=1;n<=2U*t+1U;++n) compare(c,n);
        for(std::uint64_t multiple=3;multiple<=32;++multiple) for(int delta=-1;delta<=1;++delta) {
            compare(c,static_cast<std::uint64_t>(static_cast<std::int64_t>(multiple*t)+delta));
            for(const auto& tail:c.tails)
                compare(c,static_cast<std::uint64_t>(static_cast<std::int64_t>(multiple*t+tail.rows)+delta));
        }
        compare(c,1280);compare(c,3U*t+17U,true);
        auto mixed=c;
        for(std::size_t k=0;k<mixed.tails.size();++k) mixed.tails[k].policy=static_cast<QwenPlanOutput>(k%3U);
        compare(mixed,3U*t+1280U);mixed.ladder.clear();compare(mixed,3U*t+1280U);
        c.suppressed=true;compare(c,1280);compare(c,3U*t+17U);
        if(!c.tails.empty()) {c.suppressed=false;c.tails.erase(c.tails.begin());compare(c,3U*t+1280U);}
    }
    QwenTiledPlanConfig c;c.rows=2048;c.block={2048,QwenPlanOutput::terminal};c.schedule=12;
    c.ladder={1024,256,64,16};for(const auto r:c.ladder)c.tails.push_back({r,QwenPlanOutput::terminal});
    QwenPlanCursor cursor;const auto merged=qwen_tiled_next(c,cursor,0,1280);
    require(merged && merged->merged && merged->end==1280,"1280 regression");
    require(qwen_full_t_certificate(c,1280,1024).empty(),"internal merged checkpoint");
    const auto proof=qwen_full_t_certificate(c,8192,4096,&cursor);
    require(!proof.empty(),"full-T certificate");
    const auto suffix=qwen_tiled_next(c,cursor,4096,8192);
    require(suffix && !suffix->heads && suffix->begin==4096 && suffix->end==6144,"whole prompt continuation");
    require(proof==qwen_full_t_certificate(c,32768,4096),"invariant full-T family");
    c.schedule=20;require(qwen_full_t_certificate(c,5120,4096).empty(),"different n/balanced producer refusal");
    c.schedule=12;c.profile_pair=17;
    require(proof!=qwen_full_t_certificate(c,8192,4096),"profile certificate");
    c.profile_pair=0;c.block.execution=5;
    require(proof!=qwen_full_t_certificate(c,8192,4096),"producer identity certificate");
    c.block.execution=0;c.generations_per_row[0]=1;
    require(proof!=qwen_full_t_certificate(c,8192,4096),"generation certificate");
    c.generations_per_row[0]=0;c.priming=255;require(proof!=qwen_full_t_certificate(c,8192,4096),"priming certificate");
    for(bool complete:{false,true}) for(bool pending:{false,true}) for(bool certified:{false,true}) {
        require(qwen_capture_boundary(2048,8192,2048,complete,pending,certified)==(complete&&!pending&&certified),"capture readiness");
        require(!qwen_capture_boundary(8192,8192,2048,complete,pending,certified),"no final capture");
        require(!qwen_capture_boundary(1280,8192,2048,complete,pending,certified),"no nongrid capture");
    }
    // Suffix serial/window decisions, including odd-first and final handoff.
    for(std::uint64_t n=1;n<65;++n) for(std::uint64_t p=0;p<n;++p) for(bool windows:{false,true}) {
        const auto u=qwen_suffix_unit(p,n,windows);
        std::uint32_t r=1;
        if(windows && (n-p)%2U==0) {r=8;while(r>n-p) r/=2U;}
        require(u.begin==p && u.end==p+r && u.final==(p+r==n),"suffix serial/window plan");
    }
    for(const auto t:{2048U,8192U}) for(const auto n:{8192U,32768U}) {
        c.rows=t;c.block.rows=t;c.priming=255;
        const auto every=std::max(4096U,t);
        unsigned internal=0,selected=0,captures=0;
        QwenPlanCursor walk;std::uint64_t p=0;
        while(const auto u=qwen_tiled_next(c,walk,p,n)) {
            p=u->end;
            if(p<n && !qwen_full_t_certificate(c,n,p).empty()) {
                ++internal;
                if(p%every==0 || p+t>n-1U) {++selected;if(!u->async_priming)++captures;}
            }
            if(p==n) break;
        }
        require(captures==0 && internal==n/t-1U,"default pipeline capture count");
        require(selected==(t==2048U?(n==8192U?2U:8U):(n==8192U?0U:3U)),"default cadence count");
    }
    std::cout<<"planner passed: "<<shapes<<" shapes, "<<units<<" units; certificates and capture eligibility passed\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
