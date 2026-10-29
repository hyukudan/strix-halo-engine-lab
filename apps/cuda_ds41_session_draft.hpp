#pragma once
#include "cuda_ds41_session_cache.hpp"
// CPU representation of DSpark's physical rings and cursor interpretation.
namespace strix::ds41::session {
struct DraftState {
    std::uint64_t start=0,next=0;
    std::uint32_t core=129280;
    std::vector<std::uint32_t> extra;
    std::array<Bytes,3> rings;
    void validate()const {
        require(start<=next&&next<=0xffffffffULL,"invalid drafter cursor");
        require(core<=129280&&(core!=0||!extra.empty()),"invalid drafter vocabulary");
        std::vector<bool> seen(129280);
        for(auto id:extra){require(id>=core&&id<129280&&!seen[id],"invalid extra vocabulary");seen[id]=true;}
        for(auto& ring:rings)require(ring.size()==128*528,"invalid drafter ring geometry");
    }
    Bytes encode()const {
        validate();Bytes b;text(b,"rtx-dspark-v1");u64(b,start);u64(b,next);u64(b,core);u64(b,extra.size());
        for(auto id:extra)u64(b,id);
        for(auto& ring:rings)blob(b,ring);
        return b;
    }
    static DraftState decode(std::span<const std::uint8_t> b) {
        Reader r{b};require(r.string()=="rtx-dspark-v1","unsupported drafter schema");DraftState s;
        s.start=r.integer();s.next=r.integer();const auto core=r.integer();require(core<=129280,"invalid draft core");
        s.core=static_cast<std::uint32_t>(core);auto n=r.integer();require(n<=129280,"invalid draft extras size");
        for(std::uint64_t i=0;i<n;++i){auto id=r.integer();require(id<129280,"invalid draft id");s.extra.push_back(static_cast<std::uint32_t>(id));}
        for(auto& ring:s.rings){auto p=r.data();ring.assign(p.begin(),p.end());}
        r.end();s.validate();return s;
    }
};
}

#include "strix/deepseek41_attention_state.hpp"
namespace strix::ds41::session {
inline strix::deepseek41::AttentionState validate_attention(const Snapshot& s,
        const strix::deepseek41::AttentionState& layout,std::uint32_t capacity) {
    const auto a=strix::deepseek41::deserialize_attention_state(s.attention);
    require(a.position==s.frontier.consumed&&a.next_token==s.frontier.pending&&a.tokens.size()==a.position&&
            a.position>0&&a.position<capacity&&a.layers.size()==40&&layout.layers.size()==40,
            "session attention/frontier mismatch or capacity exceeded");
    for(auto id:a.tokens)require(id<129280,"invalid attention token");
    for(std::size_t l=0;l<a.layers.size();++l) {
        const auto& x=a.layers[l];const auto& expected=layout.layers[l];
        require(x.ring.size()==136*528&&x.ratio==expected.ratio&&x.owns_kv==expected.owns_kv&&x.owns_index==expected.owns_index,
                "session attention layer layout mismatch");
        require(!x.owns_kv||(x.ratio==1||x.ratio==2),"invalid compression ratio");
        const auto count=x.ratio?a.position/x.ratio:0;
        require(x.kv.size()==(x.owns_kv?std::uint64_t(count)*288:0)&&
                x.idx_keys.size()==(x.owns_kv?std::uint64_t(count)*68:0)&&
                x.carry.size()==(x.owns_kv&&x.ratio==2?1024U:0U),"session attention cache geometry mismatch");
        require(!x.carry_valid||(x.owns_kv&&x.ratio==2),"invalid carry role");
        // Imported states may omit last-row top-k; the next decode step
        // recomputes it. A generated checkpoint always includes known selections.
        require(x.topk.size()<=512&&(!x.owns_index?x.topk.empty():true),"invalid top-k shape");
        for(auto k:x.topk)require(k<count,"session top-k outside compressed frontier");
    }
    return a;
}
}
