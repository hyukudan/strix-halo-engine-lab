#pragma once
#include <algorithm>
#include <array>
#include <future>
#include <stdexcept>
#include <vector>
namespace strix::glm::rtxpf::producer {
// File ordinal, not device parity, owns a host slot. A source may be overwritten
// only after its last H2D completion fence. All futures drain on error/destruction.
class LayerReadAhead {
    std::vector<unsigned> layers_;
    std::array<std::future<void>,2> reads_;
    std::array<std::size_t,2> pending_{};
    std::array<bool,2> acquired_{};
    std::size_t ordinal(unsigned layer)const {
        auto i=std::lower_bound(layers_.begin(),layers_.end(),layer);
        if(i==layers_.end()||*i!=layer)throw std::runtime_error("read-ahead layer");
        return std::size_t(i-layers_.begin());
    }
public:
    static constexpr unsigned depth=2;
    explicit LayerReadAhead(std::vector<unsigned> layers):layers_(std::move(layers)){
        if(!std::is_sorted(layers_.begin(),layers_.end())||std::adjacent_find(layers_.begin(),layers_.end())!=layers_.end())
            throw std::runtime_error("read-ahead order");
    }
    ~LayerReadAhead(){for(auto& r:reads_)if(r.valid())r.wait();}
    void drain(){std::exception_ptr error;for(auto& r:reads_)if(r.valid())try{r.get();}catch(...){if(!error)error=std::current_exception();}if(error)std::rethrow_exception(error);}
    template<class Load> void reset(Load load){drain();acquired_.fill(false);
        for(unsigned s=0;s<std::min<std::size_t>(depth,layers_.size());++s){pending_[s]=s;reads_[s]=std::async(std::launch::async,[=,this]{load(s,layers_[s]);});}}
    unsigned acquire(unsigned layer){auto i=ordinal(layer);auto s=unsigned(i%depth);
        if(pending_[s]!=i||!reads_[s].valid()||acquired_[s])throw std::runtime_error("read-ahead acquire");
        reads_[s].get();acquired_[s]=true;return s;}
    template<class Load,class Fence> void release(unsigned layer,Load load,Fence fence){auto i=ordinal(layer);auto s=unsigned(i%depth);auto next=i+depth;
        if(pending_[s]!=i||!acquired_[s])throw std::runtime_error("read-ahead release");acquired_[s]=false;pending_[s]=next;
        if(next<layers_.size())reads_[s]=std::async(std::launch::async,[=,this]{fence(s);load(s,layers_[next]);});}
};
}
