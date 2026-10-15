#include "strix/wave_router_contract.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace {
bool better(float a, std::uint32_t ai, float b, std::uint32_t bi) { return a > b || (a == b && ai < bi); }
struct Result { std::vector<std::uint32_t> indices; std::vector<float> values; };
void check_slots(const Result& r, const std::vector<float>& p) {
    const auto count = static_cast<std::uint32_t>(p.size());
    const auto valid = static_cast<std::size_t>(std::count_if(p.begin(), p.end(), [](float x) { return x == x; }));
    const auto occupied = std::min(valid, r.indices.size());
    std::set<std::uint32_t> seen;
    for (std::size_t s = 0; s < r.indices.size(); ++s) {
        const auto i = r.indices[s];
        if (s < occupied) {
            if (i >= count || p[i] != p[i] || !seen.insert(i).second)
                throw std::runtime_error("duplicate or hole in selected slots");
            if (std::bit_cast<std::uint32_t>(r.values[s]) != std::bit_cast<std::uint32_t>(p[i]))
                throw std::runtime_error("selected value bits changed");
        } else if (i != count || r.values[s] != -INFINITY) {
            throw std::runtime_error("missing sentinel after valid candidates");
        }
    }
}
Result insertion(const std::vector<float>& p, std::uint32_t k) {
    const auto count = static_cast<std::uint32_t>(p.size());
    Result r{std::vector<std::uint32_t>(k, count), std::vector<float>(k, -INFINITY)};
    for (std::uint32_t e = 0; e < count; ++e) for (std::uint32_t s = 0; s < k; ++s) {
        if (!better(p[e], e, r.values[s], r.indices[s])) continue;
        for (auto move = k - 1U; move > s; --move) { r.values[move] = r.values[move-1]; r.indices[move] = r.indices[move-1]; }
        r.values[s] = p[e]; r.indices[s] = e; break;
    }
    return r;
}
// Simulates synchronous XOR shuffles from snapshots, not in-place host updates.
Result wave(const std::vector<float>& p, std::uint32_t k) {
    const auto count = static_cast<std::uint32_t>(p.size());
    std::array<std::uint32_t,32> available{};
    for (std::uint32_t lane = 0; lane < 32; ++lane)
        for (std::uint32_t i = 0; i < 16; ++i) {
            const auto e = lane + 32U*i;
            if (e < count && p[e] == p[e]) available[lane] |= 1U<<i;
        }
    Result r;
    for (std::uint32_t s = 0; s < k; ++s) {
        std::array<bool,32> has{}; std::array<float,32> values{}; std::array<std::uint32_t,32> indices{};
        values.fill(-INFINITY); indices.fill(count);
        for (std::uint32_t lane = 0; lane < 32; ++lane)
            for (std::uint32_t i = 0; i < 16; ++i) if ((available[lane] & (1U<<i)) != 0) {
                const auto e = lane + 32U*i;
                if (!has[lane] || better(p[e],e,values[lane],indices[lane])) {
                    has[lane]=true; values[lane]=p[e]; indices[lane]=e;
                }
            }
        for (std::uint32_t mask = 16; mask > 0; mask >>= 1U) {
            const auto old_has=has; const auto old_values=values; const auto old_indices=indices;
            for (std::uint32_t lane = 0; lane < 32; ++lane) {
                const auto other=lane^mask;
                if (old_has[other] && (!old_has[lane] || better(old_values[other],old_indices[other],old_values[lane],old_indices[lane]))) {
                    has[lane]=true; values[lane]=old_values[other]; indices[lane]=old_indices[other];
                }
            }
        }
        for (std::uint32_t lane = 1; lane < 32; ++lane)
            if (has[lane]!=has[0] || indices[lane]!=indices[0] ||
                std::bit_cast<std::uint32_t>(values[lane])!=std::bit_cast<std::uint32_t>(values[0]))
                throw std::runtime_error("wave result not lane-uniform");
        r.indices.push_back(has[0]?indices[0]:count); r.values.push_back(has[0]?values[0]:-INFINITY);
        if (has[0]) available[indices[0]%32U] &= ~(1U<<(indices[0]/32U));
    }
    return r;
}
void epilogue(Result& r, bool normalize) {
    float sum=0.0F; for (float x:r.values) sum+=x;
    if (!normalize) sum=1.0F;
    for (float& x:r.values) x/=sum;
}
template<class F> void rejects(F f) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid wave contract admitted");
}
}

int main() try {
    std::size_t cases=0; std::uint32_t state=0x1743a251U;
    for (const auto count:{1U,8U,31U,33U,64U,255U,257U,500U,511U,512U})
        for (unsigned pattern=0;pattern<13;++pattern) {
            std::vector<float> p(count);
            for (std::uint32_t e=0;e<count;++e) {
                state^=state<<13U;state^=state>>17U;state^=state<<5U;
                switch(pattern) {
                case 0:p[e]=0.25F;break;
                case 1:p[e]=e%2U?-0.0F:0.0F;break;
                case 2:p[e]=std::bit_cast<float>(0x7fc00001U+(e%127U));break;
                case 3:p[e]=e==0?1.0F:std::numeric_limits<float>::quiet_NaN();break;
                case 4:p[e]=e%3U==0?std::numeric_limits<float>::quiet_NaN():-INFINITY;break;
                case 5:p[e]=e%2U?INFINITY:-INFINITY;break;
                case 6:p[e]=static_cast<float>(e);break;
                case 7:p[e]=static_cast<float>(count-e);break;
                case 8:p[e]=std::bit_cast<float>((e%2U?0x80000000U:0U)|(e+1U));break;
                case 9:p[e]=e%32U==0?1.0F:0.0F;break;
                case 10:p[e]=e<64?1.0F:0.0F;break;
                case 11:p[e]=e<3?0.5F:0.0F;break;
                default:p[e]=static_cast<float>(static_cast<int>(state%13U)-6);break;
                }
            }
            for (std::uint32_t k=1;k<=std::min(count,32U);++k) for (bool normalize:{false,true}) {
                auto reference=insertion(p,k),candidate=wave(p,k);check_slots(candidate,p);
                if(reference.indices!=candidate.indices)throw std::runtime_error("wave index order differs");
                epilogue(reference,normalize);epilogue(candidate,normalize);
                for(std::uint32_t s=0;s<k;++s)
                    if(std::bit_cast<std::uint32_t>(reference.values[s])!=std::bit_cast<std::uint32_t>(candidate.values[s]))
                        throw std::runtime_error("wave weight bits differ including NaN payloads");
                ++cases;
            }
        }
    const std::vector<float> p{1.0F,0.5F,0.25F};
    auto duplicate=wave(p,3);duplicate.indices[1]=duplicate.indices[0];
    bool caught=false;try{check_slots(duplicate,p);}catch(const std::runtime_error&){caught=true;}
    if(!caught)throw std::runtime_error("duplicate mutant survived");
    auto hole=wave(p,3);hole.indices[0]=3;hole.values[0]=-INFINITY;
    caught=false;try{check_slots(hole,p);}catch(const std::runtime_error&){caught=true;}
    if(!caught)throw std::runtime_error("hole mutant survived");
    strix::validate_wave_router_selection_contract(512,32,true,false);
    rejects([]{strix::validate_wave_router_selection_contract(512,33,true,false);});
    rejects([]{strix::validate_wave_router_selection_contract(512,10,false,false);});
    rejects([]{strix::validate_wave_router_selection_contract(512,10,true,true);});
    rejects([]{strix::validate_wave_router_selection_contract(513,10,true,false);});
    rejects([]{strix::validate_wave_router_selection_contract(8,10,true,false);});
    strix::gguf::FileInfo info;strix::validate_wave_router_model_inputs(info);
    strix::gguf::MetadataEntry group;group.key="qwen.expert_group_count";info.metadata.push_back(group);
    rejects([&]{strix::validate_wave_router_model_inputs(info);});
    info.metadata.front().key="qwen.expert_group_used_count";
    rejects([&]{strix::validate_wave_router_model_inputs(info);});
    info.metadata.clear();strix::gguf::TensorInfo tensor;tensor.name="blk.0.exp_probs_b.bias";info.tensors.push_back(tensor);
    rejects([&]{strix::validate_wave_router_model_inputs(info);});
    strix::ModelManifest manifest;strix::ManifestTensor bias;bias.name=tensor.name;manifest.tensors.push_back(bias);
    rejects([&]{strix::validate_wave_router_manifest(manifest);});
    std::cout<<cases<<" wave/insertion bit-exact cases, slot mutants and admission negatives passed\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
