#include "strix/batch_nested_prefill.hpp"
#include "../apps/qwen_tiled_prompt_plan.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
std::vector<unsigned> classes(std::span<const std::uint32_t> units,std::uint64_t suffix) {
    std::vector<unsigned> out;
    for(const auto rows:units) out.insert(out.end(),rows,rows%128U?rows:128U);
    out.insert(out.end(),static_cast<std::size_t>(suffix),1U);
    return out;
}
void recipes() {
    for(const auto t:{2048U,8192U}) for(std::uint64_t remaining=1;remaining<=8193;++remaining)
        for(bool worker:{false,true}) for(const auto limit:{0U,2048U,16384U}) {
            const bool bound=strix::decode_cli::qwen_batch_nest_possible(worker,limit,t,remaining);
            require(bound==(worker && limit && t==8192 && remaining<t && remaining<=limit),"nest-possible row-class mapping");
            const std::vector<std::uint32_t> standard{1024,256,64,16};
            if(worker && strix::batch_nested_recipe(remaining,limit,t,standard,standard,true))
                require(bound,"capture predicate must cover every qualified nesting recipe");
        }
    const std::vector<std::uint32_t> ladder{1024,256,64,16};
    for(const bool heads:{false,true}) for(std::uint64_t n=1;n<=2048;++n) {
        const auto forced=strix::batch_nested_recipe(n,2048,8192,ladder,ladder,true,heads);
        require(bool(forced),"supported length declined");
        auto rest=n-(heads?0U:1U);
        std::vector<std::uint32_t> expanded,ordinary;
        const auto merged=strix::known_prefill_merged_tail_rows(rest,8192,ladder);
        if(merged) ordinary.push_back(static_cast<std::uint32_t>(merged));
        // Expand the ordinary recipe's merged prefix back to its original units.
        auto prefix=merged;
        for(const auto rows:ladder) while(prefix>=rows) {expanded.push_back(rows);prefix-=rows;}
        require(prefix==0,"merged prefix has an off-grid remainder");
        rest-=merged;
        for(const auto rows:ladder) while(rest>=rows) {ordinary.push_back(rows);expanded.push_back(rows);rest-=rows;}
        require(expanded==forced.units && forced.suffix==rest+(heads?0U:1U),"forced ladder changed units or terminal suffix");
        require(classes(ordinary,forced.suffix)==classes(forced.units,forced.suffix),"forced ladder changed per-row class");
    }
    for(const auto n:{128U,384U,1152U,1192U,1280U,1536U,2048U}) {
        const auto recipe=strix::batch_nested_recipe(n,2048,8192,ladder,ladder,true);
        std::uint64_t rows=recipe.suffix;for(const auto r:recipe.units) rows+=r;
        require(rows==n,"example lost rows");
    }
    const std::vector<std::vector<std::uint32_t>> examples{{64,64},{256,64,64},{1024,64,64},
        {1024,64,64,16,16},{1024,256},{1024,256,256},{1024,1024}};
    std::size_t example=0;
    for(const auto n:{128U,384U,1152U,1192U,1280U,1536U,2048U}) {
        const auto recipe=strix::batch_nested_recipe(n,2048,8192,ladder,ladder,true);
        require(recipe.units==examples[example++] && recipe.suffix==(n==1192U?8U:0U),"explicit ladder example changed");
    }
    for(std::uint32_t t=16;t<8192;++t)
        require(!strix::batch_nested_recipe(128,2048,t,ladder,ladder,true),"unqualified T admitted");
    const auto reason=[&](std::uint64_t n,std::uint64_t limit,std::uint32_t t,const std::vector<std::uint32_t>& tails,
                          const std::vector<std::uint32_t>& owners,bool qualified) {
        const auto r=strix::batch_nested_recipe(n,limit,t,tails,owners,qualified);
        return r.decline?std::string(r.decline):std::string{};
    };
    require(reason(2049,2048,8192,ladder,ladder,true)=="rows","limit ignored");
    require(reason(1,0,8192,ladder,ladder,true)=="rows","off ignored");
    require(reason(128,2048,128,ladder,ladder,true)=="row_class","small T admitted");
    require(reason(128,2048,4096,ladder,ladder,true)=="row_class","T below 8192 admitted");
    require(reason(1192,2048,8192,ladder,{1024,256,16},true)=="missing_owner","missing 64 owner admitted");
    require(reason(1280,2048,8192,ladder,{1024,64,16},true)=="missing_owner","pruned merged owner admitted");
    require(reason(1192,2048,8192,{1024,256,32,16},ladder,true)=="row_class","custom ladder admitted");
    require(reason(1192,2048,8192,ladder,{1024,512,256,64,16},true)=="row_class","custom owner admitted");
    require(reason(1192,2048,8192,ladder,{64,1024,256,16},true)=="row_class","reordered owners admitted");
    require(reason(1192,2048,8192,ladder,{1024,256,64,64,16},true)=="row_class","duplicate owner admitted");
    require(reason(1192,2048,8192,{},ladder,true)=="row_class","missing ladder admitted");
    require(reason(1192,2048,8192,ladder,ladder,false)=="route","unqualified route admitted");
    require(reason(16,2048,8192,ladder,{16},true).empty(),"unused owner required");
}
void aliases() {
    strix::ExecutionMemoryBinding arena{},d1{},workspace{},candidate{};
    arena.arena_base=4096;arena.arena_bytes=1024;
    d1.arena_base=8192;d1.arena_bytes=8;
    workspace.arena_base=16384;workspace.arena_bytes=256;
    for(const auto binding:{arena,d1,workspace}) {
        candidate=binding;candidate.execution_identity=99;
        require(strix::batch_storage_overlap(binding,candidate),"distinct owner identity hid physical alias");
        candidate.arena_base=binding.arena_base+binding.arena_bytes-1U;candidate.arena_bytes=2;
        require(strix::batch_storage_overlap(binding,candidate) && strix::batch_storage_overlap(candidate,binding),"partial alias missed");
        ++candidate.arena_base;
        require(!strix::batch_storage_overlap(binding,candidate),"adjacent storage aliases");
    }
    require(!strix::batch_storage_overlap(arena,{}),"empty storage aliases");
}
}
int main() {
    try {recipes();aliases();} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
    std::cout<<"batch nested recipe tests passed\n";
}
