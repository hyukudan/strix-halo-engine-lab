#pragma once

#include "strix/known_prefill_limits.hpp"
#include "strix/execution_memory.hpp"

#include <array>
#include <vector>

namespace strix {
// Pure inspection: never probes a prompt window or advances a session.
struct BatchNestedRecipe {
    std::vector<std::uint32_t> units;
    std::uint64_t suffix=0;
    const char* decline=nullptr;
    explicit operator bool() const noexcept {return !decline;}
};
inline BatchNestedRecipe batch_nested_recipe(std::uint64_t remaining,std::uint64_t limit,std::uint32_t t,
    std::span<const std::uint32_t> ladder,std::span<const std::uint32_t> owners,bool qualified,bool terminal_heads=true) {
    BatchNestedRecipe out;
    if(!limit || !remaining || remaining>limit) {out.decline="rows";return out;}
    if(!qualified) {out.decline="route";return out;}
    constexpr std::array<std::uint32_t,4> supported{1024,256,64,16};
    // Smaller T and custom ladders need their own row-class proof.
    if(t!=8192 || remaining>=t || !std::equal(ladder.begin(),ladder.end(),supported.begin(),supported.end())) {
        out.decline="row_class";return out;
    }
    std::size_t next=0;
    for(const auto rows:owners) {
        while(next<supported.size() && supported[next]!=rows) ++next;
        if(next==supported.size()) {out.decline="row_class";return out;}
        ++next;
    }
    auto rest=remaining-(terminal_heads?0U:1U);
    for(const auto rows:ladder) {
        while(rest>=rows) {
            if(std::find(owners.begin(),owners.end(),rows)==owners.end()) {out.decline="missing_owner";return out;}
            out.units.push_back(rows);rest-=rows;
        }
    }
    out.suffix=rest+(terminal_heads?0U:1U);
    return out;
}
inline bool batch_storage_overlap(const ExecutionMemoryBinding& a,const ExecutionMemoryBinding& b) noexcept {
    if(!a.arena_base || !b.arena_base || !a.arena_bytes || !b.arena_bytes) return false;
    return a.arena_base<=b.arena_base?b.arena_base-a.arena_base<a.arena_bytes:a.arena_base-b.arena_base<b.arena_bytes;
}
} // namespace strix
