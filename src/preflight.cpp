#include "strix/preflight.hpp"
#include "strix/build_info.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#if defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace strix {
namespace {
using Status = LinuxUmaReadStatus;
constexpr std::uint64_t unit = 2U*1024U*1024U;
std::string_view trim(std::string_view s) {
    const auto a=s.find_first_not_of(" \t\r\n"), b=s.find_last_not_of(" \t\r\n");
    return a==s.npos ? std::string_view{} : s.substr(a,b-a+1U);
}
PreflightNumber integer(std::string_view s) {
    std::uint64_t n=0; const auto r=std::from_chars(s.data(),s.data()+s.size(),n);
    if(s.empty() || r.ec!=std::errc{} || r.ptr!=s.data()+s.size()) return {};
    return n;
}
bool add(std::uint64_t& a,std::uint64_t b) { if(b>UINT64_MAX-a) return false; a+=b; return true; }
std::uint64_t now() { return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count()); }
const char* status(Status s) {
    switch(s) {
    case Status::read:return "read"; case Status::malformed:return "malformed";
    case Status::oversized:return "truncated"; case Status::unsupported:return "unsupported";
    default:return "unavailable";
    }
}
std::string number(PreflightNumber n) { return n?std::to_string(*n):"null"; }
// Real sysfs symlinks are trusted. Fixture symlinks must remain within the root.
std::filesystem::path rooted(const PreflightPaths& p,const std::filesystem::path& relative) {
    const auto path=p.root/relative;
    if(p.root=="/") return path;
    std::error_code ec;
    const auto root=std::filesystem::weakly_canonical(p.root,ec); if(ec) return {};
    const auto resolved=std::filesystem::weakly_canonical(path,ec); if(ec) return {};
    const auto rel=resolved.lexically_relative(root);
    if(rel.empty() || *rel.begin()==".." || rel.is_absolute()) return {};
    return resolved;
}
auto read(const PreflightPaths& p,const std::filesystem::path& path,std::size_t cap) {
    return read_linux_uma_text(rooted(p,path),cap);
}
// Enumeration counts all entries against the cap, even entries not selected.
bool entries(const PreflightPaths& p,const std::filesystem::path& path,std::size_t cap,
    const std::function<void(const std::filesystem::directory_entry&)>& visit,Status& state) {
    std::error_code ec; std::filesystem::directory_iterator it(rooted(p,path),ec),end;
    state=Status::unavailable; if(ec) return false;
    std::size_t n=0;
    for(;it!=end;it.increment(ec)) {
        if(ec) return false;
        if(++n>cap) {state=Status::oversized;return false;}
        visit(*it);
    }
    if(ec) return false;
    state=Status::read;return true;
}
const char* warning(PreflightWarning w) {
    switch(w) {
    case PreflightWarning::gtt_plan_pressure:return "gtt-plan-pressure";
    case PreflightWarning::host_plan_pressure:return "host-plan-pressure";
    case PreflightWarning::preexisting_gtt_use:return "preexisting-gtt-use";
    case PreflightWarning::allocation_owner_unresolved:return "allocation-owner-unresolved";
    case PreflightWarning::firmware_reserve_review:return "firmware-reserve-review";
    case PreflightWarning::contiguous_supply_risk:return "contiguous-supply-risk";
    case PreflightWarning::startup_compaction_pressure:return "startup-compaction-pressure";
    case PreflightWarning::pin_limit_risk:return "pin-limit-risk";
    default:return "observations-unavailable";
    }
}
}
LinuxUmaObservation parse_preflight_meminfo(std::string_view s) {return parse_linux_uma_observation(s,{},1);}
PreflightScalar parse_preflight_scalar(std::string_view s) {
    if(s.size()>128) return {{},Status::oversized};
    if(s.empty()) return {};
    auto n=integer(trim(s));return {n,n?Status::read:Status::malformed};
}
PreflightBuddy parse_preflight_buddyinfo(std::string_view text,PreflightNumber page,const std::optional<std::vector<unsigned>>& nodes) {
    const auto bad=[](Status s){return PreflightBuddy{{{},s},{{},s}};};
    if(text.size()>65536) return bad(Status::oversized);
    if(text.empty() || !page || !*page || (*page & (*page-1U)) || *page>unit) return bad(Status::unavailable);
    unsigned target=0;for(auto size=*page;size<unit;size*=2U) ++target;
    std::istringstream in{std::string(text)};std::string line;std::uint64_t blocks=0,units=0;bool seen=false;
    while(std::getline(in,line)) {
        std::istringstream row(line);std::string node,id,zone,name,token;
        if(!(row>>node>>id>>zone>>name) || node!="Node" || zone!="zone" || id.empty() || id.back()!=',' ||
            !integer(std::string_view(id).substr(0,id.size()-1U))) return bad(Status::malformed);
        const auto node_id=integer(std::string_view(id).substr(0,id.size()-1U));
        const bool eligible=!nodes || std::find(nodes->begin(),nodes->end(),*node_id)!=nodes->end();
        unsigned order=0;
        while(row>>token) {
            auto count=integer(token);if(!count || order>=64) return bad(Status::malformed);
            if(eligible && order>=target) {
                const auto shift=order-target;
                if(*count>(UINT64_MAX>>shift) || !add(blocks,*count) || !add(units,*count<<shift)) return bad(Status::malformed);
            }
            ++order;
        }
        if(!order) return bad(Status::malformed);
        seen=true;
    }
    if(!seen) return bad(Status::malformed);
    return {{blocks,Status::read},{units,Status::read}};
}
PreflightVmstat parse_preflight_vmstat(std::string_view text) {
    PreflightVmstat out;
    if(text.size()>65536) {out.stall.status=out.success.status=out.fail.status=Status::oversized;return out;}
    std::istringstream in{std::string(text)};std::string line;
    while(std::getline(in,line)) {
        std::istringstream row(line);std::string key,value,extra;row>>key;
        auto* field=key=="compact_stall"?&out.stall:key=="compact_success"?&out.success:key=="compact_fail"?&out.fail:nullptr;
        if(!field) continue;
        if(field->status!=Status::unavailable || !(row>>value) || (row>>extra)) { *field={{},Status::malformed};continue; }
        auto n=integer(value);*field={n,n?Status::read:Status::malformed};
    }
    return out;
}
PreflightThp parse_preflight_thp(std::string_view text) {
    if(text.size()>128) return PreflightThp::unavailable;
    std::istringstream in{std::string(text)};std::string token;PreflightThp out=PreflightThp::unavailable;
    unsigned active=0,seen=0;
    while(in>>token) {
        bool selected=token.size()>2 && token.front()=='[' && token.back()==']';
        if(selected) token=token.substr(1,token.size()-2);
        const unsigned bit=token=="always"?1U:token=="madvise"?2U:token=="never"?4U:0U;
        if(!bit || (seen&bit)) return PreflightThp::unavailable;
        seen|=bit;
        if(selected) {++active;out=bit==1?PreflightThp::always:bit==2?PreflightThp::madvise:PreflightThp::never;}
    }
    return active==1?out:PreflightThp::unavailable;
}
PreflightProcessFacts preflight_process_facts() {
    PreflightProcessFacts f;
#if defined(__linux__)
    auto page=::sysconf(_SC_PAGESIZE);if(page>0) f.page_size=static_cast<std::uint64_t>(page);
    f.self_pid=static_cast<std::uint64_t>(::getpid());
    rlimit lim{};
    if(::getrlimit(RLIMIT_MEMLOCK,&lim)==0) {
        f.memlock_soft.infinite=lim.rlim_cur==RLIM_INFINITY;f.memlock_hard.infinite=lim.rlim_max==RLIM_INFINITY;
        if(!f.memlock_soft.infinite) f.memlock_soft.bytes=lim.rlim_cur;
        if(!f.memlock_hard.infinite) f.memlock_hard.bytes=lim.rlim_max;
    }
#endif
    return f;
}
PreflightObservation collect_preflight(const PreflightPaths& p,const PreflightFactProvider& facts) noexcept {
    PreflightObservation o;
    try {
        o.begin_ns=now();o.process=facts();
#if defined(__linux__)
        o.process_identity=static_cast<std::uint64_t>(::getpid());
#else
        if(p.root=="/") {
            o.mem_total.status=o.mem_available.status=o.gtt_total.status=o.gtt_used.status=
                o.vram_total.status=o.vram_used.status=o.other_clients.status=o.buddy.blocks.status=
                o.buddy.units_2m.status=o.compact.stall.status=o.compact.success.status=o.compact.fail.status=Status::unsupported;
            o.end_ns=now();return o;
        }
#endif
        auto mem=read(p,"proc/meminfo",65536);auto m=parse_preflight_meminfo(mem.first);
        const auto ms=mem.second==Status::read?m.meminfo_status:mem.second;
        o.mem_total={m.memory.mem_total_bytes,ms};o.mem_available={m.memory.mem_available_bytes,ms};
        auto buddy=read(p,"proc/buddyinfo",65536);o.buddy=parse_preflight_buddyinfo(buddy.first,o.process.page_size,o.process.allowed_nodes);
        if(buddy.second!=Status::read) o.buddy={{{},buddy.second},{{},buddy.second}};
        auto vm=read(p,"proc/vmstat",65536);o.compact=parse_preflight_vmstat(vm.first);
        if(vm.second!=Status::read) o.compact.stall.status=o.compact.success.status=o.compact.fail.status=vm.second;
        o.thp=parse_preflight_thp(read(p,"sys/kernel/mm/transparent_hugepage/enabled",128).first);
        std::uint64_t clients=0;bool self_known=o.process.self_pid.has_value(), client_error=false;
        if(entries(p,"sys/class/kfd/kfd/proc",4096,[&](const auto& e){
            const auto pid=integer(e.path().filename().string());std::error_code ec;
            // KFD process entries are plain directories: a symlink could escape a fixture root.
            const auto type=e.symlink_status(ec).type();
            if(pid && !ec && type==std::filesystem::file_type::symlink) client_error=true;
            else if(pid && !ec && type==std::filesystem::file_type::directory && pid!=o.process.self_pid) ++clients;
            if(ec && ec!=std::errc::no_such_file_or_directory) client_error=true;
        },o.other_clients.status) && self_known && !client_error) o.other_clients.value=clients;
        if(client_error || !self_known) o.other_clients.status=Status::unavailable;
        std::vector<unsigned> cards;Status topology;bool topology_error=false;
        bool complete=entries(p,"sys/class/drm",256,[&](const auto& e){
            auto name=e.path().filename().string();if(!name.starts_with("card")) return;
            auto id=integer(std::string_view(name).substr(4));if(!id || *id>UINT32_MAX) return;
            const auto driver=rooted(p,e.path().lexically_relative(p.root)/"device/driver");
            std::error_code ec;auto resolved=std::filesystem::canonical(driver,ec);
            if(ec) topology_error=true;
            if(!ec && resolved.filename()=="amdgpu") cards.push_back(static_cast<unsigned>(*id));
        },topology);
        if(topology_error) {complete=false;topology=Status::unavailable;}
        if(complete) o.amdgpu_cards=cards.size();
        std::optional<unsigned> selected;
        if(complete) {
            if(o.process.drm_card && std::find(cards.begin(),cards.end(),*o.process.drm_card)!=cards.end()) selected=o.process.drm_card;
            else if(!o.process.drm_card && cards.size()==1) selected=cards.front();
            o.device_ambiguous=!selected && cards.size()>1;
        }
        if(selected) {
            const auto base=std::filesystem::path("sys/class/drm")/("card"+std::to_string(*selected))/"device";
            const auto scalar=[&](const char* name){auto r=read(p,base/name,128);return r.second==Status::read?
                parse_preflight_scalar(r.first):PreflightScalar{{},r.second};};
            o.gtt_total=scalar("mem_info_gtt_total");o.gtt_used=scalar("mem_info_gtt_used");
            o.vram_total=scalar("mem_info_vram_total");o.vram_used=scalar("mem_info_vram_used");
            for(auto pair:{std::pair{&o.gtt_total,&o.gtt_used},std::pair{&o.vram_total,&o.vram_used}})
                if(pair.first->value && pair.second->value && *pair.second->value>*pair.first->value)
                    *pair.first=*pair.second={{},Status::malformed};
        } else if(!complete) o.gtt_total.status=o.gtt_used.status=o.vram_total.status=o.vram_used.status=topology;
        o.end_ns=now();
    } catch(...) {o=PreflightObservation{};}
    return o;
}
PreflightAssessment assess_preflight(const PreflightObservation& o,const PreflightInput& p,const PreflightObservation* prev) {
    PreflightAssessment a;a.plan=p;auto warn=[&](PreflightWarning w){a.warnings.push_back(w);};
    if(!o.mem_total.value || !o.mem_available.value || !o.gtt_total.value || !o.gtt_used.value ||
        !o.vram_total.value || !o.vram_used.value || !o.buddy.units_2m.value || !o.other_clients.value ||
        !o.compact.stall.value || !o.process.page_size || o.thp==PreflightThp::unavailable ||
        (!o.process.memlock_soft.bytes && !o.process.memlock_soft.infinite) ||
        (!o.process.memlock_hard.bytes && !o.process.memlock_hard.infinite)) warn(PreflightWarning::unavailable);
    if(o.gtt_total.value && o.gtt_used.value && *o.gtt_used.value<=*o.gtt_total.value) {
        a.gtt_free=*o.gtt_total.value-*o.gtt_used.value;
        if(p.G && *p.G>*a.gtt_free) {a.gtt_deficit=*p.G-*a.gtt_free;warn(PreflightWarning::gtt_plan_pressure);}
    }
    if(p.H && p.R && o.mem_available.value) {
        auto demand=*p.H;if(add(demand,*p.R) && demand>*o.mem_available.value) {
            a.host_deficit=demand-*o.mem_available.value;warn(PreflightWarning::host_plan_pressure);
        }
    }
    // Kernel/firmware GTT use is normal at idle; only the plan comparison above is pressure.
    if(o.gtt_used.value && *o.gtt_used.value) {
        a.advisories.push_back(PreflightWarning::preexisting_gtt_use);
        if(!o.other_clients.value || !*o.other_clients.value) a.advisories.push_back(PreflightWarning::allocation_owner_unresolved);
    }
    if(o.process.uma==true && o.vram_total.value &&
        ((a.host_deficit && *o.vram_total.value>=*a.host_deficit) || (a.gtt_deficit && *o.vram_total.value>=*a.gtt_deficit)))
        warn(PreflightWarning::firmware_reserve_review);
    bool short_supply=false;
    if(p.C) {a.needed_2m_units=*p.C/unit+(*p.C%unit!=0);
        short_supply=o.buddy.units_2m.value && *o.buddy.units_2m.value<*a.needed_2m_units;
        if(short_supply) warn(PreflightWarning::contiguous_supply_risk);
    }
    if(prev && o.process_identity && o.process_identity==prev->process_identity && prev->end_ns && o.end_ns>prev->end_ns &&
        o.compact.stall.value && prev->compact.stall.value && *o.compact.stall.value>=*prev->compact.stall.value) {
        bool reset=false;
        for(auto pair:{std::pair{o.compact.success.value,prev->compact.success.value},std::pair{o.compact.fail.value,prev->compact.fail.value}})
            if(pair.first && pair.second && *pair.first<*pair.second) reset=true;
        if(!reset) {a.stall_delta=*o.compact.stall.value-*prev->compact.stall.value;a.elapsed_ns=o.end_ns-prev->end_ns;}
    }
    if(a.stall_delta && *a.stall_delta && (short_supply || a.host_deficit)) warn(PreflightWarning::startup_compaction_pressure);
    if(p.L && o.process.memlock_soft.bytes && !o.process.memlock_soft.infinite && *p.L>*o.process.memlock_soft.bytes)
        warn(PreflightWarning::pin_limit_risk);
    return a;
}
std::string render_preflight(const PreflightObservation& o,const PreflightAssessment& a) {
    std::ostringstream s;s<<"{\"format\":\"strix-startup-event\",\"revision\":1,\"stage\":\"preflight\"";
    const auto scalar=[&](const char* key,const PreflightScalar& v){s<<",\""<<key<<"\":{\"value\":"<<number(v.value)
        <<",\"status\":\""<<(v.value?"read":v.status==Status::read?"unavailable":status(v.status))<<"\"}";};
    scalar("mem_total_bytes",o.mem_total);scalar("mem_available_bytes",o.mem_available);
    scalar("gtt_total_bytes",o.gtt_total);scalar("gtt_used_bytes",o.gtt_used);
    scalar("vram_total_bytes",o.vram_total);scalar("vram_used_bytes",o.vram_used);
    scalar("other_kfd_clients_global",o.other_clients);scalar("buddy_higher_order_blocks",o.buddy.blocks);
    scalar("buddy_equivalent_2m_units",o.buddy.units_2m);scalar("compact_stall_since_boot",o.compact.stall);
    scalar("compact_success_since_boot",o.compact.success);scalar("compact_fail_since_boot",o.compact.fail);
    s<<",\"buddy_scope\":\""<<(o.process.allowed_nodes?"allowed-nodes":"global-upper-bound")
        <<"\",\"amdgpu_card_count\":"<<number(o.amdgpu_cards)<<",\"device_mapping\":\""<<(o.device_ambiguous?"ambiguous-aggregate-only":o.gtt_total.value?"selected":"unavailable")
        <<"\",\"vram_interpretation\":\""<<(o.process.uma==true?"firmware-local-aperture-conditional":o.process.uma==false?"device-memory":"unknown")
        <<"\",\"sampling\":\"non-atomic\",\"begin_steady_ns\":"<<o.begin_ns<<",\"end_steady_ns\":"<<o.end_ns;
    const char* thp=o.thp==PreflightThp::always?"always":o.thp==PreflightThp::madvise?"madvise":o.thp==PreflightThp::never?"never":"unavailable";
    s<<",\"thp_active\":\""<<thp<<"\",\"allocator_contract\":\"unknown\",\"page_size_bytes\":"<<number(o.process.page_size);
    s<<",\"sampling_duration_ns\":"<<(o.end_ns>=o.begin_ns?o.end_ns-o.begin_ns:0);
    const auto limit=[&](const char* key,const PreflightLimit& l){s<<",\""<<key<<"\":"<<(l.infinite?"\"infinite\"":number(l.bytes));};
    limit("memlock_soft_bytes",o.process.memlock_soft);limit("memlock_hard_bytes",o.process.memlock_hard);
    s<<",\"pin_requirement\":\""<<(a.plan.L?"known":"unchecked")<<"\",\"plan\":{\"G\":"<<number(a.plan.G)<<",\"H\":"<<number(a.plan.H)
        <<",\"R\":"<<number(a.plan.R)<<",\"L\":"<<number(a.plan.L)<<",\"C\":"<<number(a.plan.C)<<"}"
        <<",\"gtt_free_bytes\":"<<number(a.gtt_free)<<",\"gtt_deficit_bytes\":"<<number(a.gtt_deficit)
        <<",\"host_deficit_bytes\":"<<number(a.host_deficit)<<",\"needed_2m_units\":"<<number(a.needed_2m_units)
        <<",\"compact_stall_delta\":"<<(a.stall_delta?std::to_string(*a.stall_delta):"null")
        <<",\"compact_interval_ns\":"<<(a.elapsed_ns?std::to_string(*a.elapsed_ns):"null")<<",\"warnings\":[";
    const auto codes=[&](const std::vector<PreflightWarning>& list){
        bool first=true;for(auto w:list) {if(!first)s<<',';first=false;s<<'"'<<warning(w)<<'"';}};
    codes(a.warnings);s<<"],\"advisories\":[";codes(a.advisories);
    s<<"]}";return s.str();
}
std::optional<StartupPreflight> emit_qwen_startup_diagnostics(bool inspection,const PreflightPaths& paths,const PreflightFactProvider& facts) noexcept {
    if(inspection) return {};
    try {
        const auto enabled=[](const char* key){const auto* v=std::getenv(key);return !v || std::string_view(v)!="0";};
        // Preserve the startup off switch: without events, collection remains disabled.
        if(!enabled("STRIX_RELEASE_EVENTS")) return {};
        std::cerr<<"strix-build-info "<<build_info_json()<<'\n';
        if(enabled("STRIX_PREFLIGHT")) {auto o=collect_preflight(paths,facts);auto a=assess_preflight(o);
            std::cerr<<"strix-startup-event "<<render_preflight(o,a)<<'\n';
            return StartupPreflight{std::move(o),std::move(a)};}
    } catch(...) { /* Diagnostics never prevent startup. */ }
    return {};
}
} // namespace strix
