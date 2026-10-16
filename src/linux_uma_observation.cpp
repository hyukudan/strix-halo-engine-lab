#include "strix/linux_uma_observation.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {
constexpr std::size_t limit=64U*1024U;
using Field=std::pair<std::string_view,std::optional<std::uint64_t>*>;
LinuxUmaReadStatus parse(std::string_view text,std::span<Field> fields) {
    if(text.size()>limit) return LinuxUmaReadStatus::oversized;
    if(text.empty()) return LinuxUmaReadStatus::unavailable;
    std::array<bool,4> seen{};std::size_t lines=0;
    while(!text.empty()) {
        if(++lines>2048U) return LinuxUmaReadStatus::oversized;
        const auto end=text.find('\n');const auto line=text.substr(0,end);
        if(line.size()>4096U || line.find('\0')!=std::string_view::npos) return LinuxUmaReadStatus::malformed;
        const auto colon=line.find(':');
        if(colon!=std::string_view::npos) for(std::size_t i=0;i<fields.size();++i) if(line.substr(0,colon)==fields[i].first) {
            if(seen[i]) return LinuxUmaReadStatus::malformed;
            seen[i]=true;auto value=line.substr(colon+1U);
            while(!value.empty() && (value.front()==' ' || value.front()=='\t')) value.remove_prefix(1);
            std::uint64_t number=0;std::size_t digits=0;
            while(digits<value.size() && value[digits]>='0' && value[digits]<='9') {
                const auto digit=static_cast<unsigned>(value[digits]-'0');
                if(number>(UINT64_MAX-digit)/10U) return LinuxUmaReadStatus::malformed;
                number=number*10U+digit;++digits;
            }
            if(!digits) return LinuxUmaReadStatus::malformed;
            value.remove_prefix(digits);
            if(value.empty() || (value.front()!=' ' && value.front()!='\t')) return LinuxUmaReadStatus::malformed;
            while(!value.empty() && (value.front()==' ' || value.front()=='\t')) value.remove_prefix(1);
            if(!value.starts_with("kB")) return LinuxUmaReadStatus::malformed;
            value.remove_prefix(2);
            while(!value.empty() && (value.front()==' ' || value.front()=='\t')) value.remove_prefix(1);
            if(!value.empty() || number>UINT64_MAX/1024U) return LinuxUmaReadStatus::malformed;
            *fields[i].second=number*1024U;
        }
        if(end==std::string_view::npos) break;
        text.remove_prefix(end+1U);
    }
    return LinuxUmaReadStatus::read;
}
void clear(std::span<Field> fields) {for(auto& field:fields) field.second->reset();}
const char* status_name(LinuxUmaReadStatus s) {
    switch(s) {
    case LinuxUmaReadStatus::unavailable:return "unavailable";
    case LinuxUmaReadStatus::read:return "read";
    case LinuxUmaReadStatus::malformed:return "malformed";
    case LinuxUmaReadStatus::oversized:return "oversized";
    case LinuxUmaReadStatus::unsupported:return "unsupported";
    }
    throw std::invalid_argument("unknown Linux UMA read status");
}
#if defined(__linux__)
std::pair<std::string,LinuxUmaReadStatus> read(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary);
    if(!input) return {{},LinuxUmaReadStatus::unavailable};
    std::string bytes(limit+1U,'\0');input.read(bytes.data(),static_cast<std::streamsize>(bytes.size()));
    const auto n=input.gcount();
    if(input.bad() || n<0) return {{},LinuxUmaReadStatus::unavailable};
    if(static_cast<std::uint64_t>(n)>limit) return {{},LinuxUmaReadStatus::oversized};
    bytes.resize(static_cast<std::size_t>(n));return {std::move(bytes),LinuxUmaReadStatus::read};
}
std::uint64_t now() noexcept {
    const auto value=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    return value>0?static_cast<std::uint64_t>(value):0U;
}
std::uint64_t next_id() noexcept {
    static std::atomic<std::uint64_t> next{1};auto value=next.load(std::memory_order_relaxed);
    while(value!=UINT64_MAX) if(next.compare_exchange_weak(value,value+1U,std::memory_order_relaxed)) return value;
    return 0;
}
#endif
}
LinuxUmaObservation parse_linux_uma_observation(std::string_view meminfo,std::string_view process,std::uint64_t id) {
    if(!id) throw std::invalid_argument("UMA observation requires a nonzero actual read/fixture identity");
    LinuxUmaObservation result;result.supported=true;result.memory.observation_identity=id;
    std::array<Field,2> machine{{{"MemTotal",&result.memory.mem_total_bytes},{"MemAvailable",&result.memory.mem_available_bytes}}};
    std::array<Field,2> self{{{"VmRSS",&result.memory.process_rss_bytes},{"VmHWM",&result.memory.process_peak_bytes}}};
    result.meminfo_status=parse(meminfo,machine);result.process_status=parse(process,self);
    if(result.meminfo_status!=LinuxUmaReadStatus::read ||
        (result.memory.mem_total_bytes && (!*result.memory.mem_total_bytes ||
            (result.memory.mem_available_bytes && *result.memory.mem_available_bytes>*result.memory.mem_total_bytes)))) {
        if(result.meminfo_status==LinuxUmaReadStatus::read) result.meminfo_status=LinuxUmaReadStatus::malformed;
        clear(machine);
    }
    if(result.process_status!=LinuxUmaReadStatus::read || (result.memory.process_rss_bytes && result.memory.process_peak_bytes &&
        *result.memory.process_rss_bytes>*result.memory.process_peak_bytes)) {
        if(result.process_status==LinuxUmaReadStatus::read) result.process_status=LinuxUmaReadStatus::malformed;
        clear(self);
    }
    return result;
}
LinuxUmaObservation observe_linux_uma_memory(const std::filesystem::path& root) {
#if defined(__linux__)
    const auto begin=now();const auto id=next_id();
    if(!id) return {};
    const auto machine=read(root/"meminfo");const auto process=read(root/"self"/"status");
    auto result=parse_linux_uma_observation(machine.first,process.first,id);
    if(machine.second!=LinuxUmaReadStatus::read) result.meminfo_status=machine.second;
    if(process.second!=LinuxUmaReadStatus::read) result.process_status=process.second;
    result.begin_steady_ns=begin;result.end_steady_ns=now();return result;
#else
    (void)root;LinuxUmaObservation result;result.meminfo_status=LinuxUmaReadStatus::unsupported;
    result.process_status=LinuxUmaReadStatus::unsupported;return result;
#endif
}
std::string linux_uma_observation_json(const LinuxUmaObservation& observed) {
    (void)build_uma_owner_inventory({}, observed.memory);
    const auto number=[](const std::optional<std::uint64_t>& n){return n?std::to_string(*n):std::string("null");};
    std::ostringstream out;out<<"{\"kind\":\"reported-linux-host-counters-not-physical-fit\",\"supported\":"
        <<(observed.supported?"true":"false")<<",\"observation_identity\":"<<observed.memory.observation_identity
        <<",\"scope\":\"reported-system-and-self-process-memory\""
        <<",\"sampling\":\"bounded-two-file-read-interval-not-atomic\",\"begin_steady_ns\":"<<observed.begin_steady_ns
        <<",\"end_steady_ns\":"<<observed.end_steady_ns<<",\"meminfo_status\":\""<<status_name(observed.meminfo_status)
        <<"\",\"process_status\":\""<<status_name(observed.process_status)<<"\",\"mem_total_bytes\":"<<number(observed.memory.mem_total_bytes)
        <<",\"mem_available_bytes\":"<<number(observed.memory.mem_available_bytes)<<",\"process_rss_bytes\":"<<number(observed.memory.process_rss_bytes)
        <<",\"process_peak_bytes\":"<<number(observed.memory.process_peak_bytes)
        <<",\"process_peak_kind\":\"VmHWM-resident-high-water-mark\",\"gtt_reported_bytes\":"<<number(observed.memory.gtt_reported_bytes)
        <<",\"rss_plus_gtt_computed\":false}";return out.str();
}
} // namespace strix
