#include "strix/linux_uma_observation.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix;
namespace {
int failures=0;
void expect(bool condition,const char* why) {if(!condition) {++failures;std::cerr<<why<<'\n';}}
template<class F> bool rejects(F&& f) {try {f();} catch(const std::exception&) {return true;}return false;}
}
int main() {
    const auto observed=parse_linux_uma_observation("MemTotal: 1000 kB\nMemAvailable: 400 kB\nOther: 7 MB\n",
        "Name:\tfixture\nVmRSS:\t8 kB\nVmHWM:\t12 kB\n",41);
    expect(observed.supported && observed.memory.observation_identity==41 && observed.memory.mem_total_bytes==1024000U &&
        observed.memory.mem_available_bytes==409600U && observed.memory.process_rss_bytes==8192U &&
        observed.memory.process_peak_bytes==12288U && !observed.memory.gtt_reported_bytes,
        "only actual recognized kB fields become bytes; unrelated fields and GTT are not inferred");
    expect(observed.meminfo_status==LinuxUmaReadStatus::read && observed.process_status==LinuxUmaReadStatus::read,
        "both separately parsed read statuses are retained");
    const auto json=linux_uma_observation_json(observed);
    expect(json.find("\"mem_available_bytes\":409600")!=std::string::npos &&
        json.find("\"gtt_reported_bytes\":null")!=std::string::npos &&
        json.find("\"rss_plus_gtt_computed\":false")!=std::string::npos,
        "receipt keeps host observations separate from unknown GTT/physical fit");
    for(const auto& bad:std::vector<std::string>{
        "MemTotal: -1 kB\n", "MemTotal: 1 MB\n", "MemTotal: 1kB\n", "MemTotal: +1 kB\n",
        "MemTotal: 1 kB extra\n", "MemTotal: 18446744073709551616 kB\n",
        "MemTotal: 18014398509481984 kB\n", "MemTotal: 1 kB\nMemTotal: 2 kB\n",
        "MemTotal: 0 kB\n", "MemTotal: 1 kB\nMemAvailable: 2 kB\n"}) {
        const auto invalid=parse_linux_uma_observation(bad,"VmRSS: 1 kB\nVmHWM: 2 kB\n",42);
        expect(invalid.meminfo_status==LinuxUmaReadStatus::malformed && !invalid.memory.mem_total_bytes &&
            !invalid.memory.mem_available_bytes && invalid.memory.process_rss_bytes==1024U,
            "malformed units/overflow/duplicates/contradiction invalidate only their own file's values");
    }
    const auto absent=parse_linux_uma_observation("MemTotal: 9 kB\n","Name: x\n",43);
    expect(absent.memory.mem_total_bytes==9216U && !absent.memory.mem_available_bytes && !absent.memory.process_rss_bytes,
        "missing fields remain unknown rather than zero or derived availability");
    const auto missing=parse_linux_uma_observation("","",44);
    expect(missing.meminfo_status==LinuxUmaReadStatus::unavailable && !missing.memory.mem_total_bytes &&
        !missing.memory.process_peak_bytes,"unavailable input creates no memory measurements");
    const auto oversized=parse_linux_uma_observation(std::string(65537U,'x'),"VmRSS: 1 kB\n",45);
    expect(oversized.meminfo_status==LinuxUmaReadStatus::oversized && !oversized.memory.mem_total_bytes &&
        oversized.memory.process_rss_bytes==1024U,"64KiB per-file limit is explicit and independent");
    const auto nul=parse_linux_uma_observation(std::string("MemTotal: 1 kB\0",15),"",46);
    expect(nul.meminfo_status==LinuxUmaReadStatus::malformed && !nul.memory.mem_total_bytes,"embedded NUL is not accepted as a parsed field suffix");
    const auto contradiction=parse_linux_uma_observation("MemTotal: 10 kB\n","VmRSS: 8 kB\nVmHWM: 4 kB\n",47);
    expect(contradiction.memory.mem_total_bytes==10240U && !contradiction.memory.process_rss_bytes &&
        contradiction.process_status==LinuxUmaReadStatus::malformed,"inconsistent process fields do not contaminate valid machine counters");
    expect(rejects([&]{(void)parse_linux_uma_observation("MemTotal: 1 kB\n","",0);}),
        "a parsed measurement requires its observation identity");
    LinuxUmaObservation unsupported;unsupported.meminfo_status=unsupported.process_status=LinuxUmaReadStatus::unsupported;
    const auto unavailable=linux_uma_observation_json(unsupported);
    expect(unavailable.find("\"supported\":false")!=std::string::npos &&
        unavailable.find("\"mem_total_bytes\":null")!=std::string::npos,
        "unsupported platforms preserve unknown values without fabricated observations");
    return failures?1:0;
}
