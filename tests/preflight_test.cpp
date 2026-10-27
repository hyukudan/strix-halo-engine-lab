#include "strix/preflight.hpp"
#include "strix/serve/json.hpp"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#if defined(__linux__)
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace strix;
#if defined(__linux__)
namespace {
int failures=0;
void expect(bool yes,const char* message) {if(!yes){++failures;std::cerr<<message<<'\n';}}
bool has(const PreflightAssessment& a,PreflightWarning w) {
    return std::find(a.warnings.begin(),a.warnings.end(),w)!=a.warnings.end();
}
bool advises(const PreflightAssessment& a,PreflightWarning w) {
    return std::find(a.advisories.begin(),a.advisories.end(),w)!=a.advisories.end();
}
struct Tree {
    std::filesystem::path root;
    Tree() {
        root=std::filesystem::temp_directory_path()/("strix-preflight-"+std::to_string(::getpid()));
        std::filesystem::create_directories(root);
    }
    ~Tree() {std::error_code ec;std::filesystem::remove_all(root,ec);}
    void put(const std::string& path,const std::string& value) {
        std::filesystem::create_directories((root/path).parent_path());std::ofstream(root/path)<<value;
    }
    void dir(const std::string& path) {std::filesystem::create_directories(root/path);}
    void gpu(unsigned index) {
        const auto base="sys/class/drm/card"+std::to_string(index)+"/device/";
        dir("sys/bus/pci/drivers/amdgpu");
        std::filesystem::create_directory_symlink(root/"sys/bus/pci/drivers/amdgpu",root/base/"driver");
    }
    void card(unsigned index) {
        const auto base="sys/class/drm/card"+std::to_string(index)+"/device/";
        put(base+"mem_info_gtt_total","100000000\n");put(base+"mem_info_gtt_used","0\n");
        put(base+"mem_info_vram_total","10000000\n");put(base+"mem_info_vram_used","0\n");gpu(index);
    }
};
std::string buddy(unsigned target=9) {
    std::string out="Node 0, zone Normal";
    for(unsigned i=0;i<target;++i) out+=" 0";
    return out+" 2 3 1\n"; // 2 + 3*2 + 1*4 = 12 units, six blocks.
}
}
int main() {
    Tree t;
    PreflightProcessFacts facts;facts.page_size=4096;facts.self_pid=987654321;facts.uma=true;
    facts.memlock_soft.bytes=64;facts.memlock_hard.infinite=true;
    auto collect=[&]{return collect_preflight({t.root},[&]{return facts;});};
    t.put("proc/meminfo","MemTotal: 100000 kB\nMemAvailable: 90000 kB\nHost: secret-host\nUser: /home/secret-user\nPID: 987654321\nSerial: secret-serial\nBDF: 0000:aa:00.0\nCmd: secret-command \"injection\"\n");
    t.put("proc/buddyinfo",buddy());t.put("proc/vmstat","compact_stall 999999999\ncompact_success 100\ncompact_fail 10\n");
    t.put("sys/kernel/mm/transparent_hugepage/enabled","always [madvise] never\n");
    t.dir("sys/class/kfd/kfd/proc/987654321");t.card(0);
    auto healthy=collect();auto a=assess_preflight(healthy,{1000,1000,1000,32,1000});
    expect(a.warnings.empty(),"healthy mapped single GPU has no warnings");
    expect(healthy.other_clients.value==0 && healthy.buddy.blocks.value==6 && healthy.buddy.units_2m.value==12,"self excluded; weighted buddy");
    expect(!a.stall_delta,"one sample never gives compaction delta");
    for(auto demand:{12U*2097152U-1U,12U*2097152U}) {
        PreflightInput p;p.C=demand;
        expect(!has(assess_preflight(healthy,p),PreflightWarning::contiguous_supply_risk),"demand below/equal supply");
    }
    std::ostringstream events;auto* old=std::cerr.rdbuf(events.rdbuf());unsigned calls=0;
    ::setenv("STRIX_PREFLIGHT","0",1);::setenv("STRIX_RELEASE_EVENTS","0",1);
    emit_qwen_startup_diagnostics(false,{t.root},[&]{++calls;return facts;});
    expect(calls==0 && events.str().empty(),"disabled preflight never invokes collector/provider; all logs disabled");
    ::setenv("STRIX_PREFLIGHT","1",1);::setenv("STRIX_RELEASE_EVENTS","1",1);
    emit_qwen_startup_diagnostics(true,{t.root},[&]{++calls;return facts;});
    expect(calls==0 && events.str().empty(),"inspection runs keep stderr to the one configuration object");
    const auto startup=emit_qwen_startup_diagnostics(false,{t.root},[&]{++calls;return facts;});
    expect(startup && startup->observation.end_ns && startup->observation.mem_total.value==healthy.mem_total.value,
        "startup hands its single collected sample to serving");
    std::cerr.rdbuf(old);::unsetenv("STRIX_PREFLIGHT");::unsetenv("STRIX_RELEASE_EVENTS");
    const auto event_text=events.str();
    expect(calls==1 && std::count(event_text.begin(),event_text.end(),'\n')==2,"one startup preflight and one build record");
    auto text=render_preflight(healthy,a);(void)serve::parse_json(text);
    for(const auto& secret:std::vector<std::string>{"secret-host","secret-user","/home","987654321",t.root.string(),"driver","secret-serial","0000:aa:00.0","secret-command","injection"})
        expect(text.find(secret)==text.npos,"allowlist omits fixture identities/paths");
    t.card(1);auto ambiguous=collect();
    expect(ambiguous.device_ambiguous && !ambiguous.gtt_total.value && !ambiguous.vram_total.value,"two cards never silently select first");
    facts.drm_card=1;expect(collect().gtt_total.value==100000000,"explicit DRM selection works");facts.drm_card=0;
    const std::string used="sys/class/drm/card0/device/mem_info_gtt_used";
    t.put(used,"1000");auto occupied=collect();a=assess_preflight(occupied);
    expect(advises(a,PreflightWarning::preexisting_gtt_use) && advises(a,PreflightWarning::allocation_owner_unresolved) &&
        !has(a,PreflightWarning::preexisting_gtt_use) && !has(a,PreflightWarning::gtt_plan_pressure),"idle GTT use is advisory, not pressure, with ample or unknown margin");
    expect(has(assess_preflight(occupied,{100000000-500,{},{},{},{}}),PreflightWarning::gtt_plan_pressure),"tiny use that eats the plan margin is pressure");
    {   // Wire shape: advisories stay out of warnings for unknown, ample and short margins.
        const auto codes=[](const serve::Json& j,const char* key){
            std::vector<std::string> out;
            for(const auto& c:j.find(key)->as_array()) out.push_back(c.as_string());
            return out;
        };
        const std::vector<std::string> advice{"preexisting-gtt-use","allocation-owner-unresolved"};
        for(const auto& [g,pressure]:std::vector<std::pair<PreflightNumber,bool>>{{{},false},{1000,false},{100000000-500,true}}) {
            const auto j=serve::parse_json(render_preflight(occupied,assess_preflight(occupied,{g,{},{},{},{}})));
            const auto w=codes(j,"warnings");
            expect(codes(j,"advisories")==advice && std::count(w.begin(),w.end(),"gtt-plan-pressure")==(pressure?1:0) &&
                std::count(w.begin(),w.end(),"preexisting-gtt-use")==0,"rendered advisories and warnings arrays");
        }
    }
    t.dir("sys/class/kfd/kfd/proc/12345");t.put("sys/class/kfd/kfd/proc/22222","not a directory");
    occupied=collect();a=assess_preflight(occupied);
    expect(occupied.other_clients.value==1 && !advises(a,PreflightWarning::allocation_owner_unresolved),"numeric directories only, global clients");
    {   // A numeric KFD symlink must not let outside state become a client.
        const auto outside=std::filesystem::temp_directory_path()/("strix-preflight-outside-"+std::to_string(::getpid()));
        std::filesystem::create_directories(outside);
        std::filesystem::create_directory_symlink(outside,t.root/"sys/class/kfd/kfd/proc/33333");
        expect(!collect().other_clients.value,"escaping numeric KFD symlink makes the count unavailable");
        std::filesystem::remove(t.root/"sys/class/kfd/kfd/proc/33333");std::filesystem::remove(outside);
    }
    a=assess_preflight(occupied,{100000000,100000000,1000000,65,13U*2097152U});
    expect(has(a,PreflightWarning::gtt_plan_pressure) && a.gtt_deficit==1000 && has(a,PreflightWarning::host_plan_pressure) &&
        has(a,PreflightWarning::firmware_reserve_review) && has(a,PreflightWarning::contiguous_supply_risk) && has(a,PreflightWarning::pin_limit_risk),"plan-derived deficits and finite pin warning");
    occupied.process.uma=false;expect(!has(assess_preflight(occupied,a.plan),PreflightWarning::firmware_reserve_review),"discrete VRAM is device memory");
    occupied.process.memlock_soft.infinite=true;expect(!has(assess_preflight(occupied,a.plan),PreflightWarning::pin_limit_risk),"infinite soft memlock");
    occupied.process.memlock_soft.infinite=false;
    auto later=occupied;later.end_ns+=100;
    expect(!has(assess_preflight(later,a.plan,&occupied),PreflightWarning::startup_compaction_pressure),"stable high boot counters never warn");
    later.compact.stall.value=*occupied.compact.stall.value+1;
    auto interval=assess_preflight(later,a.plan,&occupied);
    expect(interval.stall_delta==1 && interval.elapsed_ns==100 && has(interval,PreflightWarning::startup_compaction_pressure),"measured increase and deficit warns");
    expect(!has(assess_preflight(later,{},&occupied),PreflightWarning::startup_compaction_pressure),"stalls without deficit never warn");
    later.compact.stall.value=0;expect(!assess_preflight(later,a.plan,&occupied).stall_delta,"counter reset invalidates delta");
    later.compact.stall=occupied.compact.stall;later.end_ns=occupied.end_ns;expect(!assess_preflight(later,a.plan,&occupied).stall_delta,"zero interval invalid");
    later.end_ns+=100;later.process_identity+=1;expect(!assess_preflight(later,a.plan,&occupied).stall_delta,"different process invalid");
    t.put(used,"100000001");expect(!collect().gtt_total.value,"used greater than total invalidates pair");
    for(const auto& value:std::vector<std::string>{"-1","+1","1x","18446744073709551616",std::string(129,'1'),""}) {
        t.put(used,value);expect(!collect().gtt_used.value,"invalid/overflow/oversized/empty scalar unavailable");
    }
    t.put(used,"0");t.put("proc/meminfo",std::string(65537,'x'));
    expect(collect().mem_total.status==LinuxUmaReadStatus::oversized,"proc cap explicit");
    expect(!parse_preflight_meminfo("MemAvailable: 18014398509481984 kB\n").memory.mem_available_bytes,"KiB overflow rejected");
    expect(parse_preflight_meminfo("MemAvailable: 0 kB\n").memory.mem_available_bytes==0,"measured zero is known");
    expect(!parse_preflight_meminfo("MemTotal: 100 kB\n").memory.mem_available_bytes,"missing is unknown");
    auto multiple=parse_preflight_buddyinfo(buddy()+buddy(),4096);
    expect(multiple.units_2m.value==24 && multiple.blocks.value==12,"multi node/zone totals weighted");
    expect(parse_preflight_buddyinfo(buddy(),4096,std::vector<unsigned>{1}).units_2m.value==0,"restricted nodes exclude unrelated supply");
    expect(parse_preflight_buddyinfo(buddy(5),65536).units_2m.value==12,"64KiB page size changes target order");
    expect(!parse_preflight_buddyinfo(buddy(),{}).units_2m.value,"unknown page size unavailable");
    expect(!parse_preflight_buddyinfo("Node 0, zone X 0 0 0 0 0 0 0 0 0 18446744073709551615 1",4096).units_2m.value,"buddy addition overflow");
    for(auto pair:{std::pair{"[always] madvise never",PreflightThp::always},std::pair{"always [madvise] never",PreflightThp::madvise},
        std::pair{"always madvise [never]",PreflightThp::never},std::pair{"[always] [never]",PreflightThp::unavailable},
        std::pair{"[secret-host]",PreflightThp::unavailable}}) expect(parse_preflight_thp(pair.first)==pair.second,"THP active modes and malformed selections");
    expect(!parse_preflight_vmstat("compact_stall -1\n").stall.value,"malformed compaction");
    expect(!parse_preflight_vmstat("compact_stall 1\ncompact_stall 2\n").stall.value,"duplicate counters rejected");
    const auto kfd=t.root/"sys/class/kfd/kfd/proc";
    // Inject restricted credentials in a child so root runners also see EACCES.
    ::chmod(kfd.c_str(),0);
    const auto child=::fork();
    if(child==0) {
        if(::geteuid()==0 && ::seteuid(65534)!=0) ::_exit(2);
        ::_exit(!collect().other_clients.value?0:1);
    }
    int child_status=0;if(child>0) ::waitpid(child,&child_status,0);
    ::chmod(kfd.c_str(),0755);
    expect(child>0 && WIFEXITED(child_status) && WEXITSTATUS(child_status)==0,"denied KFD stays unavailable with injected credentials");
    std::filesystem::remove_all(kfd);expect(!collect().other_clients.value,"absent KFD unknown");
    std::filesystem::create_directory_symlink("proc",kfd);expect(!collect().other_clients.value,"unreadable KFD injected symlink loop unknown");
    std::filesystem::remove(kfd);t.dir("sys/class/kfd/kfd/proc");expect(collect().other_clients.value==0,"empty KFD measured zero");
    for(unsigned i=0;i<4097;++i) t.dir("sys/class/kfd/kfd/proc/"+std::to_string(i));
    expect(collect().other_clients.status==LinuxUmaReadStatus::oversized && !collect().other_clients.value,"KFD cap is unknown, never partial count");
    for(unsigned i=0;i<257;++i) t.dir("sys/class/drm/entry"+std::to_string(i));
    expect(collect().gtt_total.status==LinuxUmaReadStatus::oversized,"topology cap invalidates selected data");
    expect(!collect_preflight({t.root},[]{throw 1;return PreflightProcessFacts{};}).gtt_total.value,"provider failure cannot escape");
    // Fixture symlinks cannot escape into the host, even for proc text.
    std::filesystem::remove(t.root/"proc/meminfo");std::filesystem::create_symlink("/proc/meminfo",t.root/"proc/meminfo");
    expect(!collect().mem_total.value,"fixture root containment");
    std::cout<<"preflight failures="<<failures<<'\n';return failures?1:0;
}
#else
int main() {
    return collect_preflight().mem_total.status==LinuxUmaReadStatus::unsupported?0:1;
}
#endif
