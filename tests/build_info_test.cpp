#include "strix/build_info.hpp"
#include "strix/serve/json.hpp"
#include <iostream>
#include <string>

int main() {
    const std::string json(strix::build_info_json());
    try {
        const auto parsed=strix::serve::parse_json(json);
#if defined(STRIX_TEST_BUILD_INFO_OFF)
        if(parsed.find("status")->as_string()!="unavailable" || parsed.find("git_sha")) return 1;
#else
        if(parsed.find("status")->as_string()!="available" || !parsed.find("git_sha")) return 1;
#endif
    }
    catch(...) { std::cerr<<"Invalid build JSON\n";return 1; }
    if(json!=strix::build_info() || json.find("\"format\":\"strix-build\"")==json.npos ||
        json.find("\"revision\":1")==json.npos || json.find('/')!=json.npos ||
        json.find("C:")!=json.npos || json.find('\\')!=json.npos) return 1;
    // A cached HIP architecture is not an active target of a CPU-only build.
    if(json.find("\"STRIX_ENABLE_HIP_RUNTIME\":false")!=json.npos && json.find("\"target_isa\":[]")==json.npos) return 1;
    std::cout<<json<<'\n';
    return 0;
}
