#include "../lab/gpu_visibility/run_options.hpp"
#include <cassert>
#include <vector>
using namespace strix::transport::visibility_lab;
static RunOptions parse_values(std::vector<std::string> values){std::vector<char*> argv;for(auto& v:values)argv.push_back(v.data());return parse(static_cast<int>(argv.size()),argv.data());}
int main(){
 std::vector<std::string> args={"test","--experimental-commissioning","--check-config",
 "--campaign-id","campaign","--run-id","run","--dma-device","/dev/tbstream0",
 "--local-endpoint",std::string(32,'1'),"--peer-endpoint",std::string(32,'2'),"--gpu-uuid",std::string(32,'3'),
 "--kernel-release","test-kernel","--driver-build-id","abcd","--physical-link","link0",
 "--epoch","7","--request-id","2","--payload-bytes","4097","--control-port","8888",
 "--gpu-runtime-version","70000000","--gpu-driver-version","70000000","--role","receiver","--control-mode","listen"};
 auto good=parse_values(args);assert(good.bytes==4097&&good.epoch==7&&good.receiver&&good.check_only);
 const auto reject=[](auto input){bool fail=false;try{(void)parse_values(input);}catch(const std::invalid_argument&){fail=true;}assert(fail);};
 auto duplicate=args;duplicate.insert(duplicate.end(),{"--epoch","9"});reject(duplicate);
 auto unknown=args;unknown.insert(unknown.end(),{"--magic-coherence","1"});reject(unknown);
 auto no_opt=args;no_opt.erase(no_opt.begin()+1);reject(no_opt);
 auto duplex=args;duplex.push_back("--duplex-pairs");reject(duplex);duplex.insert(duplex.end(),{"--iterations","4"});assert(parse_values(duplex).duplex);
 auto zero=args;*std::next(std::find(zero.begin(),zero.end(),"--epoch"))="0";reject(zero);
 auto huge=args;*std::next(std::find(huge.begin(),huge.end(),"--payload-bytes"))="18446744073709551615";reject(huge);
 auto bad_budget=args;bad_budget.insert(bad_budget.end(),{"--timeout-ms","2","--consumer-delay-ms","2"});reject(bad_budget);
 auto same=args;*std::next(std::find(same.begin(),same.end(),"--peer-endpoint"))=std::string(32,'1');reject(same);
 assert(quote("x\n\"y")=="\"x\\u000a\\\"y\"");
}
