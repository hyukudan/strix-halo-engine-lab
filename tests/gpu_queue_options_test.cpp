#include "../lab/gpu_queue/run_options.hpp"
#include <cassert>
#include <iostream>
#include <vector>
using namespace strix::transport;
std::vector<std::string> arguments(){return {"queue","--experimental-commissioning","--configured-gpu-direct",
 "--campaign-id","campaign","--run-id","run","--dma-device","/dev/explicit-synthetic",
 "--local-endpoint","11111111111111111111111111111111","--peer-endpoint","22222222222222222222222222222222",
 "--gpu-uuid","33333333333333333333333333333333","--kernel-release","synthetic-kernel","--driver-build-id","aabb",
 "--physical-link","link","--role","sender","--control-mode","listen","--epoch","1","--request-id","1",
 "--payload-bytes","4097","--control-port","41100","--gpu-runtime-version","1","--gpu-driver-version","1"};}
queue_lab::Options parse(std::vector<std::string> args){std::vector<char*> pointers;for(auto& a:args)pointers.push_back(a.data());return queue_lab::parse(static_cast<int>(pointers.size()),pointers.data());}
int main(int argc,char** argv){assert(argc==2);const std::string name=argv[1];auto args=arguments();bool negative=true;
 if(name=="valid"){auto o=parse(args);assert(o.depth==8&&o.windows==4&&!o.full_duplex);negative=false;}
 else if(name=="duplex"){args.insert(args.end(),{"--full-duplex","--queue-depth","4","--windows","3","--recover-every","1"});auto o=parse(args);assert(o.depth==4&&o.windows==3&&o.recovery_every==1&&o.full_duplex);negative=false;}
 else if(name=="controller"){args.insert(args.end(),{"--controller-reset","--recover-every","1"});auto o=parse(args);assert(o.controller_reset&&!o.include_network);negative=false;}
 else if(name=="network"){args.insert(args.end(),{"--controller-reset","--recover-every","1","--include-network","--independent-control"});auto o=parse(args);assert(o.include_network&&o.independent_control);negative=false;}
 else if(name=="network_no_control"){args.insert(args.end(),{"--controller-reset","--recover-every","1","--include-network"});}
 else if(name=="network_only"){args.insert(args.end(),{"--include-network","--independent-control"});}
 else if(name=="controller_no_schedule"){args.insert(args.end(),{"--controller-reset"});}
 else if(name=="controller_duplicate"){args.insert(args.end(),{"--controller-reset","--controller-reset","--recover-every","1"});}
 else if(name=="ack"){args.erase(std::find(args.begin(),args.end(),"--configured-gpu-direct"));}
 else if(name=="duplicate"){args.insert(args.end(),{"--windows","2","--windows","3"});}
 else if(name=="geometry"){*std::next(std::find(args.begin(),args.end(),"--payload-bytes"))="12289";}
 else if(name=="depth"){args.insert(args.end(),{"--queue-depth","9"});}
 else if(name=="legacy"){args.insert(args.end(),{"--iterations","2"});}
 else if(name=="epoch"){*std::next(std::find(args.begin(),args.end(),"--epoch"))="18446744073709551615";args.insert(args.end(),{"--recover-every","1"});}
 else if(name=="budget"){*std::next(std::find(args.begin(),args.end(),"--payload-bytes"))="2093056";args.insert(args.end(),{"--ring-frames","4096","--windows","128","--full-duplex"});}
 else assert(false);
 if(negative){bool rejected=false;try{static_cast<void>(parse(args));}catch(const std::exception&){rejected=true;}assert(rejected);}
 std::cout<<"PASS "<<name<<'\n';
}
