// Resident Windows PRO worker. Build never launches this executable.
#include "sender.hpp"
#include "runtime.hpp"
#include "verified_io.hpp"
#include "policy.hpp"
#include "strix/serve/json.hpp"
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>

namespace p=strix::glm::rtxpf::producer;
namespace w=strix::glm::import;
using strix::serve::Json;
const Json& field(const Json& j,const char* name){auto* v=j.find(name);w::require(v!=nullptr,"missing request field");return *v;}
void fields(const Json& j,std::initializer_list<std::string> allowed){w::require(j.is_object(),"expected object");for(auto& [key,value]:j.as_object())w::require(std::find(allowed.begin(),allowed.end(),key)!=allowed.end(),"unsupported request field");}
struct AsyncSender {
    struct Job {unsigned layer,component;w::Bytes bytes;};
    std::unique_ptr<p::Sender> sender;std::mutex mutex;std::condition_variable cv;
    std::deque<Job> jobs;bool closed=false;std::exception_ptr error;std::future<void> worker;
    explicit AsyncSender(std::unique_ptr<p::Sender> s):sender(std::move(s)),worker(std::async(std::launch::async,[this]{
        try {while(true){Job job;{std::unique_lock lock(mutex);cv.wait(lock,[&]{return closed||!jobs.empty();});if(jobs.empty())break;job=std::move(jobs.front());jobs.pop_front();cv.notify_all();}sender->frame(job.layer,job.component,job.bytes);}}
        catch(...){std::lock_guard lock(mutex);error=std::current_exception();closed=true;jobs.clear();cv.notify_all();}
    })){}
    void submit(unsigned l,unsigned c,std::span<const std::byte> b){std::unique_lock lock(mutex);cv.wait(lock,[&]{return jobs.size()<2||closed;});if(error)std::rethrow_exception(error);w::require(!closed,"sender closed");jobs.push_back({l,c,{b.begin(),b.end()}});cv.notify_all();}
    void drain(){{std::lock_guard lock(mutex);closed=true;cv.notify_all();}if(worker.valid())worker.get();if(error)std::rethrow_exception(error);}
    ~AsyncSender(){try{drain();}catch(...){} }
};
struct Http {
    std::string method,path,body;
};
// One bounded HTTP request per connection. No chunked bodies, pipelining,
// duplicate Content-Length, implicit tokenization or unsupported generation fields.
Http request(w::Socket fd){
    std::string data;char c;while(data.find("\r\n\r\n")==std::string::npos){w::require(data.size()<16384,"HTTP header too large");w::require(recv(fd,&c,1,0)==1,"HTTP header EOF");data+=c;}
    auto end=data.find("\r\n");auto line=data.substr(0,end);auto a=line.find(' '),b=line.rfind(' ');
    w::require(a!=std::string::npos&&a<b&&line.substr(b+1)=="HTTP/1.1","HTTP request line");
    Http r{line.substr(0,a),line.substr(a+1,b-a-1),{}};bool seen=false;std::size_t size=0;
    for(auto off=end+2;off+2<data.size();){auto next=data.find("\r\n",off);if(next==off)break;auto header=data.substr(off,next-off);auto colon=header.find(':');w::require(colon!=std::string::npos,"HTTP header");auto key=header.substr(0,colon);for(char& x:key)if(x>='A'&&x<='Z')x+=32;
        if(key=="content-length"){w::require(!seen,"duplicate length");seen=true;auto value=header.substr(colon+1);auto first=value.find_first_not_of(" \t");w::require(first!=std::string::npos,"empty length");value=value.substr(first);w::require(value.find_first_not_of("0123456789")==std::string::npos&&value.size()<10,"invalid length");size=std::stoull(value);w::require(size<=4*1024*1024,"HTTP body too large");}
        w::require(key!="transfer-encoding","transfer encoding unsupported");off=next+2;
    }
    if(r.method=="POST")w::require(seen,"Content-Length required");
    r.body.resize(size);if(size)w::io(fd,std::as_writable_bytes(std::span(r.body)),false);return r;
}
void response(w::Socket fd,int code,const std::string& body){std::string text="HTTP/1.1 "+std::to_string(code)+" Result\r\nContent-Type: application/json\r\nContent-Length: "+std::to_string(body.size())+"\r\nConnection: close\r\n\r\n"+body;w::io(fd,std::as_writable_bytes(std::span(text)),true);}

int main(int argc,char** argv){try{
    std::string cache,model,manifest_hash,key_file,export_dir,prompt_file,identity="glm-pf-rtx-native-v1",weight_dirs,bind="127.0.0.1";unsigned port=8097,capacity=8192,tile=0,host_floor_gib=8,max_device_layers=43,max_pinned_layers=43,drive_readers=16;int file_cache_mode=1;bool require_file_tier=false,allow=false,graphs=true,self_check=false,legacy_index=false,serial_stage=false;std::vector<std::string> check_prompts;
    for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--require-file-tier")require_file_tier=true;else if(a=="--legacy-fp32-index")legacy_index=true;else if(a=="--serial-stage")serial_stage=true;else if(a=="--self-check-schedule")self_check=true;else if(a=="--allow-gpu")allow=true;else if(a=="--no-graphs")graphs=false;else {w::require(i+1<argc,"missing option argument");std::string value=argv[++i];if(a=="--weight-dirs")weight_dirs=value;else if(a=="--drive-readers"){drive_readers=unsigned(std::stoul(value));w::require(drive_readers>=8&&drive_readers<=32,"drive readers must be 8..32");}else if(a=="--file-tier-cache"){w::require(value=="off"||value=="auto"||value=="on","file cache mode");file_cache_mode=value=="off"?0:value=="auto"?1:2;}else if(a=="--cache")cache=value;else if(a=="--model")model=value;else if(a=="--cache-manifest-sha256")manifest_hash=value;else if(a=="--key-file")key_file=value;else if(a=="--export-state")export_dir=value;else if(a=="--prompt"){prompt_file=value;check_prompts.push_back(value);}else if(a=="--identity")identity=value;else if(a=="--bind")bind=value;else if(a=="--port")port=unsigned(std::stoul(value));else if(a=="--capacity")capacity=unsigned(std::stoul(value));else if(a=="--row-tile")tile=value=="auto"?0:unsigned(std::stoul(value));else if(a=="--max-device-layers")max_device_layers=p::layer_limit(value);else if(a=="--max-pinned-layers")max_pinned_layers=p::layer_limit(value);else if(a=="--host-floor-gib"){auto n=std::stoul(value);w::require(n<=1024,"host floor range");host_floor_gib=unsigned(n);}else throw std::runtime_error("unknown service option");}}
    w::require(allow,"requires explicit --allow-gpu during an authorized PRO window");
    const char* visible=std::getenv("CUDA_VISIBLE_DEVICES");p::require_pro_uuid(visible?visible:"");
    w::require((cache.empty()!=model.empty())&&(!cache.empty()? !manifest_hash.empty():manifest_hash.empty())&&(!key_file.empty()||!export_dir.empty()||self_check)&&port>0&&port<=65535,"exactly one of --model or --cache (with pinned manifest SHA), key-file (HTTP) and valid port required");
    if(const char* off=std::getenv("STRIX_GLM_RTX_PREFILL"))w::require(std::string(off)!="0","native prefill disabled");
    std::unique_ptr<p::VerifiedManifest> manifest_file;
    if(!cache.empty())manifest_file=std::make_unique<p::VerifiedManifest>(std::filesystem::path(cache)/"manifest.json",manifest_hash);
    auto startup_begin=std::chrono::steady_clock::now();
    p::Runtime runtime(model.empty()?cache:model,capacity,tile,!model.empty(),graphs,host_floor_gib,legacy_index,serial_stage,manifest_file.get(),max_device_layers,max_pinned_layers,require_file_tier,weight_dirs,drive_readers,file_cache_mode);auto health_value=strix::serve::parse_json(runtime.health());
    health_value.set("startup_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-startup_begin).count());const auto health=health_value.dump();
    std::wstring module(32768,L'\0');auto module_length=GetModuleFileNameW(nullptr,module.data(),DWORD(module.size()));w::require(module_length>0&&module_length<module.size(),"executable path");module.resize(module_length);
    const auto executable_hash=strix::model_content_identity_hex(strix::sha256_file_content(std::filesystem::path(module)));
    if(self_check){
        w::require(!check_prompts.empty(),"self-check requires --prompt");
        auto record=Json::object();record.set("health",health_value);std::cout<<record.dump()<<"\n"<<std::flush;
        for(const auto& file:check_prompts){
            std::ifstream in(file,std::ios::binary);w::require(bool(in),"open self-check prompt");
            std::vector<char> bytes((std::istreambuf_iterator<char>(in)),{});w::require(!in.bad()&&bytes.size()%4==0&&bytes.size()<=std::size_t(capacity)*4,"self-check prompt extent");
            std::vector<std::uint32_t> tokens(bytes.size()/4);std::memcpy(tokens.data(),bytes.data(),bytes.size());
            auto result=runtime.self_check(tokens,identity);std::cout<<result<<"\n"<<std::flush;
            if(strix::serve::parse_json(result).find("verdict")->as_string()!="EXACT")return 3;
        }
        return 0;
    }
    if(!export_dir.empty()){
        w::require(!prompt_file.empty(),"export needs --prompt little-endian u32 file");
        std::ifstream input(prompt_file,std::ios::binary);w::require(bool(input),"open prompt file");std::vector<char> bytes((std::istreambuf_iterator<char>(input)),{});w::require(!input.bad()&&bytes.size()%4==0,"prompt file");std::vector<std::uint32_t> tokens(bytes.size()/4);std::memcpy(tokens.data(),bytes.data(),bytes.size());w::require(tokens.size()>=2&&tokens.size()<=capacity,"export prompt capacity");
        std::filesystem::create_directories(export_dir);std::ofstream out[2];for(unsigned rank=0;rank<2;++rank){out[rank].open(std::filesystem::path(export_dir)/("state-r"+std::to_string(rank)+".bin.partial"),std::ios::binary);w::Bytes header;w::number(header,tokens.size()-1);w::number(header,tokens.size());out[rank].write(reinterpret_cast<const char*>(header.data()),header.size());}
        runtime.prefill(tokens,identity,[&](unsigned rank,unsigned,unsigned,std::span<const std::byte> data){out[rank].write(reinterpret_cast<const char*>(data.data()),data.size());w::require(bool(out[rank]),"export state write failed");});
        auto receipt=Json::object();receipt.set("producer_identity",identity);receipt.set("qualification","DRAFT");receipt.set("frontier",std::uint64_t(tokens.size()-1));receipt.set("producer_executable_sha256",executable_hash);receipt.set("model_sha256",runtime.model_sha256());
        for(unsigned rank=0;rank<2;++rank){out[rank].close();w::require(bool(out[rank]),"export close failed");auto path=std::filesystem::path(export_dir)/("state-r"+std::to_string(rank)+".bin");std::filesystem::rename(path.string()+".partial",path);receipt.set("rank"+std::to_string(rank)+"_sha256",strix::model_content_identity_hex(strix::sha256_file_content(path)));}
        std::cout<<receipt.dump()<<"\n";return 0;
    }
    w::require(identity=="glm-pf-rtx-native-v1","alternate identities require --export-state; wire v1 cannot accept them");
    auto expected_key=w::read_key(key_file);p::SocketRuntime sockets;
    w::SocketOwner listener;listener.fd=socket(AF_INET,SOCK_STREAM,0);w::require(listener.fd!=w::bad_socket,"HTTP socket");sockaddr_in address{};address.sin_family=AF_INET;address.sin_port=htons(static_cast<u_short>(port));w::require(inet_pton(AF_INET,bind.c_str(),&address.sin_addr)==1,"numeric HTTP bind");
    w::require(::bind(listener.fd,reinterpret_cast<sockaddr*>(&address),sizeof address)==0&&listen(listener.fd,8)==0,"HTTP bind/listen");
    std::mutex busy;std::atomic<unsigned> connections=0;
    for(;;){auto fd=accept(listener.fd,nullptr,nullptr);if(fd==w::bad_socket)continue;if(connections.fetch_add(1)>=8){--connections;w::close_socket(fd);continue;}
        std::thread([&,fd]{w::SocketOwner client;client.fd=fd;DWORD timeout=120000;setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,reinterpret_cast<char*>(&timeout),sizeof timeout);setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,reinterpret_cast<char*>(&timeout),sizeof timeout);
            try{auto req=request(fd);if(req.method=="GET"&&req.path=="/health"){response(fd,200,health);}
                else if(req.method=="POST"&&req.path=="/v1/prefill"){
                    auto body=strix::serve::parse_json(req.body);fields(body,{"tokens","handoff","identity"});auto& handoff=field(body,"handoff");fields(handoff,{"rank0","rank1","key","node_identity"});
                    // The request supplies the raw shared key as 64 hex digits;
                    // compare to the startup key before connecting to either rank.
                    auto supplied=w::unhex(field(handoff,"key").as_string());auto expected=w::hash(expected_key);
                    auto key_bytes=std::as_bytes(std::span(supplied.bytes));w::require(w::secure_equal(w::hash(key_bytes),expected),"handoff key authentication failed");
                    auto id=body.find("identity")?body.find("identity")->as_string():"glm-pf-rtx-native-v1";
                    w::require(id=="glm-pf-rtx-native-v1","receiver wire v1 accepts only glm-pf-rtx-native-v1; alternate index identities cannot hand off");
                    auto node=handoff.find("node_identity")?handoff.find("node_identity")->as_string():"r1b";w::require(node=="r1"||node=="r1b"||node=="r1b-all","node identity");
                    std::vector<std::uint32_t> tokens;for(auto& t:field(body,"tokens").as_array()){auto n=t.as_int();w::require(n>=0&&n<154880,"token out of vocabulary");tokens.push_back(std::uint32_t(n));}w::require(tokens.size()>=2&&tokens.size()<=capacity,"prompt capacity");
                    std::unique_lock lock(busy,std::try_to_lock);if(!lock.owns_lock()){response(fd,503,"{\"error\":\"producer busy\"}");--connections;return;}
                    auto started=std::chrono::steady_clock::now();w::Begin b;b.layers=46;b.frontier=tokens.size()-1;b.total=tokens.size();b.node=node;b.producer=id;b.model=w::unhex(runtime.model_sha256());b.prompt=w::hash(std::as_bytes(std::span(tokens)));std::random_device random;for(auto& x:b.transaction.bytes)x=std::uint8_t(random());
                    auto connect_rank=[&](unsigned rank){auto begin=b;begin.rank=rank;auto endpoint=field(handoff,rank?"rank1":"rank0").as_string();return std::make_unique<p::Sender>(endpoint,expected_key,begin);};
                    auto c0=std::async(std::launch::async,connect_rank,0),c1=std::async(std::launch::async,connect_rank,1);
                    AsyncSender rank0(c0.get()),rank1(c1.get());
                    runtime.prefill(tokens,id,[&](unsigned rank,unsigned layer,unsigned component,std::span<const std::byte> bytes){(rank?rank1:rank0).submit(layer,component,bytes);});
                    const auto produced=std::chrono::steady_clock::now();rank0.drain();rank1.drain();auto s0=std::async(std::launch::async,[&]{rank0.sender->commit();}),s1=std::async(std::launch::async,[&]{rank1.sender->commit();});s0.get();s1.get();
                    auto receipt=Json::object();receipt.set("status","LOCALLY_SEALED_BOTH");receipt.set("global_commit","requires both node receipts");receipt.set("qualification","DRAFT");receipt.set("requested_route","rtx-native");receipt.set("effective_route","rtx-native");receipt.set("producer_identity",id);receipt.set("node_prefill_identity",node=="r1b"?"r1":node);receipt.set("node_decode_identity",node);receipt.set("composed_identity",id+"+"+(node=="r1b"?"r1":node)+"+"+node);receipt.set("model_sha256",runtime.model_sha256());receipt.set("producer_executable_sha256",executable_hash);receipt.set("cache_manifest_sha256",model.empty()?Json(manifest_hash):Json());receipt.set("weight_source",model.empty()?"cache":"shards");receipt.set("qualification_receipt",Json());receipt.set("node_executable_hashes",Json());receipt.set("phase_timings",strix::serve::parse_json(runtime.timings()));receipt.set("fallback_reason",Json());receipt.set("frontier",std::uint64_t(b.frontier));receipt.set("mtp","prefix pairs complete; node seed pending");receipt.set("raw_bytes",rank0.sender->raw_bytes()+rank1.sender->raw_bytes());receipt.set("wire_bytes",rank0.sender->wire_bytes()+rank1.sender->wire_bytes());receipt.set("producer_ms",std::chrono::duration<double,std::milli>(produced-started).count());receipt.set("total_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count());receipt.set("capacity_cache_profile",strix::serve::parse_json(health));response(fd,200,receipt.dump());
                }else response(fd,404,"{\"error\":\"unknown endpoint\"}");
            }catch(const std::exception&){try{response(fd,400,"{\"error\":\"prefill rejected or aborted; no global commit receipt\"}");}catch(...){} }
            --connections;
        }).detach();
    }
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
