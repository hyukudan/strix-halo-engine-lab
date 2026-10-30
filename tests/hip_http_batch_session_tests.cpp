// The real fixed-session batch HTTP adapter, with inert slots and a scripted
// session runner. No device query, allocation, model weights or GPU execution.
#include "../apps/hip_http_serve.hpp"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace strix;
using namespace strix::decode_cli;
namespace {
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
void receipt_compatibility() {
    auto tokenizer=serve::BpeTokenizer::from_parts({"a","b","<|endoftext|>"},{1,1,3},{},"qwen35");
    HttpConstraintRequests constraints(tokenizer,false);
    MtpServeResult result;
    result.paid_drafts=7; result.chosen_window_rows[4]=3;
    result.final_state_digest="final"; result.final_state_position=8;
    result.prompt_state_digest="prompt"; result.session_json="{}";
    const MtpServeRunner run=[&](const MtpServeRequest&) { return result; };
    MtpBridgeEngine engine(run,64,constraints,false,false,false);
    serve::GenerationJob job; job.prompt={0}; job.max_tokens=1;
    const auto bytes=[](const serve::GenerationStats& stats) {
        auto json=serve::Json::object();
        for(const auto& [key,value]:stats.engine_extra) json.set(key,serve::parse_json(value));
        return json.dump();
    };
    const auto off=bytes(engine.generate(job));
    // Literal historical receipt payload/order, including following fields.
    require(off==R"({"final_state_digest":"final","final_state_position":8,"prompt_state_digest":"prompt","session":{}})",
        "M1 off single HTTP receipt retains historical bytes");
    serve::GenerationStats batch; batch.engine_extra.emplace_back("sentinel","1");
    const auto before=bytes(batch); append_mtp_pool_receipt(batch,result);
    require(bytes(batch)==before,"M1 off batch receipt appends zero bytes");
    result.qsa_mtp_pool_active=true;
    require(bytes(engine.generate(job)).find("\"mtp_paid_drafts\":7")!=std::string::npos,
        "M1 active owner emits receipt fields");
    append_mtp_pool_receipt(batch,result);
    require(bytes(batch).find("\"mtp_chosen_window_rows_histogram\":{\"1\":0,\"2\":0,\"3\":0,\"4\":3")!=std::string::npos,
        "M1 active batch owner emits histogram");
}
std::atomic<bool> headers_seen{false};
std::string request(unsigned port, const std::string& body) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    require(fd >= 0, "socket");
    timeval timeout{5,0};
    ::setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
    sockaddr_in address{}; address.sin_family=AF_INET; address.sin_port=htons(static_cast<std::uint16_t>(port));
    address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    if (::connect(fd,reinterpret_cast<sockaddr*>(&address),sizeof(address))) { ::close(fd); return {}; }
    const auto text = "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: " +
        std::to_string(body.size())+"\r\n\r\n"+body;
    require(::send(fd,text.data(),text.size(),MSG_NOSIGNAL)==static_cast<ssize_t>(text.size()), "send");
    std::string wire; char buffer[4096];
    for (;;) {
        const auto n=::recv(fd,buffer,sizeof(buffer),0);
        if(n<=0) break;
        wire.append(buffer,static_cast<std::size_t>(n));
        if(wire.find("\r\n\r\n")!=std::string::npos) headers_seen.store(true);
    }
    ::close(fd); return wire;
}
unsigned free_port() {
    const int fd=::socket(AF_INET,SOCK_STREAM,0);
    require(fd>=0,"port socket");
    sockaddr_in address{}; address.sin_family=AF_INET; address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    require(!::bind(fd,reinterpret_cast<sockaddr*>(&address),sizeof(address)),"port bind");
    socklen_t size=sizeof(address);
    require(!::getsockname(fd,reinterpret_cast<sockaddr*>(&address),&size),"port name");
    const auto port=ntohs(address.sin_port); ::close(fd); return port;
}
void run(bool streaming, unsigned failure_kind) {
    Options options;
    options.http_bind="127.0.0.1"; options.http_port=free_port(); options.http_batch=1;
    options.http_model_id="test"; options.context_tokens=64; options.session_management_requested=true;
    options.state_cache_stats=std::make_shared<StateCacheStatsMirror>();
    auto tokenizer=serve::BpeTokenizer::from_parts({"a","b","<|endoftext|>"},{1,1,3},{},"qwen35");
    const auto model=ModelContract::qwen38_flash_next();
    CompiledSerialExecution compiled;
    QualifiedMtpDraftPlan qualified; ActivatedMtpDraftPlan activated;
    std::vector<std::unique_ptr<MtpBatchRunner::Slot>> slots;
    slots.push_back(std::make_unique<MtpBatchRunner::Slot>());
    MtpBatchRunner runner(options,model,compiled,qualified,activated,{},std::move(slots));
    runner.session_run=[&](const MtpServeRequest& job) {
        MtpServeResult result;
        if(!job.session_operation.empty()) return result;
        if(job.on_admitted) job.on_admitted();
        if(streaming && job.session_id!="healthy") {
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
            while(!headers_seen.load() && std::chrono::steady_clock::now()<deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            require(headers_seen.load(),"post-admission headers");
        }
        if(job.session_id=="local") {
            OutputLogitsChain chain;chain.append(std::nullopt);
            require(!chain.complete(1),"missing row must fail the request");
            result.finish="diagnostic_failure";result.error=chain.error;return result;
        }
        if(job.session_id=="fatal") {
            if(failure_kind == 1U) throw 42;
            if(failure_kind == 2U) throw std::runtime_error("");
            throw std::runtime_error("injected execution failure");
        }
        result.finish="stop"; return result;
    };
    std::atomic<bool> stopped{false}; std::string failure;
    auto serve=make_http_batch_serve(options,"test",{}, {},tokenizer);
    std::thread server([&] {
        try { serve(runner); } catch(const std::exception& e) { failure=e.what(); }
        stopped.store(true);
    });
    // Always join on assertions too; CTest's timeout bounds the original hang.
    struct Join { std::thread& thread; ~Join(){http_serve_stop_requested().store(true); thread.join();} } join{server};
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    std::string healthy;
    while(healthy.empty() && !stopped.load() && std::chrono::steady_clock::now()<deadline) {
        healthy=request(options.http_port,R"({"prompt":[0],"session":"healthy","max_tokens":1})");
        if(healthy.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    require(healthy.find("200 OK")!=std::string::npos,"batch server ready");
    const auto body=[&](const char* id) {
        return std::string("{\"prompt\":[0],\"max_tokens\":1,\"session\":\"")+id+"\",\"stream\":"+(streaming?"true":"false")+"}";
    };
    headers_seen.store(false);
    const auto local=request(options.http_port,body("local"));
    require(local.find("request final missing output logits row")!=std::string::npos,"local exception reaches client");
    require(request(options.http_port,body("healthy")).find("200 OK")!=std::string::npos,"following request succeeds");
    headers_seen.store(false);
    const auto fatal=request(options.http_port,body("fatal"));
    require(fatal.find(failure_kind ? "engine failure" : "injected execution failure")!=std::string::npos,"fatal exception reaches client");
    const auto shutdown=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    while(!stopped.load() && std::chrono::steady_clock::now()<shutdown)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    require(stopped.load() && !failure.empty(),"fatal shutdown completes within bound");
}
}
int main() try {
    receipt_compatibility();
    // Explicitly disable every constructor option that can query a device.
    ::setenv("STRIX_BATCH_SERVICE_MS","0",1);
    ::setenv("STRIX_BATCH_CANCEL_POINTS","0",1);
    ::setenv("STRIX_BATCH_NEST_ROWS","0",1);
    for(bool stream:{false,true}) for(unsigned kind:{0U,1U,2U}) run(stream,kind);
    std::cout<<"batch HTTP session failures PASS\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
