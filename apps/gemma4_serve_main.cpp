#include "gemma4/engine/runtime.hpp"
#include <iostream>

int main(int argc, char** argv) {
    namespace e = strix::gemma4::engine;
    try {
        const auto options = e::parse_options(argc,argv);
        if (options.help) {
            std::cout << "strix-gemma4-serve MODEL --rope-tables TABLE_DIR --tokenizer tokenizer.json [--bind 127.0.0.1 --port 8080] [--dflash on|off|auto] [--draft-mtp DIR | --draft DRAFT.gguf --draft-rope TABLE.g4dr | --draft-eagle FILE] [--threads 1..16]\n"
                "drafters: --draft-mtp DIR (Gemma-4-E4B-it MTP assistant: a drafter-mtp/ package or a model dir containing one; recommended),\n"
                "  --draft DRAFT.gguf --draft-rope TABLE.g4dr (DFlash), --draft-eagle EAGLE.gguf (e4b-eagle fallback)\n"
                "env: STRIX_G4_DRAFT_KIND=dflash|mtp|eagle (how --draft is read), STRIX_G4_MTP_TOPK=1..64 (clusters, default 64),\n"
                "  STRIX_G4_MTP_Q8=1 (Q8_0 assistant matrices), STRIX_G4_MTP_GRAPHS=0 (no HIP graph replay), STRIX_G4_DFLASH=0 (no drafting)\n"
                "profiles: --profile default|fast|quality (default: default; explicit options and env win); the MTP drafter is found at\n"
                "  <model dir>/drafter-mtp unless --draft-mtp DIR is given; --draft off or STRIX_G4_DRAFT=0 disables it\n";
            return 0;
        }
        e::Runtime runtime(options);
        if (options.inspect) { std::cout << e::effective_config(*runtime.backend,options.request,bool(runtime.drafts)).dump() << '\n'; return 0; }
        if (!runtime.tokenizer) throw std::invalid_argument("serve requires --tokenizer");
        e::Api api(*runtime.backend,runtime.tokenizer,runtime.drafts,options.request.dflash,"nemotron-bolt-e4b",options.request);
        strix::serve::HttpServerConfig config; config.bind_address = options.bind; config.port = options.port;
        strix::serve::HttpServer server(config,[&](const auto& request, auto& writer) { api.handle(request,writer); });
        server.listen(); std::cerr << "Gemma 4 service ready\n"; server.serve();
        return 0;
    } catch (const std::exception& error) { std::cerr << "gemma4 serve: " << error.what() << '\n'; return 1; }
}
