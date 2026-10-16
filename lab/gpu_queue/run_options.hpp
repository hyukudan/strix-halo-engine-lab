#pragma once
#include "../gpu_visibility/run_options.hpp"
#include <vector>
namespace strix::transport::queue_lab {
struct Options {
    visibility_lab::RunOptions base;
    unsigned depth = 8, windows = 4, recovery_every = 0;
    bool full_duplex = false, configured_gpu_direct = false, controller_reset = false, include_network = false, independent_control = false;
};
inline Options parse(int argc, char** argv) {
    Options out; std::vector<char*> remaining{argv[0]};
    bool seen_depth=false, seen_windows=false, seen_recovery=false;
    for(int n=1;n<argc;++n) {
        const std::string_view key=argv[n];
        if(key=="--controller-reset" || key=="--include-network" || key=="--independent-control") {
            bool& value=key=="--controller-reset"?out.controller_reset:key=="--include-network"?out.include_network:out.independent_control;
            if(value)throw std::invalid_argument("duplicate controller recovery option");
            value=true;continue;
        }
        if(key=="--configured-gpu-direct") { if(out.configured_gpu_direct)throw std::invalid_argument("duplicate route acknowledgment");out.configured_gpu_direct=true;continue; }
        if(key=="--full-duplex") { if(out.full_duplex)throw std::invalid_argument("duplicate duplex mode");out.full_duplex=true;continue; }
        if(key=="--queue-depth"||key=="--windows"||key=="--recover-every") {
            if(n+1>=argc)throw std::invalid_argument("missing bounded queue option");
            bool& seen=key=="--queue-depth"?seen_depth:key=="--windows"?seen_windows:seen_recovery;
            if(seen)throw std::invalid_argument("duplicate queue option");
            seen=true;
            const auto value=visibility_lab::bounded(argv[++n],key=="--recover-every"?0:key=="--queue-depth"?2:1,key=="--queue-depth"?8:128);
            if(key=="--queue-depth")out.depth=static_cast<unsigned>(value);else if(key=="--windows")out.windows=static_cast<unsigned>(value);else out.recovery_every=static_cast<unsigned>(value);
            continue;
        }
        if(key=="--iterations"||key=="--duplex-pairs")throw std::invalid_argument("queue runner uses --windows and --full-duplex, not C1 iteration options");
        remaining.push_back(argv[n]);
    }
    out.base=visibility_lab::parse(static_cast<int>(remaining.size()),remaining.data());
    const auto& b=out.base;
    if(!out.configured_gpu_direct||b.bytes>static_cast<std::uint64_t>((b.ring_frames-out.depth)/out.depth)*4096||
        b.bytes*out.depth*out.windows*(out.full_duplex?2ULL:1ULL)>128ULL*1024*1024||
        (out.recovery_every&&static_cast<std::uint64_t>((out.windows-1)/out.recovery_every)>UINT64_MAX-b.epoch))
        throw std::invalid_argument("queue geometry, total bytes, epoch budget or configured gpu_direct acknowledgment is invalid");
    if ((out.controller_reset && (!out.recovery_every || out.recovery_every>=out.windows)) ||
        (out.include_network && (!out.controller_reset || !out.independent_control)) ||
        (out.independent_control && !out.include_network))
        throw std::invalid_argument("controller recovery requires explicit recover-every and independent control for network outage");
    return out;
}
}
