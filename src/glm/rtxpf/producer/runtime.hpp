#pragma once
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <cstdint>
#include "verified_manifest.hpp"
namespace strix::glm::rtxpf::producer {
using StateSink=std::function<void(unsigned rank,unsigned layer,unsigned component,std::span<const std::byte>)>;
// Constructor is the first device work. The service validates the UUID and all
// host arguments before constructing it. No CPU emulation hides a CUDA failure.
class Runtime {
    struct Impl;std::unique_ptr<Impl> p_;
public:
    Runtime(const std::string& path,unsigned capacity,unsigned row_tile,bool from_shards=false,bool graphs=true,unsigned host_floor_gib=8,bool legacy_index=false,bool serial_stage=false,const VerifiedManifest* manifest=nullptr,unsigned max_device_layers=43,unsigned max_pinned_layers=43,bool require_file_tier=false,const std::string& weight_dirs="",unsigned drive_readers=16,int file_cache_mode=1);
    ~Runtime();
    std::string health()const;
    std::string model_sha256()const;
    std::string timings()const;
    void prefill(std::span<const std::uint32_t> tokens,const std::string& identity,const StateSink&);
    std::string self_check(std::span<const std::uint32_t> tokens,const std::string& identity,unsigned reference_tile=256);
};
}
