#pragma once
#include "strix/ds41_prepared_load.hpp"
#include "strix/ds41_expert_tp.hpp"
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace strix::ds41::prepared {
// Default-discovered adapter; legacy loading remains the authority and overwrites the complete tensor
// after ANY cache failure, even when some validated chunks were already uploaded.
class HipLoader {
public:
    HipLoader(const std::filesystem::path& model, const std::filesystem::path& overlay,
              const deepseek41::TargetModel& target, unsigned rank, unsigned world, bool tp,
              const std::filesystem::path* trellis = nullptr, const std::string* sites = nullptr) : rank_(rank) {
        const char* directory = std::getenv("STRIX_DS41_PREPARED_LOAD");
        if ((directory && std::strcmp(directory, "0") == 0) || world != 2) return;
        const bool automatic = !directory || !*directory;
        try {
            const auto root = automatic ? default_root() : std::filesystem::path{};
            auto dense = dense_from_env();
            // a single pack directory replaces the environment's; a mix spec (class=dir,...,default=dir: the forward's
            // trellis_env_spec()) is already bound by STRIX_DS41_TRELLIS + STRIX_DS41_TRELLIS_MIX as the tool prepared it
            if (trellis && trellis->string().find('=') == std::string::npos) dense.trellis = trellis->string();
            if (sites) dense.sites = *sites;
            std::string ns;
            const auto expected = [&] {
                if (ns.empty()) ns = name_space(model, overlay, target, rank, world, tp, expert_tp::packed_from_env(), dense);
                return ns;
            };
            const auto selected = select_directory(root, directory, expected);
            if (selected.empty()) {
                std::fprintf(stderr, "prepared rank %u no matching pack; legacy fallback\n", rank_); return;
            }
            reader_ = std::make_unique<Reader>(selected, expected());
        }
        catch (const std::exception& e) {
            if (automatic) std::fprintf(stderr, "prepared rank %u no matching pack; legacy fallback\n", rank_);
            else std::fprintf(stderr, "prepared rank %u REJECT %s; legacy fallback\n", rank_, e.what());
        }
    }
    ~HipLoader() { if (pinned_) (void)hipHostFree(pinned_); }
    bool active() const noexcept { return bool(reader_); }
    bool load(const std::string& name, std::uint64_t bytes, std::uint8_t* dst, hipStream_t stream) {
        if (!reader_) return false;
        try {
            if (!pinned_) {
                const auto pin_start = std::chrono::steady_clock::now();
                if (hipHostMalloc(reinterpret_cast<void**>(&pinned_), staging_slots * chunk_bytes + 4096, hipHostMallocDefault) != hipSuccess)
                    throw std::runtime_error("pin allocation failed");
                pin_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - pin_start).count();
            }
            // Pin once before I/O, then read directly into page-aligned pinned slots: no
            // extra model-sized copy and no per-chunk host registration on restart.
            Staging staging;
            const auto aligned = (reinterpret_cast<std::uintptr_t>(pinned_) + 4095U) & ~std::uintptr_t{4095U};
            for (std::size_t i = 0; i < staging.size(); ++i)
                staging[i] = {reinterpret_cast<std::uint8_t*>(aligned) + i * chunk_bytes, chunk_bytes};
            const auto start = std::chrono::steady_clock::now();
            reader_->load(name, bytes, [&](Bytes part, std::uint64_t offset) {
                // Bounded parallel read/verify jobs overlap this upload; verified chunks stay ordered.
                const auto upload_start = std::chrono::steady_clock::now();
                if (hipMemcpyAsync(dst + offset, part.data(), part.size(), hipMemcpyHostToDevice, stream) != hipSuccess ||
                    hipStreamSynchronize(stream) != hipSuccess) throw std::runtime_error("prepared upload failed");
                h2d_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - upload_start).count();
            }, &stats_, staging);
            seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            ++hits_; bytes_ += bytes;
            return true;
        } catch (const std::exception& e) {
            (void)hipStreamSynchronize(stream);
            std::fprintf(stderr, "prepared rank %u REJECT %s: %s; complete tensor legacy fallback\n", rank_, name.c_str(), e.what());
            reader_.reset(); return false;
        }
    }
    void report() const {
        if (hits_) std::fprintf(stderr, "prepared rank %u tensors %u bytes %llu read-verify-pin-H2D_s %.6f staging_bytes %zu read_s %.6f verify_s %.6f pin_alloc_s %.6f H2D_s %.6f direct %u sha256_backend %s\n",
            rank_, hits_, static_cast<unsigned long long>(bytes_), seconds_, staging_slots * chunk_bytes + 4096, stats_.read_s, stats_.verify_s, pin_s_, h2d_s_, stats_.direct ? 1U : 0U, digest_backend());
    }
private:
    unsigned rank_ = 0, hits_ = 0;
    std::uint64_t bytes_ = 0;
    double seconds_ = 0, pin_s_ = 0, h2d_s_ = 0;
    LoadStats stats_;
    std::unique_ptr<Reader> reader_;
    std::uint8_t* pinned_ = nullptr;
};
// Diagnostic only. Hash downloaded device bytes (including dense storage) in bounded chunks.
// Per-tensor digest is SHA256(concatenated ASCII chunk SHA256s); never timed as a load benchmark.
template<class Weights>
void device_hashes(const Weights& weights, unsigned rank, hipStream_t stream) {
    const char* flag = std::getenv("STRIX_DS41_WEIGHT_HASHES");
    if (!flag || *flag != '1') return;
    if (hipStreamSynchronize(stream) != hipSuccess) throw std::runtime_error("weight hash stream drain failed");
    std::vector<std::uint8_t> host(staging_slots * chunk_bytes);
    Staging staging;
    for (std::size_t i = 0; i < staging.size(); ++i) staging[i] = {host.data() + i * chunk_bytes, chunk_bytes};
    for (const auto& [name, d] : weights) {
        const auto hash = chunked_digest(d.n, [&](std::span<std::uint8_t> part, std::uint64_t offset) {
            if (hipMemcpy(part.data(), d.p + offset, part.size(), hipMemcpyDeviceToHost) != hipSuccess)
                throw std::runtime_error("weight hash readback failed");
        }, staging);
        std::fprintf(stderr, "WEIGHT rank %u %s bytes %zu sha256_chunks %s\n", rank, name.c_str(), d.n, hash.c_str());
    }
}
} // namespace strix::ds41::prepared
