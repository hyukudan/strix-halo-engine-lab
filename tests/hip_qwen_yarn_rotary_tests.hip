#include "hip_qwen_rotary_watchdog.hpp"
#include "strix/hip_qwen_rotary_diagnostic.hpp"
#include "strix/hip_mtp_resources.hpp"
#include "strix/mtp_sidecar_binding.hpp"
#include "rotary_mock.hpp"
#include "strix/hip_qwen_rotary_table.hpp"
#include <hip/hip_runtime.h>
#include <bit>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <random>
#include "strix/model_content_hash.hpp"
#include <stdexcept>

namespace {
float bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xffff0000U);
}
void equal(const std::vector<float>& a, const std::vector<float>& b, const char* message) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)))
        throw std::runtime_error(message);
}
}
int main(int argc,char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, BUFSIZ);
    using namespace strix;
    using namespace qwen_rotary_diagnostic;
    try {
        if(argc==2 && std::string_view(argv[1])=="--watchdog-self-test") {
            site_watchdog().begin("watchdog-self-test", std::chrono::milliseconds(20));
            std::this_thread::sleep_for(std::chrono::seconds(10));
            return 0; // The test driver rejects this if the watchdog fails to exit.
        }
        site_watchdog().begin("host-admission");
        if(argc==2 && std::string_view(argv[1])=="--host-admission-only") {
            std::array<ModelContract,4> models;
            models[1].qwen_rotary_table=mock_rotary(2);
            models[2].qwen_rotary_table=mock_rotary(4);
            models[3].qwen_rotary_table=mock_rotary(2,2);
            for(const auto& producer:models) for(const auto& draft:models) {
                if(producer.rotary_identity()==draft.rotary_identity()) continue;
                bool refused=false;
                try { (void)HipMtpResources::create(draft,{}, {},nullptr,{},nullptr,{}, {},nullptr,producer.rotary_identity()); }
                catch(const std::invalid_argument& e) {refused=std::string_view(e.what()).find("rotary identity mismatch")!=std::string_view::npos;}
                if(!refused) throw std::runtime_error("MTP owner did not reject producer before device setup");
            }
            for(unsigned i=1;i<models.size();++i) {
                auto pending=models[i];pending.resolved_rotary=pending.rotary_identity();pending.qwen_rotary_table.reset();
                bool refused=false;
                try { (void)HipMtpResources::create(pending,{}, {},nullptr,{},nullptr,{}, {},nullptr,pending.rotary_identity()); }
                catch(const std::invalid_argument& e) {refused=std::string_view(e.what()).find("requires uploaded coefficient table")!=std::string_view::npos;}
                if(!refused) throw std::runtime_error("MTP owner admitted resolved identity without upload");
            }
            for(unsigned i=1;i<models.size();++i) {
                bool refused=false;
                try { validate_qwen_rotary_reader(*models[i].qwen_rotary_table,nullptr); }
                catch(const std::invalid_argument& e) {refused=std::string_view(e.what()).find("CPU rotary adoption")!=std::string_view::npos;}
                if(!refused) throw std::runtime_error("CPU adoption reached device metadata query");
            }
            std::cout<<"MTP producer and rotary allocation host admission passed (no HIP calls)\n";
            return 0;
        }
        site_watchdog().begin("device-discovery");
        int devices = 0;
        if (hipGetDeviceCount(&devices) != hipSuccess || !devices) return 77;
        std::mt19937 random(0x715ab3U);
        for (unsigned factor : {2U, 4U}) {
            site_watchdog().begin("full-upload");
            std::cout<<"begin site=full-upload factor="<<factor<<std::endl;
            const auto descriptor = RotaryDescriptor::yarn(factor);
            auto table_owner = std::make_shared<HipQwenRotaryTable>(descriptor);
            const auto& table = *table_owner;
            ModelContract model;
            model.qwen_rotary_table = table_owner;
            for (auto stream : {static_cast<hipStream_t>(nullptr), hipStreamLegacy, hipStreamPerThread}) {
                site_watchdog().begin("default-stream-admission");
                std::cout<<"begin site=default-stream-admission factor="<<factor<<" stream="<<stream<<std::endl;
                validate_qwen_rotary_reader(table, stream);
            }
            // Bounded full readback verifies every uploaded byte, not samples.
            site_watchdog().begin("full-upload");
            std::cout<<"begin site=full-upload factor="<<factor<<" readback=1"<<std::endl;
            Sha256Fast uploaded_hash;
            generate_qwen_yarn_rows(descriptor,0,table.extent(),[&](auto first,auto rows) {
                std::vector<YarnRow> uploaded(rows.size());
                if(hipMemcpy(uploaded.data(),static_cast<const YarnRow*>(table.data())+first,
                    rows.size_bytes(),hipMemcpyDeviceToHost)!=hipSuccess ||
                    std::memcmp(uploaded.data(),rows.data(),rows.size_bytes()))
                    throw std::runtime_error("full uploaded table byte verification failed");
                uploaded_hash.update(std::as_bytes(std::span(uploaded)));
            });
            if(uploaded_hash.finalize()!=table.sha256()) throw std::runtime_error("uploaded table digest mismatch");
            std::cout<<"site=full-upload factor="<<factor<<" positions_checked="<<table.extent()
                <<" bytes_compared="<<descriptor.table_bytes()<<'\n';
            site_watchdog().begin("device-admission");
            std::cout<<"begin site=device-admission factor="<<factor<<std::endl;
            device_admission_probes(model);
            site_watchdog().begin("publication");
            std::cout<<"begin site=publication factor="<<factor<<std::endl;
            publication_probes(model);
            site_watchdog().begin("rejection");
            std::cout<<"begin site=rejection factor="<<factor<<std::endl;
            rejection_probes(model);
            site_watchdog().begin("multirow");
            std::cout<<"begin site=multirow factor="<<factor<<std::endl;
            multirow_probes(model);
            std::vector<std::uint64_t> positions{0,1,8191,8192,32767,65535,131071,196607,262143,262144,524287};
            if(factor==4U) {positions.push_back(524288);positions.push_back(999999);}
            if (factor == 4U) positions.push_back(1048575);
            for (unsigned i = 0; i < 16U; ++i) positions.push_back(random() % table.extent());
            std::array<std::uint64_t,static_cast<unsigned>(Site::count)> compared{};
            constexpr std::array<const char*,static_cast<unsigned>(Site::count)> names{
                "query","key","index-query","dynamic-query","known-rows","grouped","wave","batch-grouped",
                "batch-wave","pooled","batch-pooled","score","known-score","mtp-norm","mtp-score"};
            for (auto position : positions) for (unsigned s = 0; s < static_cast<unsigned>(Site::count); ++s) {
                site_watchdog().begin(names[s]);
                std::cout<<"begin site="<<names[s]<<" factor="<<factor<<" position="<<position<<std::endl;
                const auto site = static_cast<Site>(s);
                const bool pooled = site == Site::pooled || site == Site::batch_pooled;
                const bool score = site == Site::score || site == Site::known_score || site == Site::mtp_score;
                const unsigned dim = (pooled || score || site == Site::index_query) ? 128U : 256U;
                const unsigned size = site == Site::query ? 2U * dim : dim;
                const auto coefficient_position = static_cast<std::uint32_t>((pooled || score) ? (position / 4U) * 4U : position);
                const auto cs = qwen_yarn_row(descriptor, coefficient_position);
                if (s == 0U) {
                    YarnRow uploaded{};
                    const auto* from = static_cast<const YarnRow*>(table.data()) + position;
                    if (hipMemcpy(&uploaded, from, sizeof(uploaded), hipMemcpyDeviceToHost) != hipSuccess ||
                        std::memcmp(&uploaded, &cs, sizeof(cs)))
                        throw std::runtime_error("uploaded coefficient bytes differ from B2 generator");
                }
                std::vector<float> expected(size, 7.0F), normalized(dim);
                for (unsigned i = 0; i < dim; ++i)
                    normalized[i] = bf16(((i % 3U) ? 1.0F : -1.0F) * static_cast<float>(1U + i % 7U) / 8.0F);
                for (unsigned i = 0; i < dim; ++i) {
                    float value = normalized[i];
                    if (i < 64U) {
                        const auto pair = i % 32U;
                        const auto partner = i < 32U ? i + 32U : i - 32U;
                        // The launcher is compiled with HIP's default FP contraction.
                        value = std::fma(normalized[partner], i < 32U ? -cs[pair].sine : cs[pair].sine,
                            normalized[i] * cs[pair].cosine);
                    }
                    expected[i] = bf16(value);
                }
                if (score) {
                    // Match the kernel's FP32 multiply and LDS reduction tree,
                    // then its one-head ReLU and scalar normalization.
                    for (unsigned i = 0; i < dim; ++i)
                        expected[i] *= static_cast<float>(1U + i % 5U) / 8.0F;
                    for (unsigned step = dim / 2U; step; step >>= 1U)
                        for (unsigned i = 0; i < step; ++i) expected[i] += expected[i + step];
                    expected = {std::fmax(expected[0], 0.0F) / std::sqrt(static_cast<float>(dim))};
                }
                const auto got = run(site, model, position);
                try { equal(got, expected, "YaRN site differs from coefficient/BF16 CPU reference"); }
                catch (...) { std::cerr << "factor " << factor << " position " << position << " site " << s << '\n'; throw; }
                compared[s]+=2U*got.size()*sizeof(float);
                ModelContract native;
                site_watchdog().begin(names[s]);
                std::cout<<"begin site="<<names[s]<<" factor="<<factor<<" position="<<position<<" native-pair=1"<<std::endl;
                equal(run(site, native, position), run(site, native, position, true), "native differs from frozen current kernel");
            }
            for (unsigned s = 0; s < static_cast<unsigned>(Site::count); ++s) {
                site_watchdog().begin(names[s]);
                std::cout<<"begin site="<<names[s]<<" factor="<<factor<<" out-of-range=1"<<std::endl;
                bool refused = false;
                try { (void)run(static_cast<Site>(s), model, table.extent()); }
                catch (const std::out_of_range&) { refused = true; }
                if (!refused) throw std::runtime_error("out-of-range position was accepted");
            }
            for(unsigned s=0;s<static_cast<unsigned>(Site::count);++s)
                std::cout<<"site="<<names[s]<<" factor="<<factor<<" positions_checked="<<positions.size()
                    <<" bytes_compared="<<compared[s]<<'\n';
            table_owner.reset();
            site_watchdog().begin("lifetime");
            std::cout<<"begin site=lifetime factor="<<factor<<std::endl;
            lifetime_probes(std::move(model.qwen_rotary_table));
        }
        std::cout << "all rotary site fixtures passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
