#include "test.hpp"
#include "gemma4/ref/target.hpp"
#include <algorithm>
#include <cstring>
#include <random>

using namespace strix::gemma4;
namespace {
ModelManifest tiny(bool aliases, WeightFormat format = WeightFormat::f32) {
    ModelManifest m;
    auto& c = m.config;
    c.hidden = 32; c.layers = aliases ? 6 : 2; c.inter = 48; c.vocab = 37; c.heads = 4; c.kv_heads = 1;
    if (format != WeightFormat::f32) { c.hidden = 256; c.inter = 256; }
    c.ple_dim = 16; c.window = 8; c.max_positions = 64;
    for (unsigned l = 0; l < c.layers; ++l) c.layer.push_back({l%2 ? AttentionKind::global : AttentionKind::sliding,8,l < 2 ? l : l%2});
    std::mt19937 random(471);
    for (const auto& [name, shape] : required_tensors(c)) {
        const auto outputs = shape.size() == 2 ? std::uint32_t(shape[0]) : 1U, inputs = std::uint32_t(shape.back());
        auto values = std::make_shared<std::vector<float>>(std::size_t(outputs)*inputs);
        for (auto& x : *values) {
            x = float(int(random()%127)-63)*0.0009765625F;
            if (name.find("norm") != std::string::npos) x = 0.5F + x;
            if (name.ends_with("layer_scalar")) x = 0.875F;
        }
        if (shape.size() == 2 && inputs == 256 && format != WeightFormat::f32) {
            const auto stride = packed_row_bytes(format, inputs);
            auto packed = std::make_shared<std::vector<std::uint8_t>>(std::size_t(outputs) * stride);
            auto* data = packed->data();
            const auto write16 = [](std::uint8_t* dst, std::uint16_t bits) {
                dst[0] = static_cast<std::uint8_t>(bits);
                dst[1] = static_cast<std::uint8_t>(bits >> 8);
            };
            if (format == WeightFormat::bf16) {
                for (std::size_t i = 0; i < values->size(); ++i) write16(data + 2 * i, bf16_bits((*values)[i]));
            } else {
                for (auto& byte : *packed) byte = static_cast<std::uint8_t>(random());
                for (std::uint32_t o = 0; o < outputs; ++o) {
                    auto* row = data + std::size_t(o) * stride;
                    if (format == WeightFormat::q8_0) {
                        for (unsigned k = 0; k < inputs / 32; ++k) write16(row + 34 * k, 0x1800);
                    } else if (format == WeightFormat::q4_k) {
                        write16(row, 0x1000);
                        write16(row + 2, 0x0800);
                    } else {
                        write16(row + 208, 0x0800);
                    }
                }
            }
            m.tensors.push_back({name,shape,{data,outputs,inputs,stride,format},{}});
            m.storage.push_back(packed);
            continue;
        }
        m.tensors.push_back({name,shape,{reinterpret_cast<const std::uint8_t*>(values->data()),outputs,inputs,inputs*4ULL,WeightFormat::f32},{}});
        m.storage.push_back(values);
    }
    return m;
}
std::vector<float> ingest(HostTarget& target, HostStream& stream, std::size_t count, std::uint32_t token, bool permute = false, std::vector<float>* taps = nullptr) {
    auto tx = stream.begin(); std::vector<RowPlan> plans;
    const auto p = stream.frontier();
    for (std::size_t i = 0; i < count; ++i) plans.push_back({0,0,p+i,(token+std::uint32_t(i))%37,tx.get(),i,true});
    if (permute) std::reverse(plans.begin(),plans.end());
    const auto result = target.forward_rows(plans); tx->commit(count);
    if (taps) *taps = result.taps;
    return result.logits;
}

void compare_cpu_linear(const RopeTables& tables) {
    for (const auto format : {WeightFormat::f32, WeightFormat::bf16, WeightFormat::q8_0, WeightFormat::q4_k, WeightFormat::q6_k}) {
        const auto model = tiny(false, format);
        HostTarget current(bind_host_weights(model), tables, 2);
        unsigned tested = 0;
        HostTarget legacy(bind_host_weights(model), tables, 1,
            [&](LinearView w, const float* x, std::size_t rows, float* out, const ParallelOutputs& parallel) {
                if (w.format == format) ++tested;
                // The original host linear loop and declared reduction are the independent oracle.
                parallel(w.outputs, [&](std::size_t begin, std::size_t end) {
                    std::vector<float> decoded(w.inputs);
                    for (auto o = begin; o < end; ++o) {
                        for (std::uint32_t i = 0; i < w.inputs; ++i) decoded[i] = weight_at(w, static_cast<std::uint32_t>(o), i);
                        for (std::size_t r = 0; r < rows; ++r)
                            out[r * w.outputs + o] = chunk16_dot([&](std::uint32_t i) { return decoded[i]; }, x + r * w.inputs, w.inputs);
                    }
                });
                return true;
            });
        HostStream current_stream(model.config), legacy_stream(model.config);
        for (const auto count : {1U, 8U, 1U}) {
            auto curr_tx = current_stream.begin();
            auto old_tx = legacy_stream.begin();
            std::vector<RowPlan> plans;
            for (unsigned i = 0; i < count; ++i) plans.push_back({0,0,current_stream.frontier()+i,i+2,curr_tx.get(),i,true,true});
            const auto got = current.forward_rows(plans);
            for (auto& plan : plans) plan.transaction = old_tx.get();
            const auto want = legacy.forward_rows(plans);
            curr_tx->commit(count);
            old_tx->commit(count);
            CHECK(std::memcmp(got.logits.data(), want.logits.data(), got.logits.size() * sizeof(float)) == 0);
            CHECK(!got.hidden.empty());
            CHECK(std::memcmp(got.hidden.data(), want.hidden.data(), got.hidden.size() * sizeof(float)) == 0);
            CHECK(current_stream.digest() == legacy_stream.digest());
        }
        CHECK(tested > 0);
    }
}
}
int main() {
    try {
        std::vector<float> sliding(64*256), global(64*512);
        // Exact identity rotations suffice to test row and cache arithmetic independently of tables.
        for (std::size_t i = 0; i < sliding.size(); i+=2) sliding[i] = 1;
        for (std::size_t i = 0; i < global.size(); i+=2) global[i] = 1;
        const RopeTables tables{sliding,global,64,"synthetic-identity-rotation"};
        compare_cpu_linear(tables);
        for (const bool aliases : {false,true}) {
            const auto model = tiny(aliases);
            HostTarget target(bind_host_weights(model),tables,2), single_worker(bind_host_weights(model),tables,1);
            HostStream batch(model.config), serial(model.config), one(model.config);
            for (unsigned chunk = 0; chunk < 4; ++chunk) {
                std::vector<float> batch_taps;
                const auto logits = ingest(target,batch,8,chunk+2,true,&batch_taps);
                for (unsigned r = 0; r < 8; ++r) {
                    std::vector<float> solo_taps;
                    const auto solo = ingest(target,serial,1,chunk+2+r,false,&solo_taps);
                    const auto same = ingest(single_worker,one,1,chunk+2+r);
                    CHECK(std::memcmp(solo.data(),same.data(),solo.size()*4) == 0);
                    CHECK(std::memcmp(logits.data()+r*model.config.vocab,solo.data(),solo.size()*4) == 0);
                    CHECK(std::memcmp(batch_taps.data()+r*5*model.config.hidden,solo_taps.data(),solo_taps.size()*4) == 0);
                }
                CHECK(batch.digest() == serial.digest()); CHECK(batch.digest() == one.digest());
            }
            for (std::size_t prefix = 0; prefix <= 8; ++prefix) {
                HostStream candidate(model.config), reference(model.config);
                (void)ingest(target,candidate,7,2); (void)ingest(target,reference,7,2);
                const auto before = candidate.digest();
                auto tx = candidate.begin();
                refuses([&] { (void)candidate.begin(); });
                std::vector<RowPlan> rows;
                for (unsigned r = 0; r < 8; ++r) rows.push_back({0,0,7+r,3+r,tx.get(),r,true});
                const auto out = target.forward_rows(rows);
                CHECK(candidate.digest() == before); // rings never receive speculative writes
                tx->commit(prefix); refuses([&] { tx->commit(prefix); });
                for (std::size_t r = 0; r < prefix; ++r) (void)ingest(target,reference,1,3+std::uint32_t(r));
                CHECK(candidate.digest() == reference.digest());
                const auto a = ingest(target,candidate,1,17), b = ingest(target,reference,1,17);
                CHECK(a == b); CHECK(candidate.digest() == reference.digest());
                if (aliases) {
                    bool nonzero = false;
                    for (auto x : out.taps) nonzero |= x != 0;
                    CHECK(nonzero); // after block five, including PLE/scalar
                }
            }
            HostStream a(model.config), b(model.config);
            auto ta = a.begin(), tb = b.begin();
            RowPlan peers[] = {{0,0,0,2,ta.get(),1,true},{999,0,999,999,nullptr,0,false},{1,0,0,3,tb.get(),0,true}};
            const auto out = target.forward_rows(peers); ta->commit(1); tb->commit(1);
            HostStream expected(model.config);
            const auto solo = ingest(target,expected,1,2);
            CHECK(std::memcmp(out.logits.data()+model.config.vocab,solo.data(),solo.size()*4) == 0);
            CHECK(a.digest() == expected.digest());
            const auto before = a.digest();
            { auto cancelled = a.begin(); RowPlan row{0,0,1,4,cancelled.get(),0,true}; (void)target.forward_rows({&row,1}); }
            CHECK(a.digest() == before);
        }
        std::cout << "rows: R=8/serial, worker count, rings, global, aliases, taps, prefix rollback, peers passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
