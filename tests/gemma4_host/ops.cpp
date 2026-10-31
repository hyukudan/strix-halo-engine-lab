#include "test.hpp"
#include "gemma4/contracts/g4_identity.hpp"
#include "gemma4/model/model.hpp"
#include <array>
#include <fstream>
#include <limits>
#include <vector>

using namespace strix::gemma4;
int main() {
    try {
        std::ifstream f(std::string(G4_FIXTURE_DIR) + "/rsqrt.bin", std::ios::binary);
        CHECK(f.good()); std::uint32_t pair[2]; std::size_t count = 0;
        while (f.read(reinterpret_cast<char*>(pair), sizeof(pair))) {
            CHECK(bit_cast<std::uint32_t>(pow32_rsqrt(bit_cast<float>(pair[0]))) == pair[1]); ++count;
        }
        CHECK(count >= 8000);
        CHECK(pow32_rsqrt(4.0F) == 0.5F); CHECK(std::isnan(pow32_rsqrt(-1.0F)));
        CHECK(bit_cast<std::uint32_t>(pow32_rsqrt(-0.0F)) == 0xff800000U);
        CHECK(pow32_rsqrt(std::numeric_limits<float>::infinity()) == 0.0F);
        CHECK(bf16_value(0x3f80) == 1); CHECK(fp16_value(0x0001) == 0x1p-24F);
        CHECK(fp16_value(0x8000) == 0 && std::signbit(fp16_value(0x8000)));
        CHECK(bf16_bits(bit_cast<float>(0x3f808000U)) == 0x3f80);
        CHECK(bf16_bits(bit_cast<float>(0x3f818000U)) == 0x3f82);
        const float embed = bit_cast<float>(0x3d01abcdU);
        CHECK(embedding_value(embed) == bf16_value(bf16_bits(bf16_value(bf16_bits(embed)) * 50.5F)));
        CHECK(embedding_value(embed) != embed * 50.5F);
        CHECK(embedding_value(embed, 16) == bf16_value(bf16_bits(bf16_value(bf16_bits(embed)) * 16)));
        CHECK(better(2, 4, 2, 5)); CHECK(!better(2, 5, 2, 4));
        CHECK(softcap(1000) == 30 && softcap(2000) == 30);
        CHECK(std::isnan(gelu_tanh(0x1p+100F)));
        CHECK(std::isnan(declared_tanh(std::numeric_limits<float>::max())));
        CHECK(bit_cast<std::uint32_t>(declared_tanh(-0.0F)) == 0x80000000U);
        CHECK(declared_exp(0) == 1); CHECK(declared_exp(-104) == 0); CHECK(std::isinf(declared_exp(89)));
        std::ifstream nonlinear(std::string(G4_FIXTURE_DIR)+"/nonlinear.bin",std::ios::binary);
        CHECK(nonlinear.good()); float record[4]; unsigned records = 0;
        while (nonlinear.read(reinterpret_cast<char*>(record),sizeof(record))) {
            CHECK(bit_cast<std::uint32_t>(declared_exp(record[0])) == bit_cast<std::uint32_t>(record[1]));
            CHECK(bit_cast<std::uint32_t>(declared_tanh(record[0])) == bit_cast<std::uint32_t>(record[2]));
            CHECK(bit_cast<std::uint32_t>(gelu_tanh(record[0])) == bit_cast<std::uint32_t>(record[3])); ++records;
        }
        CHECK(records == 14);
        std::ifstream quant(std::string(G4_FIXTURE_DIR) + "/quant.bin",std::ios::binary);
        CHECK(quant.good());
        for (unsigned format = 0; format < 3; ++format) {
            const auto bytes = format == 0 ? 34U : (format == 1 ? 144U : 210U), width = format == 0 ? 32U : 256U;
            std::vector<std::uint8_t> blocks;
            std::vector<float> all;
            for (unsigned block = 0; block < 2; ++block) {
                std::vector<std::uint8_t> b(bytes); std::vector<float> expected(width);
                quant.read(reinterpret_cast<char*>(b.data()),bytes); quant.read(reinterpret_cast<char*>(expected.data()),width*4);
                CHECK(quant.good());
                blocks.insert(blocks.end(),b.begin(),b.end()); all.insert(all.end(),expected.begin(),expected.end());
                for (unsigned i = 0; i < width; ++i) {
                    const auto value = format == 0 ? q8_0_value(b.data(),i) : (format == 1 ? q4_k_value(b.data(),i) : q6_k_value(b.data(),i));
                    CHECK(bit_cast<std::uint32_t>(value) == bit_cast<std::uint32_t>(expected[i]));
                }
            }
            const auto kind = format == 0 ? WeightFormat::q8_0 : (format == 1 ? WeightFormat::q4_k : WeightFormat::q6_k);
            const LinearView wide{blocks.data(),1,width*2,bytes*2,kind}, rows{blocks.data(),2,width,bytes,kind};
            for (unsigned i = 0; i < all.size(); ++i) {
                CHECK(weight_at(wide,0,i) == all[i]); CHECK(weight_at(rows,i/width,i%width) == all[i]);
            }
        }
        std::vector<float> a(1031), b(1031), lane(32), weight(1031,2), result(1031);
        for (unsigned i = 0; i < a.size(); ++i) { a[i] = float(int(i%17)-8)*0.03125F; b[i] = float(int(i%23)-11)*0.0625F; }
        for (unsigned l = 0; l < 32; ++l)
            for (unsigned chunk = l; chunk < (a.size()+15)/16; chunk += 32)
                for (unsigned e = 0; e < 16; ++e) { const auto i = chunk*16+e; lane[l] = lane[l] + (i < a.size() ? a[i]*b[i] : 0); }
        for (unsigned distance : {16,8,4,2,1}) {
            auto old = lane;
            for (unsigned l = 0; l < 32; ++l) lane[l] = old[l] + old[l^distance];
        }
        CHECK(chunk16_dot([&](unsigned i) { return a[i]; },b.data(),std::uint32_t(b.size())) == lane[0]);
        rms_norm(a.data(), weight.data(),result.data(),std::uint32_t(a.size()));
        const auto r = pow32_rsqrt(lane_dot(a.data(),a.data(),std::uint32_t(a.size()))/float(a.size())+norm_eps);
        for (unsigned i = 0; i < a.size(); ++i) CHECK(result[i] == (a[i]*r)*2);
        std::array<float,32> edge{}, normalized{};
        rms_norm(edge.data(),nullptr,normalized.data(),32);
        for (auto x : normalized) CHECK(bit_cast<std::uint32_t>(x) == 0);
        edge.fill(0x1p-130F); rms_norm(edge.data(),nullptr,normalized.data(),32);
        CHECK(normalized[0] > 0 && normalized[0] == edge[0]*pow32_rsqrt(norm_eps));
        edge.fill(0x1p+50F); rms_norm(edge.data(),nullptr,normalized.data(),32);
        CHECK(normalized[0] == 1.0F);
        edge.fill(0x1p+100F); rms_norm(edge.data(),nullptr,normalized.data(),32);
        CHECK(std::isnan(normalized[0]));
        std::vector<float> cs(256, 0.5F),sn(256,0.25F),global(512),old;
        for (unsigned i = 0; i < global.size(); ++i) global[i] = float(i);
        old = global; apply_rope(global.data(),512,64,cs.data(),sn.data());
        for (unsigned i = 64; i < 256; ++i) CHECK(global[i] == old[i] && global[i+256] == old[i+256]);
        CHECK(global[0] == old[0]*cs[0]-old[256]*sn[0]);
        std::ifstream rotary(std::string(G4_FIXTURE_DIR) + "/rope.bin",std::ios::binary);
        CHECK(rotary.good());
        for (unsigned position : {0,511,512,65535}) for (unsigned route = 0; route < 2; ++route) {
            const auto pairs = route ? 256U : 128U;
            std::vector<float> table(pairs*2); rotary.read(reinterpret_cast<char*>(table.data()),std::streamsize(table.size()*4));
            CHECK(rotary.good());
            for (unsigned i = 0; i < pairs; ++i) {
                if (position == 0 || (route && i >= 64)) CHECK(table[2*i] == 1 && bit_cast<std::uint32_t>(table[2*i+1]) == 0);
                const auto theta = route ? 1000000.0 : 10000.0;
                if (!route || i < 64) {
                    const auto power = float(std::pow(theta,double(float(2*i)/float(pairs*2))));
                    const auto angle = float(position)*(1.0F/power);
                    // Diagnostic independent libm agreement; the fixture itself is MPFR checked.
                    CHECK(std::abs(table[2*i]-float(std::cos(double(angle)))) <= 0x1p-23F);
                    CHECK(std::abs(table[2*i+1]-float(std::sin(double(angle)))) <= 0x1p-23F);
                }
            }
        }
        std::cout << "operators: MPFR rsqrt, packed blocks, norm, embedding, rotary, ties passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
