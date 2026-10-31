#include <hip/hip_runtime.h>
#include "gemma4/kernels/ops.hpp"
#include "gemma4/kernels/internal.hpp"
#include "gemma4/kv/storage.hpp"
#include "fixture.hpp"
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <string_view>

using namespace strix::gemma4;
using namespace strix::gemma4::gpu;
namespace {
template <class T> Buffer device(const std::vector<T>& values) {
    Buffer b(values.size() * sizeof(T));
    b.upload(values.data(), b.size());
    return b;
}
std::vector<float> read(const Buffer& b, std::size_t count) {
    std::vector<float> v(count);
    b.download(v.data(), count * sizeof(float));
    return v;
}
std::vector<float> inputs(std::size_t n) {
    std::vector<float> x(n);
    std::mt19937 rng(913);
    for (auto& a : x)
        a = float(int(rng() % 255) - 127) * 0.0078125F;
    return x;
}
__global__ void scalars(const float* x, float* out, unsigned n) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[4ULL * i] = pow32_rsqrt(x[i]);
        out[4ULL * i + 1] = declared_exp(x[i]);
        out[4ULL * i + 2] = declared_tanh(x[i]);
        out[4ULL * i + 3] = gelu_tanh(x[i]);
    }
}
__global__ void cooperative_scalars(const float* x, float* out, unsigned n, bool bounded, bool intervals) {
    const auto i = blockIdx.x * 4U + threadIdx.x / 32U;
    if (i < n) {
        const float r = wave_rsqrt(x[i], bounded, intervals);
        if (threadIdx.x % 32U == 0)
            out[i] = r;
    }
}
void scalar_tests() {
    std::vector<float> x = {
        0.0F,   -0.0F, bit_cast<float>(1U), bit_cast<float>(0x7fffffU), 1e-20F, 1e-10F, 1.0F, 2.0F, 30.0F, 88.0F, 89.0F,
        103.0F, 1e20F};
    std::mt19937 rng(334);
    for (unsigned exponent = 0; exponent < 255; ++exponent)
        for (unsigned mantissa : {0U, 1U, 0x7ffffeU, 0x7fffffU})
            x.push_back(bit_cast<float>((exponent << 23U) | mantissa));
    for (unsigned i = 0; i < 16384; ++i)
        x.push_back(bit_cast<float>(1U + unsigned(rng() % 0x7f7ffffeU)));
    std::vector<float> expected(x.size() * 4);
    for (std::size_t i = 0; i < x.size(); ++i) {
        expected[4 * i] = pow32_rsqrt(x[i]);
        expected[4 * i + 1] = declared_exp(x[i]);
        expected[4 * i + 2] = declared_tanh(x[i]);
        expected[4 * i + 3] = gelu_tanh(x[i]);
    }
    auto dx = device(x);
    Buffer out(expected.size() * 4);
    hipLaunchKernelGGL(scalars, dim3(unsigned((x.size() + 255) / 256)), dim3(256), 0, nullptr, dx.as<float>(),
                       out.as<float>(), unsigned(x.size()));
    synchronize();
    bytes(read(out, expected.size()), expected);
    std::vector<float> rsqrt(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        rsqrt[i] = expected[4 * i];
    for (bool bounded : {false, true})
        for (bool intervals : {false, true}) {
            hipLaunchKernelGGL(cooperative_scalars, dim3(unsigned((x.size() + 3) / 4)), dim3(128), 0, nullptr, dx.as<float>(),
                               out.as<float>(), unsigned(x.size()), bounded, intervals);
            synchronize();
            bytes(read(out, rsqrt.size()), rsqrt);
        }
}
void linear_tests() {
    for (const auto f :
         {WeightFormat::bf16, WeightFormat::f32, WeightFormat::q8_0, WeightFormat::q4_k, WeightFormat::q6_k}) {
        const std::vector<unsigned> widths = f == WeightFormat::bf16 || f == WeightFormat::f32
                                                 ? std::vector<unsigned>{1, 17, 511, 512, 513, 1024, 1280, 10240, 10752}
                                                 : std::vector<unsigned>{256, 512, 768, 1024, 1280, 2560, 10240, 10752};
        for (const auto width : widths) {
            const unsigned outputs = width >= 256U && width <= 4096U && width % 16U == 0 ? 33U : 5U;
            const auto packed = fixture::packed(f, outputs, width);
            auto dw = device(packed);
            LinearView host{packed.data(), outputs, width, packed_row_bytes(f, width), f}, w = host;
            w.data = dw.as<std::uint8_t>();
            for (unsigned rows = 1; rows <= 8; ++rows) {
                const auto x = inputs(std::size_t(rows) * width);
                auto dx = device(x);
                std::vector<float> expected(std::size_t(rows) * outputs);
                for (unsigned r = 0; r < rows; ++r)
                    for (unsigned o = 0; o < outputs; ++o)
                        expected[std::size_t(r) * outputs + o] =
                            chunk16_dot([&](unsigned i) { return weight_at(host, o, i); },
                                        x.data() + std::size_t(r) * width, width);
                Buffer out(expected.size() * 4);
                // Poison the destination; no output may depend on prior scratch or peer rows.
                std::vector<float> poison(expected.size(), bit_cast<float>(0x7fc01234U));
                out.upload(poison.data(), out.size());
                linear(w, dx.as<float>(), rows, out.as<float>());
                synchronize();
                bytes(read(out, expected.size()), expected);
                linear(w, dx.as<float>(), rows, out.as<float>(), false);
                synchronize();
                bytes(read(out, expected.size()), expected);
                Buffer failure(sizeof(unsigned));
                failure.zero();
                linear(w, dx.as<float>(), rows, out.as<float>(), true, nullptr, failure.as<unsigned>());
                synchronize();
                bytes(read(out, expected.size()), expected);
                unsigned flag = 1;
                failure.download(&flag, sizeof(flag));
                CHECK(flag == 0);
                if (rows > 1) {
                    auto changed = x;
                    std::fill_n(changed.begin() + width, width, 123.25F);
                    dx.upload(changed.data(), dx.size());
                    linear(w, dx.as<float>(), rows, out.as<float>());
                    synchronize();
                    const auto got = read(out, expected.size());
                    CHECK(std::memcmp(got.data(), expected.data(), outputs * 4) == 0);
                }
            }
        }
    }
    for (const auto f : {WeightFormat::bf16, WeightFormat::q8_0, WeightFormat::q4_k, WeightFormat::q6_k}) {
        for (unsigned width : {512U, 1280U, 2560U, 10240U}) {
            const auto a = fixture::packed(f, 33, width, 1), b = fixture::packed(f, 33, width, 2);
            auto da = device(a), db = device(b);
            LinearView ha{a.data(), 33, width, packed_row_bytes(f, width), f}, hb = ha;
            hb.data = b.data();
            auto ga = ha, gb = hb;
            ga.data = da.as<std::uint8_t>();
            gb.data = db.as<std::uint8_t>();
            for (unsigned rows = 1; rows <= 8; ++rows) {
                const auto x = inputs(std::size_t(rows) * width);
                auto dx = device(x);
                Buffer out(std::size_t(rows) * 33 * 4);
                std::vector<float> expected(std::size_t(rows) * 33);
                for (unsigned r = 0; r < rows; ++r)
                    for (unsigned o = 0; o < 33; ++o) {
                        const auto* row = x.data() + std::size_t(r) * width;
                        const float g = chunk16_dot([&](unsigned i) { return weight_at(ha, o, i); }, row, width),
                                    u = chunk16_dot([&](unsigned i) { return weight_at(hb, o, i); }, row, width);
                        expected[std::size_t(r) * 33 + o] = gelu_tanh(g) * u;
                    }
                gate_up(ga, gb, dx.as<float>(), rows, out.as<float>());
                synchronize();
                bytes(read(out, expected.size()), expected);
                Buffer failure(sizeof(unsigned));
                failure.zero();
                gate_up(ga, gb, dx.as<float>(), rows, out.as<float>(), true, nullptr, failure.as<unsigned>());
                synchronize();
                bytes(read(out, expected.size()), expected);
                unsigned flag = 1;
                failure.download(&flag, sizeof(flag));
                CHECK(flag == 0);
            }
        }
    }
}
void linear_failure_tests() {
    for (unsigned width : {32U, 512U, 2560U}) {
        const std::vector<float> weights(4ULL * width, 1.0F);
        auto dw = device(weights);
        const LinearView w{dw.as<std::uint8_t>(), 4, width, width * sizeof(float), WeightFormat::f32};
        Buffer out(8 * 4 * sizeof(float)), failure(sizeof(unsigned));
        for (float bad : {std::numeric_limits<float>::infinity(), bit_cast<float>(0x7fc01234U)}) {
            auto x = inputs(8ULL * width);
            x.back() = bad;
            auto dx = device(x);
            for (bool gate : {false, true})
                for (bool shared : {false, true}) {
                    failure.zero();
                    if (gate)
                        gate_up(w, w, dx.as<float>(), 8, out.as<float>(), shared, nullptr, failure.as<unsigned>());
                    else
                        linear(w, dx.as<float>(), 8, out.as<float>(), shared, nullptr, failure.as<unsigned>());
                    synchronize();
                    unsigned flag = 0;
                    failure.download(&flag, sizeof(flag));
                    CHECK(flag == 1);
                }
        }
    }
}
// Explicit diagnostic only: production matrix geometry, synthetic resident bytes.
// It is deliberately excluded from ctest and does not qualify model throughput.
void linear_bench(bool fused = true, bool small = false) {
    struct Shape {
        const char* name;
        WeightFormat format;
        unsigned outputs, inputs;
        bool gate;
    };
    std::vector<Shape> shapes = {{"q4_gate_up", WeightFormat::q4_k, 10240, 2560, true},
                                 {"q4_down", WeightFormat::q4_k, 2560, 10240, false},
                                 {"q6_down", WeightFormat::q6_k, 2560, 10240, false},
                                 {"q8_head", WeightFormat::q8_0, 262144, 2560, false},
                                 {"bf16_gate_up", WeightFormat::bf16, 10240, 2560, true},
                                 {"bf16_down", WeightFormat::bf16, 2560, 10240, false}};
    if (small)
        shapes = {{"q4_q", WeightFormat::q4_k, 2048, 2560, false},
                  {"q4_k", WeightFormat::q4_k, 512, 2560, false},
                  {"q4_o", WeightFormat::q4_k, 2560, 2048, false},
                  {"q4_ple_gate", WeightFormat::q4_k, 256, 2560, false},
                  {"q4_ple_projection", WeightFormat::q4_k, 2560, 256, false},
                  {"bf16_q", WeightFormat::bf16, 2048, 2560, false},
                  {"bf16_k", WeightFormat::bf16, 512, 2560, false},
                  {"bf16_o", WeightFormat::bf16, 2560, 2048, false}};
    for (const auto shape : shapes) {
        const auto row_bytes = packed_row_bytes(shape.format, shape.inputs);
        const auto seed = fixture::packed(shape.format, 1, shape.inputs);
        std::vector<std::uint8_t> packed(std::size_t(shape.outputs) * row_bytes);
        for (unsigned o = 0; o < shape.outputs; ++o)
            std::copy(seed.begin(), seed.end(), packed.data() + std::size_t(o) * row_bytes);
        auto weights = device(packed);
        LinearView view{weights.as<std::uint8_t>(), shape.outputs, shape.inputs, row_bytes, shape.format};
        // Two distinct resident matrices in the gate/up path.
        Buffer up;
        auto uv = view;
        if (shape.gate) {
            up = device(packed);
            uv.data = up.as<std::uint8_t>();
        }
        for (unsigned rows : {1U, 4U, 8U}) {
            const auto x = inputs(std::size_t(rows) * shape.inputs);
            auto dx = device(x);
            Buffer out(std::size_t(rows) * shape.outputs * sizeof(float));
            Buffer separate(shape.gate && !fused ? std::size_t(rows) * shape.outputs * sizeof(float) : 0);
            const auto run = [&] {
                if (shape.gate && !fused) {
                    linear(view, dx.as<float>(), rows, out.as<float>());
                    linear(uv, dx.as<float>(), rows, separate.as<float>());
                    gelu_product(out.as<float>(), separate.as<float>(), rows * shape.outputs);
                } else if (shape.gate)
                    gate_up(view, uv, dx.as<float>(), rows, out.as<float>());
                else
                    linear(view, dx.as<float>(), rows, out.as<float>());
            };
            for (unsigned warmup = 0; warmup < 5; ++warmup)
                run();
            synchronize();
            hipEvent_t begin = nullptr, end = nullptr;
            CHECK(hipEventCreate(&begin) == hipSuccess);
            CHECK(hipEventCreate(&end) == hipSuccess);
            std::vector<float> timings;
            for (unsigned sample = 0; sample < 5; ++sample) {
                CHECK(hipEventRecord(begin) == hipSuccess);
                for (unsigned repeat = 0; repeat < 20; ++repeat)
                    run();
                CHECK(hipEventRecord(end) == hipSuccess);
                CHECK(hipEventSynchronize(end) == hipSuccess);
                float ms = 0.0F;
                CHECK(hipEventElapsedTime(&ms, begin, end) == hipSuccess);
                timings.push_back(ms / 20.0F);
            }
            CHECK(hipEventDestroy(begin) == hipSuccess);
            CHECK(hipEventDestroy(end) == hipSuccess);
            const auto got = read(out, std::size_t(rows) * shape.outputs);
            const LinearView host{seed.data(), 1, shape.inputs, row_bytes, shape.format};
            for (unsigned r = 0; r < rows; ++r) {
                const float a = chunk16_dot([&](unsigned i) { return weight_at(host, 0, i); },
                                            x.data() + std::size_t(r) * shape.inputs, shape.inputs);
                const float expected = shape.gate ? gelu_tanh(a) * a : a;
                for (unsigned o : {0U, shape.outputs - 1U})
                    CHECK(bit_cast<unsigned>(got[std::size_t(r) * shape.outputs + o]) == bit_cast<unsigned>(expected));
            }
            std::sort(timings.begin(), timings.end());
            const auto traffic = packed.size() * (shape.gate ? 2ULL : 1ULL);
            const auto* tile = std::getenv("STRIX_G4_OUTPUT_TILE");
            const auto* prefetch = std::getenv("STRIX_G4_CHUNK_PREFETCH");
            const auto* header = std::getenv("STRIX_G4_PACKED_HEADER");
            const auto* streaming = std::getenv("STRIX_G4_STREAMING_WEIGHTS");
            const auto* input_cache = std::getenv("STRIX_G4_INPUT_CACHE");
            std::cout << std::setprecision(8) << "{\"synthetic_linear\":\"" << shape.name << "\",\"rows\":" << rows
                      << ",\"median_ms\":" << timings[2] << ",\"packed_bytes\":" << traffic
                      << ",\"fused_gate_up\":" << (shape.gate && fused ? "true" : "false")
                      << ",\"effective_weight_gb_s\":" << double(traffic) / double(timings[2]) / 1e6
                      << ",\"output_tile_requested\":" << ((tile && std::string_view(tile) == "1") ? "true" : "false")
                      << ",\"chunk_prefetch_requested\":"
                      << ((prefetch && std::string_view(prefetch) == "1") ? "true" : "false")
                      << ",\"packed_header_requested\":" << ((header && std::string_view(header) == "1") ? "true" : "false")
                      << ",\"streaming_weights_requested\":"
                      << ((streaming && std::string_view(streaming) == "1") ? "true" : "false")
                      << ",\"input_cache_requested\":"
                      << ((input_cache && std::string_view(input_cache) == "1") ? "true" : "false")
                      << ",\"sample_bytes\":\"EXACT\"}\n";
        }
    }
}
void norm_rope_tests() {
    {
        const std::vector<float> overflow(32, 1e38F);
        auto dx = device(overflow);
        Buffer out(overflow.size() * sizeof(float)), failure(sizeof(unsigned));
        failure.zero();
        norm(dx.as<float>(), {}, out.as<float>(), 1, 32);
        check_finite(out.as<float>(), overflow.size(), failure.as<unsigned>());
        synchronize();
        unsigned flag = 0;
        failure.download(&flag, sizeof(flag));
        CHECK(flag == 1);
        failure.zero();
        norm(dx.as<float>(), {}, out.as<float>(), 1, 32, nullptr, failure.as<unsigned>());
        synchronize();
        failure.download(&flag, sizeof(flag));
        CHECK(flag == 1);
    }
    for (unsigned width : {1U, 16U, 223U, 224U, 225U, 255U, 256U, 257U, 511U, 512U, 513U,
                           2560U, 4095U, 4096U, 4097U, 10240U, 10752U}) {
        auto x = inputs(8ULL * width);
        x[0] = bit_cast<float>(1U);
        if (x.size() > 32)
            x[32] = 1e15F;
        const auto weights = inputs(width);
        auto dx = device(x), dw = device(weights);
        Buffer out(x.size() * 4), failure(sizeof(unsigned));
        for (bool weighted : {false, true}) {
            std::vector<float> expected(x.size());
            for (unsigned r = 0; r < 8; ++r)
                rms_norm(x.data() + std::size_t(r) * width, weighted ? weights.data() : nullptr,
                         expected.data() + std::size_t(r) * width, width);
            failure.zero();
            norm(dx.as<float>(),
                 weighted ? LinearView{dw.as<std::uint8_t>(), 1, width, std::size_t(width) * 4, WeightFormat::f32}
                          : LinearView{},
                 out.as<float>(), 8, width, nullptr, failure.as<unsigned>());
            synchronize();
            bytes(read(out, x.size()), expected);
            unsigned flag = 1;
            failure.download(&flag, sizeof(flag));
            CHECK(flag == 0);
            // In-place use must finish every source read before publishing output.
            dx.upload(x.data(), dx.size());
            norm(dx.as<float>(),
                 weighted ? LinearView{dw.as<std::uint8_t>(), 1, width, std::size_t(width) * 4, WeightFormat::f32}
                          : LinearView{},
                 dx.as<float>(), 8, width);
            synchronize();
            bytes(read(dx, x.size()), expected);
            dx.upload(x.data(), dx.size());
        }
    }
    fixture::Tables tables(65536);
    auto sliding = device(tables.sliding), global = device(tables.global);
    const std::vector<std::uint64_t> positions = {0, 511, 512, 65535, 1, 2, 3, 513};
    auto dp = device(positions);
    for (bool g : {false, true}) {
        const unsigned dim = g ? 512U : 256U;
        auto x = inputs(8ULL * 8 * dim);
        x[0] = bit_cast<float>(1U);
        x[32] = 1e15F;
        auto dx = device(x);
        const auto& table = g ? tables.global : tables.sliding;
        for (const auto format : {WeightFormat::bf16, WeightFormat::f32, WeightFormat::q8_0,
                                  WeightFormat::q4_k, WeightFormat::q6_k}) {
            const auto packed = fixture::packed(format, 1, dim);
            auto dw = device(packed);
            const LinearView host{packed.data(), 1, dim, packed_row_bytes(format, dim), format};
            auto weight = host;
            weight.data = dw.as<std::uint8_t>();
            std::vector<float> w(dim);
            for (unsigned i = 0; i < dim; ++i)
                w[i] = weight_at(host, 0, i);
            for (bool weighted : {false, true}) {
                if (!weighted && format != WeightFormat::f32)
                    continue;
                std::vector<float> expected = x;
                for (unsigned r = 0; r < 8; ++r)
                    for (unsigned h = 0; h < 8; ++h) {
                        auto* a = expected.data() + (std::size_t(r) * 8 + h) * dim;
                        rms_norm(a, weighted ? w.data() : nullptr, a, dim);
                        const auto base = std::size_t(positions[r]) * (g ? 512U : 256U);
                        for (unsigned i = 0; i < (g ? 64U : 128U); ++i) {
                            const float u = a[i], v = a[i + dim / 2], cs = table[base + 2 * i], sn = table[base + 2 * i + 1];
                            a[i] = u * cs - v * sn;
                            a[i + dim / 2] = v * cs + u * sn;
                        }
                    }
                Buffer failure(sizeof(unsigned));
                failure.zero();
                dx.upload(x.data(), dx.size());
                head_norm_rope(dx.as<float>(), weighted ? weight : LinearView{}, 8, 8, dim, dp.as<std::uint64_t>(),
                               g ? global.as<float>() : sliding.as<float>(), g, nullptr, failure.as<unsigned>());
                synchronize();
                bytes(read(dx, x.size()), expected);
                unsigned flag = 1;
                failure.download(&flag, sizeof(flag));
                CHECK(flag == 0);
                auto poisoned = x;
                poisoned.back() = bit_cast<float>(0x7fc01234U);
                dx.upload(poisoned.data(), dx.size());
                head_norm_rope(dx.as<float>(), weighted ? weight : LinearView{}, 8, 8, dim, dp.as<std::uint64_t>(),
                               g ? global.as<float>() : sliding.as<float>(), g, nullptr, failure.as<unsigned>());
                synchronize();
                failure.download(&flag, sizeof(flag));
                CHECK(flag == 1);
                const auto got = read(dx, x.size());
                CHECK(std::memcmp(got.data(), expected.data(), (x.size() - dim) * sizeof(float)) == 0);
            }
        }
    }
}
void epilogue_tests() {
    for (unsigned width : {1U, 17U, 225U, 257U, 511U, 2560U, 4095U, 4096U, 4097U}) {
        const auto branch = inputs(8ULL * width), weight = inputs(width);
        auto residual_host = inputs(branch.size());
        for (auto& a : residual_host)
            a = a + 0.125F;
        const float scale = 0.875F;
        auto db = device(branch), dw = device(weight), ds = device(std::vector<float>{scale});
        Buffer failure(4), next(branch.size() * 4);
        for (bool following : {false, true}) {
            auto expected = residual_host;
            std::vector<float> normalized(branch.size()), expected_next(branch.size());
            for (unsigned r = 0; r < 8; ++r) {
                const auto base = std::size_t(r) * width;
                rms_norm(branch.data() + base, weight.data(), normalized.data() + base, width);
                for (unsigned i = 0; i < width; ++i) {
                    const float a = expected[base + i] + normalized[base + i];
                    expected[base + i] = following ? a : a * scale;
                }
                if (following)
                    rms_norm(expected.data() + base, weight.data(), expected_next.data() + base, width);
            }
            auto dr = device(residual_host);
            const LinearView w{dw.as<std::uint8_t>(), 1, width, std::size_t(width) * 4, WeightFormat::f32};
            const LinearView scalar{ds.as<std::uint8_t>(), 1, 1, 4, WeightFormat::f32};
            failure.zero();
            norm_residual(db.as<float>(), w, dr.as<float>(), 8, width, following ? LinearView{} : scalar,
                          following ? next.as<float>() : nullptr, following ? w : LinearView{}, failure.as<unsigned>());
            synchronize();
            unsigned flag = 0;
            failure.download(&flag, 4);
            CHECK(flag == 0);
            bytes(read(dr, expected.size()), expected);
            if (following)
                bytes(read(next, expected_next.size()), expected_next);
        }
    }
}
void ple_head_tests() {
    const std::vector<unsigned> token = {4, 3, 2, 1, 0, 6, 5, 7};
    auto dt = device(token);
    for (auto f : {WeightFormat::bf16, WeightFormat::q8_0}) {
        const unsigned width = 256, layer_count = 42, total = width * layer_count;
        const auto b = fixture::packed(f, 8, total);
        auto db = device(b);
        LinearView ht{b.data(), 8, total, packed_row_bytes(f, total), f}, t = ht;
        t.data = db.as<std::uint8_t>();
        Buffer out(8ULL * total * 4);
        std::vector<float> expected(8ULL * total);
        for (unsigned r = 0; r < 8; ++r)
            for (unsigned i = 0; i < total; ++i)
                expected[std::size_t(r) * total + i] = embedding_value(weight_at(ht, token[r], i), ple_embedding_scale);
        embedding(t, dt.as<unsigned>(), 8, out.as<float>(), ple_embedding_scale);
        synchronize();
        bytes(read(out, expected.size()), expected);
        auto projection = inputs(expected.size()), weights = inputs(width);
        auto dp = device(projection), dw = device(weights);
        for (auto& a : projection)
            a = a * model_ple_scale;
        for (unsigned r = 0; r < 8; ++r)
            for (unsigned l = 0; l < layer_count; ++l) {
                auto* p = projection.data() + (std::size_t(r) * layer_count + l) * width;
                rms_norm(p, weights.data(), p, width);
                for (unsigned i = 0; i < width; ++i)
                    p[i] = (p[i] + expected[(std::size_t(r) * layer_count + l) * width + i]) * ple_combine_scale;
            }
        Buffer failure(sizeof(unsigned));
        failure.zero();
        ple_combine(dp.as<float>(), t, {dw.as<std::uint8_t>(), 1, width, width * 4, WeightFormat::f32},
                    dt.as<unsigned>(), 8, layer_count, width, nullptr, failure.as<unsigned>());
        synchronize();
        bytes(read(dp, projection.size()), projection);
        unsigned flag = 1;
        failure.download(&flag, sizeof(flag));
        CHECK(flag == 0);
        auto gate = inputs(8ULL * width), eg = gate;
        auto dg = device(gate);
        for (unsigned r = 0; r < 8; ++r)
            for (unsigned i = 0; i < width; ++i)
                eg[std::size_t(r) * width + i] = gelu_tanh(eg[std::size_t(r) * width + i]) *
                                                 projection[(std::size_t(r) * layer_count + 23) * width + i];
        ple_gate(dg.as<float>(), dp.as<float>(), 8, layer_count, 23, width, nullptr, failure.as<unsigned>());
        synchronize();
        bytes(read(dg, eg.size()), eg);
        failure.download(&flag, sizeof(flag));
        CHECK(flag == 0);
    }
    std::vector<float> x(8 * 1025, -1000.0F);
    for (unsigned r = 0; r < 8; ++r) {
        x[r * 1025 + 257] = 1000;
        x[r * 1025 + 513] = 2000;
    }
    auto dx = device(x);
    Buffer scratch(8 * 5 * sizeof(Candidate)), out(8 * sizeof(Candidate));
    cap_logits(dx.as<float>(), unsigned(x.size()));
    argmax(dx.as<float>(), 8, 1025, scratch.as<Candidate>(), out.as<Candidate>());
    synchronize();
    std::vector<Candidate> got(8);
    out.download(got.data(), out.size());
    for (auto a : got) {
        CHECK(a.id == 257);
        CHECK(a.value == 30.0F);
    }
    auto a = inputs(8 * 257), b = inputs(8 * 257);
    std::vector<float> expected(a.size());
    const float scalar = 0.875F;
    auto da = device(a), db = device(b);
    Buffer ds(4);
    ds.upload(&scalar, 4);
    for (std::size_t i = 0; i < a.size(); ++i)
        expected[i] = (a[i] + b[i]) * scalar;
    residual(da.as<float>(), db.as<float>(), unsigned(a.size()), {ds.as<std::uint8_t>(), 1, 1, 4, WeightFormat::f32});
    synchronize();
    bytes(read(da, a.size()), expected);
}
void attention_tests() {
    const unsigned rows = 8, hcount = 8;
    struct Scenario {
        std::uint64_t frontier;
        bool global;
    };
    for (const auto scenario : {Scenario{1, false}, Scenario{1, true}, Scenario{56, false}, Scenario{56, true},
                                Scenario{515, false}, Scenario{515, true}}) {
        const auto frontier = scenario.frontier;
        const bool global = scenario.global;
        const unsigned stride = frontier < 64U ? 64U : unsigned(frontier) + rows;
        const unsigned dim = global ? 512U : 256U, width = 2 * dim, slots = global ? stride : 512U;
        const auto keys = inputs(std::size_t(stride) * width), values = inputs(std::size_t(stride) * width),
                   q = inputs(std::size_t(rows) * hcount * dim);
        std::vector<float> sk(std::size_t(slots) * width), sv(sk.size());
        std::vector<std::uint64_t> tags(slots, UINT64_MAX);
        const unsigned first = !global && frontier > 512U ? unsigned(frontier - 512U) : 0U;
        for (unsigned p = first; p < frontier; ++p) {
            const auto slot = global ? p : p % slots;
            tags[slot] = p;
            std::copy_n(keys.data() + std::size_t(p) * width, width, sk.data() + std::size_t(slot) * width);
            std::copy_n(values.data() + std::size_t(p) * width, width, sv.data() + std::size_t(slot) * width);
        }
        auto dk = device(sk), dv = device(sv), tags_d = device(tags), dq = device(q);
        std::vector<float> ok(keys.begin() + std::ptrdiff_t(frontier * width), keys.end()),
            ov(values.begin() + std::ptrdiff_t(frontier * width), values.end());
        auto dok = device(ok), dov = device(ov);
        std::vector<KvRead> reads;
        for (unsigned r = 0; r < rows; ++r)
            reads.push_back({dk.as<float>(), dv.as<float>(), tags_d.as<std::uint64_t>(), dok.as<float>(),
                             dov.as<float>(), frontier, frontier + r, width, slots, 8, 512, global});
        auto dr = device(reads);
        const unsigned chunks = (stride + 63) / 64;
        Buffer scores(std::size_t(rows) * hcount * stride * 4), maxima(std::size_t(rows) * hcount * (chunks + 1) * 4),
            parts(std::size_t(rows) * hcount * chunks * (dim + 1) * 4), out(std::size_t(rows) * hcount * dim * 4),
            failure(4);
        failure.zero();
        std::vector<float> expected(std::size_t(rows) * hcount * dim);
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned h = 0; h < hcount; ++h) {
                const auto p = frontier + r;
                const auto range = visible(p, global, 512);
                std::vector<float> scores_h(stride);
                float maximum = -std::numeric_limits<float>::infinity();
                for (auto j = range.first; j <= p; ++j) {
                    auto s = lane_dot(q.data() + (std::size_t(r) * hcount + h) * dim,
                                      keys.data() + std::size_t(j) * width + (h / 4) * dim, dim);
                    scores_h[j] = s;
                    maximum = std::max(maximum, s);
                }
                for (auto j = range.first; j <= p; ++j)
                    scores_h[j] = declared_exp(scores_h[j] - maximum);
                float den = 0;
                std::vector<float> num(dim);
                for (auto c = range.first / 64; c <= p / 64; ++c) {
                    const auto begin = std::max(range.first, c * 64), end = std::min(p + 1, (c + 1) * 64);
                    float part = 0;
                    for (auto j = begin; j < end; ++j)
                        part = part + scores_h[j];
                    den = den + part;
                    for (unsigned i = 0; i < dim; ++i) {
                        part = 0;
                        for (auto j = begin; j < end; ++j)
                            part = part + scores_h[j] * values[std::size_t(j) * width + (h / 4) * dim + i];
                        num[i] = num[i] + part;
                    }
                }
                for (unsigned i = 0; i < dim; ++i)
                    expected[(std::size_t(r) * hcount + h) * dim + i] = num[i] / den;
            }
        attention(dq.as<float>(), dr.as<KvRead>(), rows, hcount, dim, stride, scores.as<float>(), maxima.as<float>(),
                  parts.as<float>(), out.as<float>(), failure.as<unsigned>());
        synchronize();
        unsigned flag = 1;
        failure.download(&flag, 4);
        CHECK(flag == 0);
        bytes(read(out, expected.size()), expected);
        tags.assign(slots, UINT64_MAX);
        tags_d.upload(tags.data(), tags_d.size());
        failure.zero();
        attention(dq.as<float>(), dr.as<KvRead>(), rows, hcount, dim, stride, scores.as<float>(), maxima.as<float>(),
                  parts.as<float>(), out.as<float>(), failure.as<unsigned>());
        synchronize();
        failure.download(&flag, 4);
        CHECK(flag & 2U);
    }
}
} // namespace
int main(int argc, char** argv) {
    try {
        require_device();
        if (argc == 2 && std::string_view(argv[1]) == "--bench-linear") {
            linear_bench();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--bench-linear-unfused") {
            linear_bench(false);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--bench-linear-small") {
            linear_bench(true, true);
            return 0;
        }
        if (argc != 1)
            throw std::runtime_error("usage: [--bench-linear | --bench-linear-unfused | --bench-linear-small]");
        scalar_tests();
        linear_tests();
        linear_failure_tests();
        norm_rope_tests();
        epilogue_tests();
        ple_head_tests();
        attention_tests();
        std::cout
            << "g4-v1 operators: byte equality, packed formats, R=1..8, poison, norms/rope/PLE/head/attention passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
