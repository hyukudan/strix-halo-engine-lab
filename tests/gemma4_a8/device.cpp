// g4-a8-v1 device bytes: quantization and every quantized format/epilogue against the host oracle for
// R=1..8, then the complete synthetic graph against the host graph, R rows against serial passes.
#include "gemma4/kernels/a8_host.hpp"
#include "gemma4/kernels/a8_ops.hpp"
#include "gemma4/kernels/device.hpp"
#include "../gemma4_gpu/fixture.hpp"

using namespace strix::gemma4;
using namespace strix::gemma4::gpu;
namespace {
std::vector<float> download(const Buffer& b, std::size_t count) {
    std::vector<float> v(count);
    b.download(v.data(), count * sizeof(float));
    return v;
}
LinearView device_view(Buffer& storage, const std::vector<std::uint8_t>& bytes, WeightFormat f, unsigned outputs,
                       unsigned width) {
    storage.resize(bytes.size());
    storage.upload(bytes.data(), bytes.size());
    return {storage.as<std::uint8_t>(), outputs, width, packed_row_bytes(f, width), f};
}
void kernel_tests() {
    std::mt19937 rng(91);
    for (const auto f : {WeightFormat::q8_0, WeightFormat::q4_k, WeightFormat::q6_k})
        for (const unsigned width : {256U, 768U, 2560U}) {
            const unsigned outputs = 37;
            const auto wb = fixture::packed(f, outputs, width, 7U * width + unsigned(f)),
                       ub = fixture::packed(f, outputs, width, 9U * width + unsigned(f)),
                       eb = fixture::packed(f, 21, width, 11U * width + unsigned(f));
            const LinearView hw{wb.data(), outputs, width, packed_row_bytes(f, width), f},
                hu{ub.data(), outputs, width, packed_row_bytes(f, width), f},
                he{eb.data(), 21, width, packed_row_bytes(f, width), f};
            Buffer ws, us, es;
            const auto dw = device_view(ws, wb, f, outputs, width), du = device_view(us, ub, f, outputs, width),
                       de = device_view(es, eb, f, 21, width);
            std::vector<float> x(8ULL * width), product(8ULL * outputs);
            for (auto& v : x)
                v = std::ldexp(float(int(rng() % 4001) - 2000), int(rng() % 9) - 12);
            for (unsigned i = 0; i < 64; ++i)
                x[rng() % x.size()] = 0.0F; // zero groups and signed zeros
            for (unsigned i = 0; i < 32; ++i)
                x[width + i] = i % 2 ? -0.0F : 0.0F;
            for (auto& v : product)
                v = float(int(rng() % 201) - 100) * 0.01F;
            Buffer dx(x.size() * 4), dp(product.size() * 4), codes(8ULL * width), meta(8ULL * width / 32U * 8U),
                out(8ULL * outputs * 4), extra(8ULL * 21 * 4), failure(4);
            dx.upload(x.data(), x.size() * 4);
            dp.upload(product.data(), product.size() * 4);
            failure.zero();
            const A8Rows rows_view{codes.as<std::int8_t>(), meta.as<a8::Meta>(), width};
            for (unsigned rows = 1; rows <= 8; ++rows) {
                std::vector<std::int8_t> hc;
                std::vector<a8::Meta> hm;
                a8::quantize_rows(x.data(), rows, width, hc, hm);
                a8_quantize(dx.as<float>(), rows, width, rows_view);
                synchronize();
                std::vector<std::int8_t> dc(hc.size());
                std::vector<a8::Meta> dm(hm.size());
                codes.download(dc.data(), dc.size());
                meta.download(dm.data(), dm.size() * sizeof(a8::Meta));
                CHECK(std::memcmp(dc.data(), hc.data(), hc.size()) == 0);
                CHECK(std::memcmp(dm.data(), hm.data(), hm.size() * sizeof(a8::Meta)) == 0);
                const auto groups = width / 32U;
                auto host_dot = [&](LinearView v, unsigned o, unsigned r) {
                    return a8::dot(v, o, hc.data() + std::size_t(r) * width, hm.data() + std::size_t(r) * groups);
                };
                // Plain, PLE-product, softcap, gate/up and a three-matrix shared grid.
                for (unsigned mode = 0; mode < 5; ++mode) {
                    LinearEpilogue ep;
                    if (mode == 1) {
                        ep.product = dp.as<float>();
                        ep.product_stride = outputs;
                    }
                    ep.cap = mode == 2;
                    Buffer second(8ULL * outputs * 4);
                    if (mode == 4) {
                        ep.extra[0] = de;
                        ep.extra_out[0] = extra.as<float>();
                        ep.extra[1] = dw;
                        ep.extra_out[1] = second.as<float>();
                    }
                    if (mode == 3)
                        CHECK(a8_gate_up(dw, du, rows_view, rows, out.as<float>(), nullptr, failure.as<unsigned>()));
                    else
                        CHECK(a8_linear(dw, rows_view, rows, out.as<float>(), ep, nullptr, failure.as<unsigned>()));
                    synchronize();
                    const auto got = download(out, std::size_t(rows) * outputs);
                    std::vector<float> want(got.size());
                    for (unsigned r = 0; r < rows; ++r)
                        for (unsigned o = 0; o < outputs; ++o) {
                            float v = host_dot(hw, o, r);
                            if (mode == 1)
                                v = gelu_tanh(v) * product[std::size_t(r) * outputs + o];
                            if (mode == 2)
                                v = softcap(v);
                            if (mode == 3)
                                v = gelu_tanh(v) * host_dot(hu, o, r);
                            want[std::size_t(r) * outputs + o] = v;
                        }
                    bytes(got, want);
                    if (mode == 4) {
                        const auto e = download(extra, std::size_t(rows) * 21);
                        std::vector<float> ew(e.size());
                        for (unsigned r = 0; r < rows; ++r)
                            for (unsigned o = 0; o < 21; ++o)
                                ew[std::size_t(r) * 21 + o] = host_dot(he, o, r);
                        bytes(e, ew);
                        const auto s = download(second, std::size_t(rows) * outputs);
                        std::vector<float> sw(s.size());
                        for (unsigned r = 0; r < rows; ++r)
                            for (unsigned o = 0; o < outputs; ++o)
                                sw[std::size_t(r) * outputs + o] = host_dot(hw, o, r);
                        bytes(s, sw);
                    }
                }
            }
            unsigned flag = 1;
            failure.download(&flag, 4);
            CHECK(flag == 0);
        }
    // Quantized outputs (blocks of 32 outputs) equal the separate quantization of the stored values.
    for (const auto f : {WeightFormat::q4_k, WeightFormat::q6_k}) {
        const unsigned width = 512, outputs = 256;
        const auto wb = fixture::packed(f, outputs, width, 41U + unsigned(f)),
                   ub = fixture::packed(f, outputs, width, 43U + unsigned(f));
        Buffer ws, us;
        const auto dw = device_view(ws, wb, f, outputs, width), du = device_view(us, ub, f, outputs, width);
        std::vector<float> x(8ULL * width);
        for (auto& v : x)
            v = float(int(rng() % 2001) - 1000) * 0.001F;
        Buffer dx(x.size() * 4), codes(8ULL * width), meta(8ULL * width / 4U), out(8ULL * outputs * 4),
            qcodes(8ULL * outputs), qmeta(8ULL * outputs / 4U), failure(4);
        dx.upload(x.data(), x.size() * 4);
        failure.zero();
        const A8Rows in{codes.as<std::int8_t>(), meta.as<a8::Meta>(), width},
            quant{qcodes.as<std::int8_t>(), qmeta.as<a8::Meta>(), outputs};
        for (unsigned rows = 1; rows <= 8; ++rows)
            for (const bool gate : {false, true}) {
                a8_quantize(dx.as<float>(), rows, width, in);
                if (gate)
                    CHECK(a8_gate_up(dw, du, in, rows, out.as<float>(), nullptr, failure.as<unsigned>(), quant));
                else
                    CHECK(a8_linear(dw, in, rows, out.as<float>(), {}, nullptr, failure.as<unsigned>(), quant));
                synchronize();
                const auto values = download(out, std::size_t(rows) * outputs);
                std::vector<std::int8_t> hc, dc(std::size_t(rows) * outputs);
                std::vector<a8::Meta> hm, dm(std::size_t(rows) * outputs / 32U);
                a8::quantize_rows(values.data(), rows, outputs, hc, hm);
                qcodes.download(dc.data(), dc.size());
                qmeta.download(dm.data(), dm.size() * sizeof(a8::Meta));
                CHECK(std::memcmp(dc.data(), hc.data(), hc.size()) == 0);
                CHECK(std::memcmp(dm.data(), hm.data(), hm.size() * sizeof(a8::Meta)) == 0);
            }
    }
    // BF16 views are not quantized linears.
    const auto bb = fixture::packed(WeightFormat::bf16, 4, 256);
    Buffer bs, codes(256), meta(64), out(16), failure(4);
    const auto bv = device_view(bs, bb, WeightFormat::bf16, 4, 256);
    CHECK(!a8_linear(bv, {codes.as<std::int8_t>(), meta.as<a8::Meta>(), 256}, 1, out.as<float>(), {}, nullptr,
                     failure.as<unsigned>()));
}
template <class Stream>
ForwardResult ingest(Target& target, Stream& stream, unsigned n, unsigned token, bool permute = false) {
    auto tx = stream.begin();
    std::vector<RowPlan> plans;
    for (unsigned r = 0; r < n; ++r)
        plans.push_back({13, 7, stream.frontier() + r, (token + r) % 37, tx.get(), r, true});
    if (permute)
        std::reverse(plans.begin(), plans.end());
    const auto out = target.forward_rows(plans);
    tx->commit(n);
    return out;
}
void compare(const ForwardResult& a, const ForwardResult& b) {
    CHECK(a.rows == b.rows);
    bytes(a.logits, b.logits);
    bytes(a.taps, b.taps);
}
void graph_tests(unsigned count) {
    const auto model = fixture::tiny(count, true);
    fixture::Tables tables;
    const auto weights = bind_host_weights(model);
    HostTarget host(weights, tables.view(), 2, a8::host_recipe());
    DeviceOptions options;
    options.identity = a8::arithmetic_id;
    DeviceTarget device(weights, tables.view(), options);
    CHECK(device.stats().identity == a8::arithmetic_id);
    DeviceStream batch(model.config, 1024), serial(model.config, 1024);
    HostStream reference(model.config);
    for (unsigned n = 1; n <= 8; ++n) {
        const auto out = ingest(device, batch, n, 2 + n, true), expected = ingest(host, reference, n, 2 + n, true);
        compare(out, expected);
        const auto selected = device.selected();
        for (unsigned r = 0; r < n; ++r) {
            const auto* logits = out.logits.data() + std::size_t(r) * model.config.vocab;
            unsigned best = 0;
            for (unsigned i = 1; i < model.config.vocab; ++i)
                if (better(logits[i], i, logits[best], best))
                    best = i;
            CHECK(selected[r].id == best);
            const auto single = ingest(device, serial, 1, 2 + n + r);
            CHECK(std::memcmp(single.logits.data(), out.logits.data() + std::size_t(r) * model.config.vocab,
                              single.logits.size() * 4) == 0);
            CHECK(std::memcmp(single.taps.data(), out.taps.data() + std::size_t(r) * 5 * model.config.hidden,
                              single.taps.size() * 4) == 0);
        }
        CHECK(batch.digest() == serial.digest());
        CHECK(batch.digest() == reference.digest());
    }
    DeviceOptions wrong;
    wrong.identity = "g4-a8-v0";
    refuses([&] { DeviceTarget invalid(weights, tables.view(), wrong); });
}
} // namespace
int main() {
    try {
        require_device();
        kernel_tests();
        graph_tests(1);
        graph_tests(6);
        std::cout << "g4-a8-v1 device: quantization, Q8_0/Q4_K/Q6_K epilogues, graph R=1..8 host/serial bytes passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
