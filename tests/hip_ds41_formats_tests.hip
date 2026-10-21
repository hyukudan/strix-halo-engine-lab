// deepseek41 format fixtures: device decoders, A8 quantizer and decode products against the host
// reference, bitwise. Synthetic blocks always; real blocks when STRIX_DS41_Q2_GGUF and/or
// STRIX_DS41_DSPARK_GGUF name the model files. STRIX_DS41_BENCH=1 adds product timings on real
// layers (achieved weight bytes per second, cold caches).

#include "strix/gguf_reader.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/dist/communicator.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <exception>
#include <memory>
#include <thread>
#include <tuple>
#include <vector>

namespace {

using strix::ds41::WeightFormat;
int failures = 0;
void expect(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}
void check(hipError_t status, const char* what) {
    if (status != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
}

template <class T>
struct Device {
    T* data = nullptr;
    std::size_t count = 0;
    explicit Device(std::size_t n) : count(n) { check(hipMalloc(reinterpret_cast<void**>(&data), std::max<std::size_t>(n, 1U) * sizeof(T)), "hipMalloc"); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() { (void)hipFree(data); }
    void upload(const T* host, std::size_t n) { check(hipMemcpy(data, host, n * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    std::vector<T> download() const {
        std::vector<T> out(count);
        check(hipMemcpy(out.data(), data, count * sizeof(T), hipMemcpyDeviceToHost), "download");
        return out;
    }
};

std::size_t mismatches(const std::vector<float>& a, const std::vector<float>& b) {
    std::size_t bad = 0;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i)
        bad += std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]);
    return bad + (a.size() != b.size());
}

std::vector<float> host_decode(WeightFormat format, const std::vector<std::uint8_t>& bytes, std::size_t blocks) {
    using namespace strix::ds41;
    std::vector<float> out(blocks * block_elements(format));
    for (std::size_t b = 0; b < blocks; ++b) {
        const auto* p = bytes.data() + b * block_bytes(format);
        if (format == WeightFormat::iq2_xxs)
            decode_iq2_xxs(std::span<const std::uint8_t, iq2_xxs_block_bytes>(p, iq2_xxs_block_bytes), std::span<float, 256>(out.data() + 256U * b, 256U));
        else if (format == WeightFormat::q2_k)
            decode_q2_k(std::span<const std::uint8_t, q2_k_block_bytes>(p, q2_k_block_bytes), std::span<float, 256>(out.data() + 256U * b, 256U));
        else if (format == WeightFormat::q8_0)
            decode_q8_0(std::span<const std::uint8_t, q8_0_block_bytes>(p, q8_0_block_bytes), std::span<float, 32>(out.data() + 32U * b, 32U));
        else
            decode_mxfp4(std::span<const std::uint8_t, mxfp4_block_bytes>(p, mxfp4_block_bytes), std::span<float, 32>(out.data() + 32U * b, 32U));
    }
    return out;
}

bool check_decode(WeightFormat format, const std::vector<std::uint8_t>& bytes, const std::string& label) {
    const auto blocks = bytes.size() / strix::ds41::block_bytes(format);
    Device<std::uint8_t> device(bytes.size());
    device.upload(bytes.data(), bytes.size());
    Device<float> out(blocks * strix::ds41::block_elements(format));
    strix::ds41::enqueue_decode_blocks(format, device.data, blocks, out.data, nullptr);
    check(hipDeviceSynchronize(), "decode");
    const auto bad = mismatches(out.download(), host_decode(format, bytes, blocks));
    expect(bad == 0U, label + ": device decode equals host decode bitwise (" + std::to_string(bad) + " differ)");
    return bad == 0U;
}

std::vector<float> activations(std::uint32_t rows, std::uint32_t width, unsigned seed) {
    std::mt19937 random(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> values(static_cast<std::size_t>(rows) * width);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const float magnitude = std::ldexp(1.0F, static_cast<int>(i / 32U % 13U) - 6);
        values[i] = normal(random) * magnitude;
    }
    // A zero group, a tiny group and a group with an exact tie in every row.
    for (std::uint32_t r = 0; r < rows; ++r) {
        auto* row = values.data() + static_cast<std::size_t>(r) * width;
        std::fill(row, row + 32, 0.0F);
        for (unsigned j = 32; j < 64; ++j) row[j] = 1e-37F * static_cast<float>(j);
        for (unsigned j = 64; j < 96; ++j) row[j] = j == 64 ? 127.0F : 0.5F + static_cast<float>(j % 4);
    }
    return values;
}

// Products of experts (laid out consecutively in `weights`) for rows 1, 3 and 8.
bool check_products(WeightFormat format, const std::vector<std::uint8_t>& weights, std::uint32_t experts, std::uint32_t width,
                    std::uint32_t outputs, const std::string& label, unsigned seed) {
    using namespace strix::ds41;
    const std::uint64_t row_bytes = width / block_elements(format) * block_bytes(format);
    const std::uint64_t stride = row_bytes * outputs;
    if (weights.size() < stride * experts) throw std::logic_error("product fixture weights too short");
    constexpr std::uint32_t max_rows = product_max_rows;
    const auto values = activations(max_rows, width, seed);
    Device<float> x(values.size());
    x.upload(values.data(), values.size());
    Device<std::int8_t> codes(values.size());
    Device<float> deltas(values.size() / 32U);
    Device<std::int32_t> sums(values.size() / 32U);
    enqueue_quantize_a8(x.data, width, max_rows, codes.data, deltas.data, sums.data, nullptr);
    check(hipDeviceSynchronize(), "quantize");
    const auto device_codes = codes.download();
    const auto device_deltas = deltas.download();
    const auto device_sums = sums.download();
    std::vector<A8Activation> host(max_rows);
    bool quantized = true;
    for (std::uint32_t r = 0; r < max_rows; ++r) {
        host[r] = quantize_a8(std::span<const float>(values.data() + static_cast<std::size_t>(r) * width, width));
        quantized = quantized && std::equal(host[r].codes.begin(), host[r].codes.end(), device_codes.begin() + static_cast<std::ptrdiff_t>(r) * width) &&
            std::equal(host[r].sums.begin(), host[r].sums.end(), device_sums.begin() + static_cast<std::ptrdiff_t>(r) * (width / 32U));
        for (std::size_t g = 0; g < width / 32U; ++g)
            quantized = quantized && std::bit_cast<std::uint32_t>(host[r].deltas[g]) ==
                std::bit_cast<std::uint32_t>(device_deltas[static_cast<std::size_t>(r) * (width / 32U) + g]);
    }
    expect(quantized, label + ": device A8 quantization equals host codes, deltas and sums");

    Device<std::uint8_t> device_weights(weights.size());
    device_weights.upload(weights.data(), weights.size());
    std::vector<std::uint32_t> order(experts);
    std::iota(order.begin(), order.end(), 0U);
    std::reverse(order.begin(), order.end());
    Device<std::uint32_t> device_order(order.size());
    device_order.upload(order.data(), order.size());
    std::size_t bad = 0, total = 0;
    for (const std::uint32_t rows : {1U, 3U, 8U}) {
        Device<float> out(static_cast<std::size_t>(rows) * experts * outputs);
        ProductView view;
        view.weights = device_weights.data; view.expert_stride_bytes = stride; view.row_bytes = row_bytes;
        view.width = width; view.outputs = outputs; view.experts = device_order.data; view.expert_count = experts;
        view.codes = codes.data; view.deltas = deltas.data; view.rows = rows; view.out = out.data;
        enqueue_product(format, view, nullptr);
        check(hipDeviceSynchronize(), "product");
        const auto result = out.download();
        for (std::uint32_t r = 0; r < rows; ++r)
            for (std::uint32_t e = 0; e < experts; ++e)
                for (std::uint32_t o = 0; o < outputs; ++o) {
                    const auto row = std::span<const std::uint8_t>(weights.data() + order[e] * stride + o * row_bytes, row_bytes);
                    const float expected = product(format, row, host[r], width);
                    const float got = result[(static_cast<std::size_t>(r) * experts + e) * outputs + o];
                    bad += std::bit_cast<std::uint32_t>(expected) != std::bit_cast<std::uint32_t>(got);
                    ++total;
                }
    }
    expect(bad == 0U, label + ": device products equal host products bitwise (" + std::to_string(bad) + " of " +
        std::to_string(total) + " differ)");
    std::cout << label << ": products rows 1/3/8 " << total - bad << "/" << total << " bit-exact\n";
    return bad == 0U;
}

// CB4 (requantization candidates): products of 4-bit rows over a shuffled int8 table equal the host products of the
// equivalent Q8_0 rows bitwise (rows 1/3/8, three experts in reverse order).
void check_cb4_products() {
    using namespace strix::ds41;
    constexpr std::uint32_t experts = 3, width = 5120, outputs = 40;
    std::mt19937 random(21);
    std::vector<std::int8_t> table{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    std::shuffle(table.begin(), table.end(), random);
    const std::size_t blocks = std::size_t{experts} * outputs * width / 32U;
    std::vector<std::uint8_t> emu(blocks * q8_0_block_bytes), packed(blocks * cb4_block_bytes);
    std::uniform_int_distribution<int> index(0, 15), scale(0, 0x3ff);
    for (std::size_t b = 0; b < blocks; ++b) {
        const auto d = static_cast<std::uint16_t>(0x1400U | static_cast<unsigned>(scale(random)));  // finite scales near 2^-10
        auto* e = emu.data() + b * q8_0_block_bytes;
        auto* p = packed.data() + b * cb4_block_bytes;
        e[0] = p[0] = static_cast<std::uint8_t>(d & 0xffU);
        e[1] = p[1] = static_cast<std::uint8_t>(d >> 8U);
        int k[32];
        for (int& v : k) v = index(random);
        for (int j = 0; j < 32; ++j) e[2 + j] = static_cast<std::uint8_t>(table[static_cast<std::size_t>(k[j])]);
        for (int j = 0; j < 16; ++j) p[2 + j] = static_cast<std::uint8_t>(k[j] | (k[j + 16] << 4));
    }
    const auto values = activations(product_max_rows, width, 31);
    std::vector<A8Activation> host(product_max_rows);
    for (std::uint32_t r = 0; r < product_max_rows; ++r)
        host[r] = quantize_a8(std::span<const float>(values.data() + static_cast<std::size_t>(r) * width, width));
    Device<float> x(values.size());
    x.upload(values.data(), values.size());
    Device<std::int8_t> codes(values.size());
    Device<float> deltas(values.size() / 32U);
    Device<std::int32_t> sums(values.size() / 32U);
    enqueue_quantize_a8(x.data, width, product_max_rows, codes.data, deltas.data, sums.data, nullptr);
    Device<std::uint8_t> device_weights(packed.size());
    device_weights.upload(packed.data(), packed.size());
    std::vector<std::uint32_t> order{2, 1, 0};
    Device<std::uint32_t> device_order(order.size());
    device_order.upload(order.data(), order.size());
    const std::uint64_t row_q8 = width / 32U * q8_0_block_bytes, row_cb4 = width / 32U * cb4_block_bytes;
    std::size_t bad = 0, total = 0;
    for (const std::uint32_t rows : {1U, 3U, 8U}) {
        Device<float> out(static_cast<std::size_t>(rows) * experts * outputs);
        ProductView view;
        view.weights = device_weights.data; view.expert_stride_bytes = row_cb4 * outputs; view.row_bytes = row_cb4;
        view.width = width; view.outputs = outputs; view.experts = device_order.data; view.expert_count = experts;
        view.codes = codes.data; view.deltas = deltas.data; view.rows = rows; view.out = out.data;
        std::memcpy(view.table, table.data(), 16);
        enqueue_product(WeightFormat::cb4, view, nullptr);
        check(hipDeviceSynchronize(), "CB4 product");
        const auto result = out.download();
        for (std::uint32_t r = 0; r < rows; ++r)
            for (std::uint32_t e = 0; e < experts; ++e)
                for (std::uint32_t o = 0; o < outputs; ++o) {
                    const auto row = std::span<const std::uint8_t>(emu.data() + (order[e] * outputs + o) * row_q8, row_q8);
                    const float expected = product(WeightFormat::q8_0, row, host[r], width);
                    const float got = result[(static_cast<std::size_t>(r) * experts + e) * outputs + o];
                    bad += std::bit_cast<std::uint32_t>(expected) != std::bit_cast<std::uint32_t>(got);
                    ++total;
                }
    }
    expect(bad == 0U, "CB4 products equal the host Q8_0 products of the same weights bitwise (" + std::to_string(bad) + " of " +
        std::to_string(total) + " differ)");
    std::cout << "CB4: products rows 1/3/8 " << total - bad << "/" << total << " bit-exact\n";
}

__global__ void declared_functions_kernel(const float* in, unsigned n, float* out) {
    const unsigned i = blockIdx.x * 256U + threadIdx.x;
    if (i >= n) return;
    out[3U * i] = strix::ds41::declared_exp(in[i]);
    out[3U * i + 1U] = strix::ds41::declared_log1p(fabsf(in[i]));
    out[3U * i + 2U] = strix::ds41::router_score(in[i]);
}
void test_declared_functions() {
    std::vector<float> in;
    for (int i = -40000; i <= 40000; ++i) in.push_back(static_cast<float>(i) * 1.37e-3F);
    for (int i = -149; i < 127; ++i) in.push_back(std::ldexp(1.25F, i));
    const auto n = static_cast<unsigned>(in.size());
    Device<float> din(n), dout(3U * n);
    din.upload(in.data(), n);
    hipLaunchKernelGGL(declared_functions_kernel, dim3((n + 255U) / 256U), dim3(256), 0, nullptr, din.data, n, dout.data);
    check(hipDeviceSynchronize(), "declared functions");
    const auto got = dout.download();
    std::size_t bad[3] = {0, 0, 0};
    std::size_t first[3] = {n, n, n};
    for (unsigned i = 0; i < n; ++i) {
        const float host[3] = {strix::ds41::declared_exp(in[i]), strix::ds41::declared_log1p(std::fabs(in[i])), strix::ds41::router_score(in[i])};
        for (unsigned k = 0; k < 3U; ++k)
            if (std::bit_cast<std::uint32_t>(host[k]) != std::bit_cast<std::uint32_t>(got[3U * i + k])) { ++bad[k]; if (first[k] == n) first[k] = i; }
    }
    expect(bad[0] == 0U && bad[1] == 0U && bad[2] == 0U, "declared exp/log1p/router score: device equals host bitwise (" +
        std::to_string(bad[0]) + "/" + std::to_string(bad[1]) + "/" + std::to_string(bad[2]) + " differ, first at " +
        (first[2] < n ? std::to_string(in[first[2]]) : first[1] < n ? std::to_string(in[first[1]]) : first[0] < n ? std::to_string(in[first[0]]) : std::string("-")) + ")");
}

void test_scalar_tables() {
    using namespace strix::ds41;
    Device<float> out(784);
    enqueue_decode_scalar_tables(out.data, nullptr);
    check(hipDeviceSynchronize(), "scalar tables");
    const auto device = out.download();
    std::vector<float> host(784);
    for (unsigned c = 0; c < 256U; ++c) {
        host[c] = e4m3_to_float(static_cast<std::uint8_t>(c));
        host[256U + c] = e8m0_to_float(static_cast<std::uint8_t>(c));
        host[512U + c] = e8m0_half_to_float(static_cast<std::uint8_t>(c));
    }
    for (unsigned c = 0; c < 16U; ++c) host[768U + c] = e2m1_to_float(static_cast<std::uint8_t>(c));
    expect(mismatches(device, host) == 0U, "E4M3, E8M0, E8M0/2 and E2M1 device tables equal the host tables bitwise");
}

std::vector<std::uint8_t> synthetic(WeightFormat format, std::size_t blocks, unsigned seed) {
    std::mt19937 random(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    const auto size = strix::ds41::block_bytes(format);
    std::vector<std::uint8_t> bytes(blocks * size);
    for (auto& b : bytes) b = static_cast<std::uint8_t>(byte(random));
    for (std::size_t b = 0; b < blocks; ++b) {
        auto* p = bytes.data() + b * size;
        // Finite FP16 scales, including subnormals; E8M0 exponents over the whole finite range.
        const auto finite_half = [&](std::uint8_t* h) { if ((h[1] & 0x7cU) == 0x7cU) h[1] = static_cast<std::uint8_t>(h[1] & 0x83U); };
        if (format == WeightFormat::iq2_xxs || format == WeightFormat::q8_0) finite_half(p);
        else if (format == WeightFormat::q2_k) { finite_half(p + 80); finite_half(p + 82); }
        else if (p[0] == 255U) p[0] = 254U;
    }
    return bytes;
}

void test_synthetic() {
    check_decode(WeightFormat::iq2_xxs, synthetic(WeightFormat::iq2_xxs, 4096, 1), "synthetic IQ2_XXS");
    check_decode(WeightFormat::q2_k, synthetic(WeightFormat::q2_k, 4096, 2), "synthetic Q2_K");
    check_decode(WeightFormat::mxfp4, synthetic(WeightFormat::mxfp4, 8192, 3), "synthetic MXFP4");
    check_decode(WeightFormat::q8_0, synthetic(WeightFormat::q8_0, 8192, 7), "synthetic Q8_0");
    auto q8 = synthetic(WeightFormat::q8_0, 3U * 40U * 160U, 8);
    for (std::size_t b = 0; b < q8.size() / 34U; ++b) q8[34U * b + 1U] = static_cast<std::uint8_t>(0x10U | (q8[34U * b + 1U] & 0x03U));
    check_products(WeightFormat::q8_0, q8, 3, 5120, 40, "synthetic Q8_0", 14);
    check_cb4_products();
    // Small scales keep the products finite: rescale the random FP16 scales into [2^-14, 2^-6).
    auto iq = synthetic(WeightFormat::iq2_xxs, 3U * 40U * 20U, 4);
    for (std::size_t b = 0; b < iq.size() / 66U; ++b) iq[66U * b + 1U] = static_cast<std::uint8_t>(0x04U | (iq[66U * b + 1U] & 0x1bU));
    check_products(WeightFormat::iq2_xxs, iq, 3, 5120, 40, "synthetic IQ2_XXS", 11);
    auto q2 = synthetic(WeightFormat::q2_k, 3U * 40U * 9U, 5);
    for (std::size_t b = 0; b < q2.size() / 84U; ++b) {
        q2[84U * b + 81U] = static_cast<std::uint8_t>(0x04U | (q2[84U * b + 81U] & 0x1bU));
        q2[84U * b + 83U] = static_cast<std::uint8_t>(0x04U | (q2[84U * b + 83U] & 0x1bU));
    }
    check_products(WeightFormat::q2_k, q2, 3, 2304, 40, "synthetic Q2_K", 12);
    auto mx = synthetic(WeightFormat::mxfp4, 3U * 40U * 160U, 6);
    for (std::size_t b = 0; b < mx.size() / 17U; ++b) mx[17U * b] = static_cast<std::uint8_t>(100U + mx[17U * b] % 40U);
    check_products(WeightFormat::mxfp4, mx, 3, 5120, 40, "synthetic MXFP4", 13);
}

std::vector<std::uint8_t> read_range(const std::string& path, std::uint64_t offset, std::uint64_t bytes) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path);
    file.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::uint8_t> out(bytes);
    file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    if (!file) throw std::runtime_error("short read in " + path);
    return out;
}

struct ExpertTensor {
    const strix::gguf::TensorInfo* info = nullptr;
    WeightFormat format{};
    std::uint32_t width = 0, outputs = 0, experts = 0;
    std::uint64_t row_bytes = 0, stride = 0;
};
ExpertTensor expert_tensor(const std::map<std::string, const strix::gguf::TensorInfo*>& tensors, const std::string& name) {
    const auto found = tensors.find(name);
    if (found == tensors.end()) throw std::runtime_error("missing tensor " + name);
    ExpertTensor t;
    t.info = found->second;
    t.format = t.info->ggml_type == 16U ? WeightFormat::iq2_xxs : t.info->ggml_type == 10U ? WeightFormat::q2_k
        : t.info->ggml_type == 8U ? WeightFormat::q8_0 : WeightFormat::mxfp4;
    if (t.info->ggml_type != 16U && t.info->ggml_type != 10U && t.info->ggml_type != 39U && t.info->ggml_type != 8U)
        throw std::runtime_error(name + " has an unexpected type");
    t.width = static_cast<std::uint32_t>(t.info->dimensions.at(0));
    t.outputs = static_cast<std::uint32_t>(t.info->dimensions.at(1));
    t.experts = t.info->dimensions.size() > 2U ? static_cast<std::uint32_t>(t.info->dimensions[2]) : 1U;
    t.row_bytes = t.width / strix::ds41::block_elements(t.format) * strix::ds41::block_bytes(t.format);
    t.stride = t.row_bytes * t.outputs;
    if (t.stride * t.experts != t.info->byte_size) throw std::runtime_error(name + " byte size does not match its geometry");
    return t;
}

void test_real_experts(const std::string& path, const std::vector<std::string>& names, const std::vector<std::uint32_t>& experts) {
    const auto file = strix::gguf::read_file(path);
    std::map<std::string, const strix::gguf::TensorInfo*> tensors;
    for (const auto& t : file.tensors) tensors[t.name] = &t;
    for (const auto& name : names) {
        const auto t = expert_tensor(tensors, name);
        std::vector<std::uint8_t> slabs;
        for (const auto e : experts) {
            const auto bytes = read_range(path, t.info->absolute_offset + e * t.stride, t.stride);
            slabs.insert(slabs.end(), bytes.begin(), bytes.end());
        }
        const auto label = "real " + name;
        const bool decoded = check_decode(t.format, slabs, label);
        std::cout << label << ": " << slabs.size() / strix::ds41::block_bytes(t.format) << " blocks of experts";
        for (const auto e : experts) std::cout << ' ' << e;
        std::cout << (decoded ? " decoded bit-exact\n" : " DECODE MISMATCH\n");
        check_products(t.format, slabs, static_cast<std::uint32_t>(experts.size()), t.width, t.outputs, label, 21U + t.width);
    }
}

void test_real_engram(const std::string& path) {
    const auto file = strix::gguf::read_file(path);
    std::mt19937_64 random(99U);
    for (const auto& t : file.tensors) {
        if (t.ggml_type != 24U || t.dimensions.size() != 2U || t.dimensions[0] != strix::ds41::engram_row_bytes) continue;
        const auto rows = t.dimensions[1];
        constexpr std::size_t sample = 4096;
        std::vector<std::uint8_t> bytes;
        bytes.reserve(sample * strix::ds41::engram_row_bytes);
        for (std::size_t i = 0; i < sample; ++i) {
            const auto row = random() % rows;
            const auto one = read_range(path, t.absolute_offset + row * strix::ds41::engram_row_bytes, strix::ds41::engram_row_bytes);
            bytes.insert(bytes.end(), one.begin(), one.end());
        }
        Device<std::uint8_t> device(bytes.size());
        device.upload(bytes.data(), bytes.size());
        Device<float> out(sample * strix::ds41::engram_row_elements);
        strix::ds41::enqueue_decode_engram_rows(device.data, sample, out.data, nullptr);
        check(hipDeviceSynchronize(), "engram decode");
        std::vector<float> host(sample * strix::ds41::engram_row_elements);
        std::size_t nan_codes = 0, scale_range[2] = {255, 0};
        for (std::size_t i = 0; i < sample; ++i) {
            const auto* row = bytes.data() + i * strix::ds41::engram_row_bytes;
            strix::ds41::decode_engram_row(std::span<const std::uint8_t, 264>(row, 264),
                std::span<float, 256>(host.data() + i * 256U, 256U));
            for (unsigned j = 0; j < 256U; ++j) nan_codes += (row[j] & 0x7fU) == 0x7fU;
            for (unsigned j = 256U; j < 264U; ++j) {
                scale_range[0] = std::min<std::size_t>(scale_range[0], row[j]);
                scale_range[1] = std::max<std::size_t>(scale_range[1], row[j]);
            }
        }
        const auto bad = mismatches(out.download(), host);
        expect(bad == 0U, "real " + t.name + ": Engram rows decode bitwise (" + std::to_string(bad) + " differ)");
        std::cout << "real " << t.name << ": " << sample << " random rows of " << rows << " decoded "
                  << (bad ? "WITH MISMATCHES" : "bit-exact") << "; E4M3 NaN codes " << nan_codes << ", E8M0 scales "
                  << scale_range[0] << ".." << scale_range[1] << '\n';
    }
}


// ---- Routed-expert windows: fused gate/up SwiGLU, down raw slot vectors, combine ----

struct DeviceRoutes {
    Device<std::uint32_t> experts, begins, rows, slots, order, low_counts;
    std::uint32_t expert_count, max_rows;
    explicit DeviceRoutes(const strix::ds41::WindowRoutes& r)
        : experts(r.experts.size()), begins(r.begins.size()), rows(r.rows.size()), slots(r.slots.size()), order(r.order.size()),
          low_counts(r.low_counts.size()), expert_count(static_cast<std::uint32_t>(r.experts.size())), max_rows(r.max_rows) {
        experts.upload(r.experts.data(), r.experts.size()); begins.upload(r.begins.data(), r.begins.size());
        rows.upload(r.rows.data(), r.rows.size()); slots.upload(r.slots.data(), r.slots.size());
        order.upload(r.order.data(), r.order.size()); low_counts.upload(r.low_counts.data(), r.low_counts.size());
    }
    strix::ds41::RoutesView gate_up() const { return {experts.data, begins.data, rows.data, slots.data, expert_count, max_rows}; }
    strix::ds41::RoutesView down() const { return {experts.data, begins.data, slots.data, slots.data, expert_count, max_rows}; }
};

// Expert tensors resident on the device.
struct DeviceExperts {
    strix::ds41::ExpertTensors host;
    Device<std::uint8_t> gate, up, down;
    explicit DeviceExperts(const strix::ds41::ExpertTensors& t) : host(t), gate(t.gate.size()), up(t.up.size()), down(t.down.size()) {
        gate.upload(t.gate.data(), t.gate.size()); up.upload(t.up.data(), t.up.size()); down.upload(t.down.data(), t.down.size());
    }
};

struct WindowBuffers {
    Device<std::int8_t> codes; Device<float> deltas; Device<std::int32_t> sums;
    Device<float> gate, up; Device<std::int8_t> act_codes; Device<float> act_deltas; Device<std::int32_t> act_sums;
    Device<float> raw, shared, weights, y;
    WindowBuffers(std::uint32_t rows, std::uint32_t slots, std::uint32_t hidden, std::uint32_t inter)
        : codes(static_cast<std::size_t>(rows) * hidden), deltas(rows * hidden / 32U), sums(rows * hidden / 32U),
          gate(static_cast<std::size_t>(slots) * inter), up(static_cast<std::size_t>(slots) * inter),
          act_codes(static_cast<std::size_t>(slots) * inter), act_deltas(slots * inter / 32U), act_sums(slots * inter / 32U),
          raw(static_cast<std::size_t>(slots) * hidden), shared(static_cast<std::size_t>(rows) * hidden), weights(slots),
          y(static_cast<std::size_t>(rows) * hidden) {}
};

struct WindowTimes { float gate_up = 0, down = 0, combine = 0; };
// Enqueues the three window launches; with events, records the time of each.
void enqueue_window(const DeviceExperts& e, const DeviceRoutes& routes, WindowBuffers& b, std::uint32_t rows, std::uint32_t top_k,
                    bool keep_gate_up, hipEvent_t* events = nullptr) {
    using namespace strix::ds41;
    const auto& t = e.host;
    if (events) check(hipEventRecord(events[0], nullptr), "event");
    GateUpView gu;
    gu.gate = e.gate.data; gu.up = e.up.data; gu.expert_stride_bytes = t.gate_row_bytes() * t.inter; gu.row_bytes = t.gate_row_bytes();
    gu.hidden = t.hidden; gu.inter = t.inter; gu.routes = routes.gate_up(); gu.codes = b.codes.data; gu.deltas = b.deltas.data;
    gu.limit = t.limit; gu.gate_out = keep_gate_up ? b.gate.data : nullptr; gu.up_out = keep_gate_up ? b.up.data : nullptr;
    gu.act_codes = b.act_codes.data; gu.act_deltas = b.act_deltas.data; gu.act_sums = b.act_sums.data;
    enqueue_gate_up(t.gate_up_format, gu, nullptr);
    if (events) check(hipEventRecord(events[1], nullptr), "event");
    RoutesProductView down;
    down.weights = e.down.data; down.expert_stride_bytes = t.down_row_bytes() * t.hidden; down.row_bytes = t.down_row_bytes();
    down.width = t.inter; down.outputs = t.hidden; down.routes = routes.down(); down.codes = b.act_codes.data;
    down.deltas = b.act_deltas.data; down.out = b.raw.data;
    enqueue_routes_product(t.down_format, down, nullptr);
    if (events) check(hipEventRecord(events[2], nullptr), "event");
    CombineView c;
    c.raw = b.raw.data; c.shared = b.shared.data; c.weights = b.weights.data; c.order = routes.order.data;
    c.low_counts = routes.low_counts.data; c.rows = rows; c.top_k = top_k; c.hidden = t.hidden; c.out = b.y.data;
    enqueue_combine(c, nullptr);
    if (events) check(hipEventRecord(events[3], nullptr), "event");
}

struct WindowCase {
    std::vector<std::uint32_t> ids;
    std::vector<float> x, weights, shared;
    std::uint32_t rows = 0, top_k = 0;
};
// Routing of `rows` rows: independent distinct experts per row, or (overlap) the first `common`
// experts shared by every row.
WindowCase window_case(std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts, std::uint32_t hidden, std::uint32_t common,
                       unsigned seed) {
    std::mt19937 random(seed);
    WindowCase c;
    c.rows = rows; c.top_k = top_k;
    std::vector<std::uint32_t> all(experts);
    std::iota(all.begin(), all.end(), 0U);
    std::shuffle(all.begin(), all.end(), random);
    const std::vector<std::uint32_t> shared_ids(all.begin(), all.begin() + common);
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::vector<std::uint32_t> row(shared_ids);
        std::vector<std::uint32_t> rest(all.begin() + common, all.end());
        std::shuffle(rest.begin(), rest.end(), random);
        row.insert(row.end(), rest.begin(), rest.begin() + (top_k - common));
        std::shuffle(row.begin(), row.end(), random);
        c.ids.insert(c.ids.end(), row.begin(), row.end());
    }
    c.x = activations(rows, hidden, seed + 1U);
    std::uniform_real_distribution<float> weight(0.05F, 0.6F);
    for (std::uint32_t s = 0; s < rows * top_k; ++s) c.weights.push_back(weight(random));
    std::normal_distribution<float> normal(0.0F, 0.5F);
    for (std::size_t i = 0; i < static_cast<std::size_t>(rows) * hidden; ++i) c.shared.push_back(normal(random));
    return c;
}

std::size_t bit_mismatches(const std::vector<float>& a, const std::vector<float>& b) { return mismatches(a, b); }

// Runs one window on the device and on the host and compares every intermediate bitwise.
void check_window(const DeviceExperts& e, const WindowCase& c, std::uint32_t split, const std::string& label) {
    using namespace strix::ds41;
    const auto& t = e.host;
    const auto routes = build_window_routes(c.ids, c.rows, c.top_k, split);
    DeviceRoutes device_routes(routes);
    const auto slots = c.rows * c.top_k;
    WindowBuffers b(c.rows, slots, t.hidden, t.inter);
    Device<float> x(c.x.size());
    x.upload(c.x.data(), c.x.size());
    enqueue_quantize_a8(x.data, t.hidden, c.rows, b.codes.data, b.deltas.data, b.sums.data, nullptr);
    b.weights.upload(c.weights.data(), c.weights.size());
    b.shared.upload(c.shared.data(), c.shared.size());
    enqueue_window(e, device_routes, b, c.rows, c.top_k, true);
    check(hipDeviceSynchronize(), "window");
    std::vector<A8Activation> xs;
    for (std::uint32_t r = 0; r < c.rows; ++r) xs.push_back(quantize_a8(std::span<const float>(c.x.data() + static_cast<std::size_t>(r) * t.hidden, t.hidden)));
    const auto ref = reference_window(t, xs, routes, c.weights, c.shared, c.top_k);
    const auto device_codes = b.act_codes.download();
    const auto device_deltas = b.act_deltas.download();
    const auto device_sums = b.act_sums.download();
    std::size_t act_bad = 0;
    for (std::uint32_t s = 0; s < slots; ++s) {
        const auto& a = ref.activation[s];
        for (std::uint32_t i = 0; i < t.inter; ++i) act_bad += a.codes[i] != device_codes[static_cast<std::size_t>(s) * t.inter + i];
        for (std::uint32_t g = 0; g < t.inter / 32U; ++g) {
            act_bad += std::bit_cast<std::uint32_t>(a.deltas[g]) != std::bit_cast<std::uint32_t>(device_deltas[s * t.inter / 32U + g]);
            act_bad += a.sums[g] != device_sums[s * t.inter / 32U + g];
        }
    }
    const auto gate_bad = bit_mismatches(b.gate.download(), ref.gate) + bit_mismatches(b.up.download(), ref.up);
    const auto raw_bad = bit_mismatches(b.raw.download(), ref.raw);
    const auto y_bad = bit_mismatches(b.y.download(), ref.y);
    const bool ok = act_bad == 0U && gate_bad == 0U && raw_bad == 0U && y_bad == 0U;
    expect(ok, label + ": window gate/up " + std::to_string(gate_bad) + ", activations " + std::to_string(act_bad) + ", raw " +
        std::to_string(raw_bad) + ", y " + std::to_string(y_bad) + " differ");
    std::cout << label << ": rows " << c.rows << ", union " << routes.experts.size() << " experts, max " << routes.max_rows
              << " rows/expert: gate/up, A8 activations, raw slot vectors and y " << (ok ? "bit-exact" : "MISMATCH") << '\n';
}

strix::ds41::ExpertTensors synthetic_experts(strix::ds41::WeightFormat gate_up, strix::ds41::WeightFormat down, std::uint32_t experts,
                                             std::uint32_t hidden, std::uint32_t inter, std::vector<std::uint8_t> (&storage)[3],
                                             unsigned seed) {
    using namespace strix::ds41;
    ExpertTensors t;
    t.gate_up_format = gate_up; t.down_format = down; t.hidden = hidden; t.inter = inter;
    const auto blocks_gate = static_cast<std::size_t>(experts) * inter * (hidden / block_elements(gate_up));
    const auto blocks_down = static_cast<std::size_t>(experts) * hidden * (inter / block_elements(down));
    const auto tame = [](WeightFormat f, std::vector<std::uint8_t>& bytes) {
        const auto size = block_bytes(f);
        for (std::size_t b = 0; b < bytes.size() / size; ++b) {
            auto* p = bytes.data() + b * size;
            if (f == WeightFormat::iq2_xxs) p[1] = static_cast<std::uint8_t>(0x10U | (p[1] & 0x03U));
            else if (f == WeightFormat::q2_k) { p[81] = static_cast<std::uint8_t>(0x10U | (p[81] & 0x03U)); p[83] = static_cast<std::uint8_t>(0x10U | (p[83] & 0x03U)); }
            else p[0] = static_cast<std::uint8_t>(118U + p[0] % 8U);
        }
    };
    storage[0] = synthetic(gate_up, blocks_gate, seed); tame(gate_up, storage[0]);
    storage[1] = synthetic(gate_up, blocks_gate, seed + 1U); tame(gate_up, storage[1]);
    storage[2] = synthetic(down, blocks_down, seed + 2U); tame(down, storage[2]);
    t.gate = storage[0]; t.up = storage[1]; t.down = storage[2];
    return t;
}

void test_synthetic_windows() {
    using strix::ds41::WeightFormat;
    std::vector<std::uint8_t> storage[3];
    {
        const DeviceExperts e(synthetic_experts(WeightFormat::iq2_xxs, WeightFormat::q2_k, 16, 512, 256, storage, 31));
        for (const std::uint32_t rows : {1U, 2U, 3U, 5U, 8U}) check_window(e, window_case(rows, 6, 16, 512, rows % 2U ? 0U : 3U, 40U + rows), 8U, "synthetic IQ2_XXS/Q2_K window");
    }
    {
        const DeviceExperts e(synthetic_experts(WeightFormat::mxfp4, WeightFormat::mxfp4, 12, 512, 256, storage, 51));
        for (const std::uint32_t rows : {1U, 4U, 6U}) check_window(e, window_case(rows, 3, 12, 512, rows == 4U ? 2U : 0U, 60U + rows), 6U, "synthetic MXFP4 window");
    }
}

strix::ds41::ExpertTensors file_experts(const std::string& path, const std::string& prefix, std::vector<std::uint8_t> (&storage)[3]) {
    const auto file = strix::gguf::read_file(path);
    std::map<std::string, const strix::gguf::TensorInfo*> tensors;
    for (const auto& t : file.tensors) tensors[t.name] = &t;
    const auto gate = expert_tensor(tensors, prefix + "ffn_gate_exps.weight");
    const auto up = expert_tensor(tensors, prefix + "ffn_up_exps.weight");
    const auto down = expert_tensor(tensors, prefix + "ffn_down_exps.weight");
    storage[0] = read_range(path, gate.info->absolute_offset, gate.info->byte_size);
    storage[1] = read_range(path, up.info->absolute_offset, up.info->byte_size);
    storage[2] = read_range(path, down.info->absolute_offset, down.info->byte_size);
    strix::ds41::ExpertTensors t;
    t.gate_up_format = gate.format; t.down_format = down.format; t.hidden = gate.width; t.inter = gate.outputs;
    t.gate = storage[0]; t.up = storage[1]; t.down = storage[2];
    return t;
}

void test_real_windows(const std::string& q2, const std::string& dspark) {
    std::vector<std::uint8_t> storage[3];
    if (!q2.empty()) {
        const DeviceExperts e(file_experts(q2, "blk.20.", storage));
        check_window(e, window_case(1, 6, 384, e.host.hidden, 0, 71), 192U, "real blk.20 window");
        check_window(e, window_case(3, 6, 384, e.host.hidden, 2, 72), 192U, "real blk.20 window");
        check_window(e, window_case(8, 6, 384, e.host.hidden, 4, 73), 192U, "real blk.20 window");
    }
    if (!dspark.empty()) {
        const DeviceExperts e(file_experts(dspark, "mtp.0.", storage));
        check_window(e, window_case(1, 3, 128, e.host.hidden, 0, 81), 64U, "real mtp.0 window");
        check_window(e, window_case(6, 3, 128, e.host.hidden, 1, 82), 64U, "real mtp.0 window");
    }
}

// ---- Cold timings ----

// Reads (never dirties) a buffer much larger than the caches, so the timed kernels start cold and
// without pending write-backs.
__global__ void flush_kernel(int* buffer, std::size_t count, int value) {
    int sum = 0;
    for (std::size_t i = blockIdx.x * 256ULL + threadIdx.x; i < count; i += gridDim.x * 256ULL) sum += buffer[i];
    if (sum == value) buffer[0] = sum;
}
struct Flush {
    Device<int> buffer{64U * 1024U * 1024U}; // 256 MB, eight times the last-level cache
    Flush() { check(hipMemset(buffer.data, 0, buffer.count * sizeof(int)), "flush buffer"); check(hipDeviceSynchronize(), "flush buffer"); }
    void operator()() {
        hipLaunchKernelGGL(flush_kernel, dim3(1024), dim3(256), 0, nullptr, buffer.data, buffer.count, 0x5a5a5a5a);
        check(hipGetLastError(), "flush");
    }
};

float median_of(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2U]; }

void bench_windows(const DeviceExperts& e, std::uint32_t top_k, std::uint32_t experts, std::uint32_t split, const std::string& label) {
    const auto& t = e.host;
    Flush flush;
    hipEvent_t events[4];
    for (auto& ev : events) check(hipEventCreate(&ev), "event");
    const double gate_expert = static_cast<double>(t.gate_row_bytes()) * t.inter * 2.0;
    const double down_expert = static_cast<double>(t.down_row_bytes()) * t.hidden;
    for (const std::uint32_t common : {0U, top_k}) {
        for (const std::uint32_t rows : {1U, 2U, 4U, 6U, 8U}) {
            const auto c = window_case(rows, top_k, experts, t.hidden, common, 90U + rows);
            const auto routes = strix::ds41::build_window_routes(c.ids, rows, top_k, split);
            DeviceRoutes device_routes(routes);
            WindowBuffers b(rows, rows * top_k, t.hidden, t.inter);
            Device<float> x(c.x.size());
            x.upload(c.x.data(), c.x.size());
            strix::ds41::enqueue_quantize_a8(x.data, t.hidden, rows, b.codes.data, b.deltas.data, b.sums.data, nullptr);
            b.weights.upload(c.weights.data(), c.weights.size());
            b.shared.upload(c.shared.data(), c.shared.size());
            std::vector<float> gu, dn, cb;
            for (unsigned rep = 0; rep < 23U; ++rep) {
                flush();
                enqueue_window(e, device_routes, b, rows, top_k, false, events);
                check(hipEventSynchronize(events[3]), "sync");
                float a = 0, d = 0, m = 0;
                check(hipEventElapsedTime(&a, events[0], events[1]), "elapsed");
                check(hipEventElapsedTime(&d, events[1], events[2]), "elapsed");
                check(hipEventElapsedTime(&m, events[2], events[3]), "elapsed");
                if (rep >= 3U) { gu.push_back(a); dn.push_back(d); cb.push_back(m); }
            }
            const auto n = static_cast<double>(routes.experts.size());
            const float g = median_of(gu), d = median_of(dn), m = median_of(cb);
            std::cout << "WINDOW " << label << (common ? " shared-top" : " independent") << " rows=" << rows << " union=" << n
                      << " gate_up_ms=" << g << " (" << n * gate_expert / (g * 1e6) << " GB/s) down_ms=" << d << " ("
                      << n * down_expert / (d * 1e6) << " GB/s) combine_ms=" << m << " total_ms=" << g + d + m << " total_GB/s="
                      << n * (gate_expert + down_expert) / ((g + d + m) * 1e6) << '\n';
        }
    }
    for (auto& ev : events) (void)hipEventDestroy(ev);
}


// ---- Whole MoE layer: router, shared expert, routed experts, combine; and two ranks ----

strix::ds41::ExpertTensors named_tensors(const std::string& path, const std::string& gate, const std::string& up,
                                         const std::string& down, std::vector<std::uint8_t> (&storage)[3]) {
    const auto file = strix::gguf::read_file(path);
    std::map<std::string, const strix::gguf::TensorInfo*> tensors;
    for (const auto& t : file.tensors) tensors[t.name] = &t;
    const auto g = expert_tensor(tensors, gate), u = expert_tensor(tensors, up), d = expert_tensor(tensors, down);
    storage[0] = read_range(path, g.info->absolute_offset, g.info->byte_size);
    storage[1] = read_range(path, u.info->absolute_offset, u.info->byte_size);
    storage[2] = read_range(path, d.info->absolute_offset, d.info->byte_size);
    strix::ds41::ExpertTensors t;
    t.gate_up_format = g.format; t.down_format = d.format; t.hidden = g.width; t.inter = g.outputs;
    t.gate = storage[0]; t.up = storage[1]; t.down = storage[2];
    return t;
}
std::vector<float> read_floats(const std::string& path, const std::string& name, std::size_t count) {
    const auto file = strix::gguf::read_file(path);
    for (const auto& t : file.tensors)
        if (t.name == name) {
            if (t.ggml_type != 0U || t.byte_size != count * 4U) throw std::runtime_error(name + " is not the expected F32 tensor");
            const auto bytes = read_range(path, t.absolute_offset, t.byte_size);
            std::vector<float> out(count);
            std::memcpy(out.data(), bytes.data(), bytes.size());
            return out;
        }
    throw std::runtime_error("missing tensor " + name);
}

struct Layer {
    std::vector<float> router, bias, bias_image;
    std::uint32_t experts = 0, top_k = 0, split = 0;
    float scale = 1.5F;
    std::vector<std::uint8_t> shared_storage[3], routed_storage[3];
    std::unique_ptr<DeviceExperts> shared, routed;
    std::unique_ptr<Device<float>> router_device, bias_device, bias_image_device;
};
std::unique_ptr<Layer> load_layer(const std::string& path, const std::string& prefix, std::uint32_t experts, std::uint32_t top_k) {
    auto layer = std::make_unique<Layer>();
    layer->experts = experts; layer->top_k = top_k; layer->split = experts / 2U;
    layer->routed = std::make_unique<DeviceExperts>(named_tensors(path, prefix + "ffn_gate_exps.weight", prefix + "ffn_up_exps.weight",
                                                                  prefix + "ffn_down_exps.weight", layer->routed_storage));
    layer->shared = std::make_unique<DeviceExperts>(named_tensors(path, prefix + "ffn_gate_shexp.weight", prefix + "ffn_up_shexp.weight",
                                                                  prefix + "ffn_down_shexp.weight", layer->shared_storage));
    const auto hidden = layer->routed->host.hidden;
    layer->router = read_floats(path, prefix + "ffn_gate_inp.weight", static_cast<std::size_t>(experts) * hidden);
    layer->bias = read_floats(path, prefix + "exp_probs_b.bias", experts);
    layer->bias_image = read_floats(path, prefix + "exp_probs_b_vl.bias", experts);
    layer->router_device = std::make_unique<Device<float>>(layer->router.size());
    layer->router_device->upload(layer->router.data(), layer->router.size());
    layer->bias_device = std::make_unique<Device<float>>(experts);
    layer->bias_device->upload(layer->bias.data(), experts);
    layer->bias_image_device = std::make_unique<Device<float>>(experts);
    layer->bias_image_device->upload(layer->bias_image.data(), experts);
    return layer;
}

// Routes built on the device from device ids, sized for rows * top_k entries.
struct DeviceBuiltRoutes {
    Device<std::uint32_t> experts, begins, rows, slots, order, low_counts, counts;
    std::uint32_t bound, staged;
    DeviceBuiltRoutes(std::uint32_t row_count, std::uint32_t top_k)
        : experts(row_count * top_k), begins(row_count * top_k + 1U), rows(row_count * top_k), slots(row_count * top_k),
          order(row_count * top_k), low_counts(row_count), counts(2), bound(row_count * top_k),
          staged(row_count == 1U ? 1U : row_count == 2U ? 2U : 4U) {}
    void build(const std::uint32_t* ids, std::uint32_t row_count, std::uint32_t top_k, std::uint32_t split, std::uint32_t first,
               std::uint32_t end, hipStream_t stream, std::uint32_t base = 0U) {
        strix::ds41::BuildRoutesView v;
        v.ids = ids; v.rows = row_count; v.top_k = top_k; v.split = split; v.first = first; v.end = end; v.base = base;
        v.experts = experts.data; v.begins = begins.data; v.rows_out = rows.data; v.slots = slots.data; v.order = order.data;
        v.low_counts = low_counts.data; v.counts = counts.data;
        strix::ds41::enqueue_build_routes(v, stream);
    }
    strix::ds41::RoutesView gate_up() const { return {experts.data, begins.data, rows.data, slots.data, bound, staged, counts.data}; }
    strix::ds41::RoutesView down() const { return {experts.data, begins.data, slots.data, slots.data, bound, staged, counts.data}; }
};

void test_device_routes() {
    using namespace strix::ds41;
    std::size_t bad = 0, cases = 0;
    for (const std::uint32_t rows : {1U, 2U, 3U, 6U, 8U})
        for (const std::uint32_t common : {0U, 3U, 6U})
            for (const std::uint32_t half : {0U, 1U, 2U}) {
                const auto c = window_case(rows, 6U, 384U, 512U, common, 500U + rows * 7U + common);
                const auto all = build_window_routes(c.ids, rows, 6U, 192U);
                const std::uint32_t first = half == 1U ? 192U : 0U, end = half == 0U ? 192U : 384U;
                const auto expected = half == 2U ? all : restrict_routes(all, half == 0U ? 0U : 192U, half == 0U ? 192U : 384U);
                Device<std::uint32_t> ids(c.ids.size());
                ids.upload(c.ids.data(), c.ids.size());
                DeviceBuiltRoutes routes(rows, 6U);
                routes.build(ids.data, rows, 6U, 192U, half == 2U ? 0U : first, half == 2U ? 384U : end, nullptr);
                check(hipDeviceSynchronize(), "build routes");
                const auto counts = routes.counts.download();
                const auto n = counts[0];
                auto prefix = [](const std::vector<std::uint32_t>& v, std::size_t k) { return std::vector<std::uint32_t>(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k)); };
                const bool ok = n == expected.experts.size() && counts[1] == expected.max_rows &&
                    prefix(routes.experts.download(), n) == expected.experts && prefix(routes.begins.download(), n + 1U) == expected.begins &&
                    prefix(routes.rows.download(), expected.rows.size()) == expected.rows &&
                    prefix(routes.slots.download(), expected.slots.size()) == expected.slots && routes.order.download() == expected.order &&
                    routes.low_counts.download() == expected.low_counts;
                bool rebased_ok = true;
                if (half == 1U) {
                    // Rebased expert indices (base = first) for a rank holding only its slice.
                    routes.build(ids.data, rows, 6U, 192U, first, end, nullptr, first);
                    check(hipDeviceSynchronize(), "build routes");
                    auto rebased = prefix(routes.experts.download(), n);
                    for (auto& e : rebased) e += first;
                    rebased_ok = rebased == expected.experts && routes.order.download() == expected.order;
                }
                bad += ok && rebased_ok ? 0U : 1U;
                ++cases;
            }
    expect(bad == 0U, "device-built window routes equal the host routes (" + std::to_string(bad) + " of " + std::to_string(cases) + " differ)");
    std::cout << "device window routes: " << cases - bad << "/" << cases << " cases equal the host routes\n";
}

struct LayerResult {
    std::vector<float> scores, weights, y;
    std::vector<std::uint32_t> ids;
    double exchange_seconds = 0.0;
};

// Device state of one layer run on one rank: buffers, routes and the shared expert's fixed routes.
struct LayerRun {
    const Layer& layer;
    std::uint32_t rows;
    hipStream_t stream;
    WindowBuffers b, sb;
    Device<float> x, scores, own, other;
    Device<std::uint8_t> image;
    Device<std::uint32_t> ids;
    DeviceBuiltRoutes routes;
    strix::ds41::WindowRoutes shared_host;
    DeviceRoutes shared_routes;
    LayerRun(const Layer& l, std::uint32_t row_count, hipStream_t s)
        : layer(l), rows(row_count), stream(s),
          b(row_count, row_count * l.top_k, l.routed->host.hidden, l.routed->host.inter),
          sb(row_count, row_count, l.routed->host.hidden, l.routed->host.inter),
          x(static_cast<std::size_t>(row_count) * l.routed->host.hidden), scores(static_cast<std::size_t>(row_count) * l.experts),
          own(static_cast<std::size_t>(row_count) * l.routed->host.hidden), other(static_cast<std::size_t>(row_count) * l.routed->host.hidden),
          image(row_count), ids(row_count * l.top_k), routes(row_count, l.top_k),
          shared_host(strix::ds41::build_window_routes(std::vector<std::uint32_t>(row_count, 0U), row_count, 1U, 1U)),
          shared_routes(shared_host) {}

    // Enqueues the whole layer (half < 0) or this rank's half; with a communicator, exchanges the half
    // sums and forms the canonical sum. Nothing is read back. With `overlap` (default; STRIX_DS41_EP_OVERLAP=0
    // turns it off) a rank enqueues its routed experts first, starts the exchange as soon as its half sum
    // is copied out, and computes the shared expert while the half sums travel.
    bool overlap = [] { const char* v = std::getenv("STRIX_DS41_EP_OVERLAP"); return !(v && v[0] == '0'); }();
    std::vector<hipEvent_t> marks;
    void mark() {
        if (!std::getenv("STRIX_DS41_PHASES")) return;
        hipEvent_t e{};
        check(hipEventCreate(&e), "phase event");
        check(hipEventRecord(e, stream), "phase event");
        marks.push_back(e);
    }
    void enqueue(const std::vector<float>& input, const std::vector<std::uint8_t>& image_rows, int half, strix::dist::Communicator* comm) {
        using namespace strix::ds41;
        mark();
        const auto& t = layer.routed->host;
        const auto hidden = t.hidden, top_k = layer.top_k;
        check(hipMemcpyAsync(x.data, input.data(), input.size() * 4U, hipMemcpyHostToDevice, stream), "upload x");
        check(hipMemcpyAsync(image.data, image_rows.data(), rows, hipMemcpyHostToDevice, stream), "upload image rows");
        mark();
        enqueue_quantize_a8(x.data, hidden, rows, b.codes.data, b.deltas.data, b.sums.data, stream);
        RouterView rv;
        rv.x = x.data; rv.weight = layer.router_device->data; rv.bias = layer.bias_device->data;
        rv.bias_image = layer.bias_image_device->data; rv.image_rows = image.data; rv.rows = rows; rv.hidden = hidden;
        rv.experts = layer.experts; rv.top_k = top_k; rv.route_scale = layer.scale; rv.scores = scores.data; rv.ids = ids.data;
        rv.weights = b.weights.data;
        enqueue_router(rv, stream);
        mark();
        const std::uint32_t first = half == 1 ? layer.split : 0U, end = half == 0 ? layer.split : layer.experts;
        routes.build(ids.data, rows, top_k, layer.split, first, end, stream);
        mark();
        // Shared expert: every row on expert 0 of the Q8_0 tensors.
        const auto shared_expert = [&] {
            check(hipMemcpyAsync(sb.codes.data, b.codes.data, static_cast<std::size_t>(rows) * hidden, hipMemcpyDeviceToDevice, stream), "codes");
            check(hipMemcpyAsync(sb.deltas.data, b.deltas.data, rows * hidden / 32U * 4U, hipMemcpyDeviceToDevice, stream), "deltas");
            const auto& st = layer.shared->host;
            GateUpView sgu;
            sgu.gate = layer.shared->gate.data; sgu.up = layer.shared->up.data; sgu.expert_stride_bytes = st.gate_row_bytes() * st.inter;
            sgu.row_bytes = st.gate_row_bytes(); sgu.hidden = st.hidden; sgu.inter = st.inter; sgu.routes = shared_routes.gate_up();
            sgu.codes = sb.codes.data; sgu.deltas = sb.deltas.data; sgu.limit = st.limit;
            sgu.act_codes = sb.act_codes.data; sgu.act_deltas = sb.act_deltas.data; sgu.act_sums = sb.act_sums.data;
            enqueue_gate_up(st.gate_up_format, sgu, stream);
            RoutesProductView sdown;
            sdown.weights = layer.shared->down.data; sdown.expert_stride_bytes = st.down_row_bytes() * st.hidden;
            sdown.row_bytes = st.down_row_bytes(); sdown.width = st.inter; sdown.outputs = st.hidden; sdown.routes = shared_routes.down();
            sdown.codes = sb.act_codes.data; sdown.deltas = sb.act_deltas.data; sdown.out = b.shared.data;
            enqueue_routes_product(st.down_format, sdown, stream);
            mark();
        };
        const bool overlapped = half >= 0 && overlap;
        if (!overlapped) shared_expert();
        // Routed experts of this rank (device routes; blocks past the expert count do nothing).
        check(hipMemsetAsync(b.raw.data, 0, b.raw.count * 4U, stream), "clear raw");
        GateUpView gu;
        gu.gate = layer.routed->gate.data; gu.up = layer.routed->up.data; gu.expert_stride_bytes = t.gate_row_bytes() * t.inter;
        gu.row_bytes = t.gate_row_bytes(); gu.hidden = hidden; gu.inter = t.inter; gu.routes = routes.gate_up();
        gu.codes = b.codes.data; gu.deltas = b.deltas.data; gu.limit = t.limit;
        gu.act_codes = b.act_codes.data; gu.act_deltas = b.act_deltas.data; gu.act_sums = b.act_sums.data;
        enqueue_gate_up(t.gate_up_format, gu, stream);
        mark();
        RoutesProductView down;
        down.weights = layer.routed->down.data; down.expert_stride_bytes = t.down_row_bytes() * hidden; down.row_bytes = t.down_row_bytes();
        down.width = t.inter; down.outputs = hidden; down.routes = routes.down(); down.codes = b.act_codes.data;
        down.deltas = b.act_deltas.data; down.out = b.raw.data;
        enqueue_routes_product(t.down_format, down, stream);
        mark();
        if (half < 0) {
            CombineView c;
            c.raw = b.raw.data; c.shared = b.shared.data; c.weights = b.weights.data; c.order = routes.order.data;
            c.low_counts = routes.low_counts.data; c.rows = rows; c.top_k = top_k; c.hidden = hidden; c.out = b.y.data;
            enqueue_combine(c, stream);
            mark();
            return;
        }
        HalfSumView hs;
        hs.raw = b.raw.data; hs.weights = b.weights.data; hs.order = routes.order.data; hs.low_counts = routes.low_counts.data;
        hs.rows = rows; hs.top_k = top_k; hs.hidden = hidden; hs.half = static_cast<std::uint32_t>(half); hs.out = own.data;
        enqueue_half_sum(hs, stream);
        if (overlapped) {
            comm->start(stream, own.data, own.count * 4U, strix::dist::Tag{1U, 20U, 7U});
            shared_expert();
            comm->finish(stream, other.data);
        } else {
            comm->exchange(stream, own.data, other.data, own.count * 4U, strix::dist::Tag{1U, 20U, 7U});
        }
        enqueue_canonical_sum(b.shared.data, half == 0 ? own.data : other.data, half == 0 ? other.data : own.data,
                              static_cast<std::size_t>(rows) * hidden, b.y.data, stream);
    }
    LayerResult result() {
        check(hipStreamSynchronize(stream), "layer");
        LayerResult out;
        out.ids = ids.download();
        out.weights = b.weights.download();
        out.scores = scores.download();
        out.y = b.y.download();
        return out;
    }
};

LayerResult run_layer(const Layer& layer, const std::vector<float>& x, std::uint32_t rows, const std::vector<std::uint8_t>& image_rows,
                      hipStream_t stream, int half, strix::dist::Communicator* comm) {
    LayerRun run(layer, rows, stream);
    run.enqueue(x, image_rows, half, comm);
    auto out = run.result();
    if (comm) out.exchange_seconds = comm->stats().seconds;
    return out;
}

void check_layer(const Layer& layer, std::uint32_t rows, const std::string& label) {
    using namespace strix::ds41;
    const auto& t = layer.routed->host;
    const auto x = activations(rows, t.hidden, 300U + rows);
    std::vector<std::uint8_t> image_rows(rows, 0U);
    if (rows > 1U) image_rows[rows - 1U] = 1U;
    hipStream_t stream{};
    check(hipStreamCreate(&stream), "stream");
    const auto single = run_layer(layer, x, rows, image_rows, stream, -1, nullptr);
    if (std::getenv("STRIX_DS41_PHASES")) {
        Flush flush;
        LayerRun timed(layer, rows, stream);
        for (int pass = 0; pass < 3; ++pass) {
            flush();
            check(hipStreamSynchronize(stream), "flush");
            for (auto e : timed.marks) (void)hipEventDestroy(e);
            timed.marks.clear();
            const auto start = std::chrono::steady_clock::now();
            timed.enqueue(x, image_rows, -1, nullptr);
            check(hipStreamSynchronize(stream), "timed layer");
            const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            std::cout << label << " phases rows " << rows << " wall_ms " << wall << " (upload, quantize+router, routes, shared, routed gate/up, routed down, combine):";
            for (std::size_t k = 1; k < timed.marks.size(); ++k) {
                float ms = 0;
                check(hipEventElapsedTime(&ms, timed.marks[k - 1], timed.marks[k]), "phase");
                std::cout << ' ' << ms;
            }
            std::cout << '\n';
        }
    }
    check(hipStreamDestroy(stream), "stream");
    // Host reference: router, shared expert (raw vectors of expert 0), routed window.
    const auto router = reference_router(x, rows, t.hidden, layer.router, layer.experts, layer.bias, layer.bias_image, image_rows,
                                         layer.top_k, layer.scale);
    std::vector<A8Activation> xs;
    for (std::uint32_t r = 0; r < rows; ++r) xs.push_back(quantize_a8(std::span<const float>(x.data() + static_cast<std::size_t>(r) * t.hidden, t.hidden)));
    const std::vector<std::uint32_t> zeros(rows, 0U);
    const std::vector<float> ones(rows, 1.0F), no_shared(static_cast<std::size_t>(rows) * t.hidden, 0.0F);
    const auto shared = reference_window(layer.shared->host, xs, build_window_routes(zeros, rows, 1U, 1U), ones, no_shared, 1U).raw;
    const auto routes = build_window_routes(router.ids, rows, layer.top_k, layer.split);
    const auto ref = reference_window(t, xs, routes, router.weights, shared, layer.top_k);
    const bool ok = single.ids == router.ids && mismatches(single.scores, router.scores) == 0U &&
        mismatches(single.weights, router.weights) == 0U && mismatches(single.y, ref.y) == 0U;
    expect(ok, label + ": layer router scores/ids/weights and y equal the host reference bitwise (scores " +
        std::to_string(mismatches(single.scores, router.scores)) + ", ids " + (single.ids == router.ids ? "same" : "differ") +
        ", weights " + std::to_string(mismatches(single.weights, router.weights)) + ", y " + std::to_string(mismatches(single.y, ref.y)) + ")");
    if (mismatches(single.scores, router.scores) != 0U)
        for (std::size_t i = 0; i < router.scores.size(); ++i)
            if (std::bit_cast<std::uint32_t>(single.scores[i]) != std::bit_cast<std::uint32_t>(router.scores[i])) {
                std::cerr << "  first score mismatch at " << i << ": device " << single.scores[i] << " host " << router.scores[i] << '\n';
                break;
            }
    std::cout << label << ": rows " << rows << ", union " << routes.experts.size() << ": router (scores, ids, weights), shared expert, routed "
              << "experts and combine " << (ok ? "bit-exact" : "MISMATCH") << '\n';

    // Two ranks: emulated (in-process transport) and TCP over loopback; each owns one half of the experts.
    for (const bool tcp : {false, true}) {
        std::unique_ptr<strix::dist::Transport> a, bt;
        std::uint16_t port = static_cast<std::uint16_t>(47000U + rows);
        if (!tcp) std::tie(a, bt) = strix::dist::memory_pair();
        LayerResult results[2];
        std::exception_ptr errors[2];
        std::thread ranks[2];
        for (int r = 0; r < 2; ++r)
            ranks[r] = std::thread([&, r] {
                try {
                    std::unique_ptr<strix::dist::Transport> mine;
                    if (tcp) mine = r == 0 ? strix::dist::tcp_listen(port, std::chrono::milliseconds(20000))
                                           : strix::dist::tcp_connect("127.0.0.1", port, std::chrono::milliseconds(20000));
                    else mine = std::move(r == 0 ? a : bt);
                    strix::dist::Communicator comm({static_cast<std::uint32_t>(r), 2U, 1U, 41U}, std::move(mine),
                                                   static_cast<std::size_t>(rows) * t.hidden * 4U);
                    hipStream_t s{};
                    check(hipStreamCreate(&s), "rank stream");
                    results[r] = run_layer(layer, x, rows, image_rows, s, r, &comm);
                    check(hipStreamDestroy(s), "rank stream");
                } catch (...) { errors[r] = std::current_exception(); }
            });
        for (auto& th : ranks) th.join();
        for (auto& e : errors) if (e) std::rethrow_exception(e);
        const bool same = mismatches(results[0].y, single.y) == 0U && mismatches(results[1].y, single.y) == 0U;
        expect(same, label + ": two ranks (" + (tcp ? "tcp" : "emulated") + ") give the single-node y bitwise");
        std::cout << label << ": two ranks over " << (tcp ? "tcp loopback" : "in-process transport") << ", each owning half the experts: y "
                  << (same ? "bit-exact" : "MISMATCH") << " on both ranks (exchange " << results[0].exchange_seconds * 1e6 << " us)\n";
    }
}

void test_real_layers(const std::string& q2, const std::string& dspark) {
    if (!q2.empty()) {
        const auto layer = load_layer(q2, "blk.20.", 384U, 6U);
        for (const std::uint32_t rows : {1U, 3U, 8U}) check_layer(*layer, rows, "real blk.20 MoE layer");
    }
    if (!dspark.empty()) {
        const auto layer = load_layer(dspark, "mtp.0.", 128U, 3U);
        for (const std::uint32_t rows : {1U, 6U}) check_layer(*layer, rows, "real mtp.0 MoE layer");
    }
}


// ---- Two nodes: one rank per process (STRIX_DS41_EP_RANK), USB4 streams or TCP between them ----

std::uint64_t fnv1a(const std::vector<float>& values) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const float v : values) {
        auto bits = std::bit_cast<std::uint32_t>(v);
        for (int k = 0; k < 4; ++k) { h ^= bits & 0xffU; h *= 1099511628211ULL; bits >>= 8U; }
    }
    return h;
}
double percentile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1U, static_cast<std::size_t>(q * static_cast<double>(v.size())))];
}

// Router alone, cold (read-only flush before each pass), the LDS-staged form against the previous
// one-wave-per-expert form, alternating; also checks that both give the same scores, ids and weights.
void bench_router(const std::string& q2) {
    using namespace strix::ds41;
    const auto layer = load_layer(q2, "blk.20.", 384U, 6U);
    const auto hidden = layer->routed->host.hidden;
    Flush flush;
    hipStream_t stream{};
    check(hipStreamCreate(&stream), "stream");
    hipEvent_t start{}, stop{};
    check(hipEventCreate(&start), "event");
    check(hipEventCreate(&stop), "event");
    // The model's router is BF16-valued: its BF16 copy (RouterView::weight_bf16) must give the same bits.
    const auto router32 = layer->router_device->download();
    std::vector<std::uint16_t> router16(router32.size());
    bool bf16_valued = true;
    for (std::size_t i = 0; i < router32.size(); ++i) {
        const auto u = std::bit_cast<std::uint32_t>(router32[i]);
        bf16_valued = bf16_valued && (u & 0xffffU) == 0U;
        router16[i] = static_cast<std::uint16_t>(u >> 16U);
    }
    expect(bf16_valued, "blk.20 router values are all BF16 values");
    Device<std::uint16_t> router16_device(router16.size());
    router16_device.upload(router16.data(), router16.size());
    for (const std::uint32_t rows : {1U, 2U, 3U, 4U, 6U, 8U}) {
        const auto x = activations(rows, hidden, 900U + rows);
        Device<float> xd(x.size()), scores(static_cast<std::size_t>(rows) * 384U), weights(rows * 6U);
        Device<std::uint32_t> ids(rows * 6U);
        Device<std::uint8_t> image(rows);
        xd.upload(x.data(), x.size());
        std::vector<std::uint8_t> image_rows(rows, 0U);
        image.upload(image_rows.data(), rows);
        RouterView v;
        v.x = xd.data; v.weight = layer->router_device->data; v.bias = layer->bias_device->data;
        v.bias_image = layer->bias_image_device->data; v.image_rows = image.data; v.rows = rows; v.hidden = hidden;
        v.experts = 384U; v.top_k = 6U; v.route_scale = layer->scale; v.scores = scores.data; v.ids = ids.data; v.weights = weights.data;
        std::vector<double> ms[3];
        std::vector<float> first_scores, first_weights;
        std::vector<std::uint32_t> first_ids;
        bool same = true;
        for (int pass = 0; pass < 21; ++pass) {  // staged, wave, row waves
            v.staged = pass % 3 == 0;
            v.row_waves = pass % 3 == 2;
            flush();
            check(hipDeviceSynchronize(), "flush");
            check(hipEventRecord(start, stream), "event");
            enqueue_router(v, stream);
            check(hipEventRecord(stop, stream), "event");
            check(hipStreamSynchronize(stream), "router");
            float t = 0.0F;
            check(hipEventElapsedTime(&t, start, stop), "event");
            ms[pass % 3].push_back(t);
            if (pass == 0) { first_scores = scores.download(); first_ids = ids.download(); first_weights = weights.download(); }
            else same = same && mismatches(scores.download(), first_scores) == 0U && ids.download() == first_ids &&
                        mismatches(weights.download(), first_weights) == 0U;
        }
        for (const int form : {0, 1, 2}) {  // BF16 copy, every form
            const bool staged = form == 0;
            v.staged = staged;
            v.row_waves = form == 2;
            v.weight_bf16 = router16_device.data;
            if (form == 2) {  // timing of the BF16 copy, row waves vs staged
                for (int rep = 0; rep < 2; ++rep) {
                    v.row_waves = rep == 1;
                    v.staged = true;
                    std::vector<double> tb;
                    for (int k = 0; k < 7; ++k) {
                        flush();
                        check(hipDeviceSynchronize(), "flush");
                        check(hipEventRecord(start, stream), "event");
                        enqueue_router(v, stream);
                        check(hipEventRecord(stop, stream), "event");
                        check(hipStreamSynchronize(stream), "router");
                        float tt = 0.0F;
                        check(hipEventElapsedTime(&tt, start, stop), "event");
                        tb.push_back(tt);
                    }
                    std::sort(tb.begin(), tb.end());
                    std::cout << "router rows " << rows << " BF16 " << (rep ? "row waves " : "staged ") << tb[3] * 1000.0 << " us\n";
                }
            }
            enqueue_router(v, stream);
            check(hipStreamSynchronize(stream), "router bf16");
            expect(mismatches(scores.download(), first_scores) == 0U && ids.download() == first_ids &&
                       mismatches(weights.download(), first_weights) == 0U,
                   "router BF16 copy gives the F32 bits, rows " + std::to_string(rows) + (form == 0 ? " staged" : form == 1 ? " wave" : " row waves"));
            v.weight_bf16 = nullptr;
            v.row_waves = false;
        }
        for (auto& m : ms) std::sort(m.begin(), m.end());
        expect(same, "router staged, wave and row-wave forms give the same bits");
        std::cout << "router rows " << rows << ": staged " << ms[0][ms[0].size() / 2U] * 1000.0 << " us, wave "
                  << ms[1][ms[1].size() / 2U] * 1000.0 << " us, row waves " << ms[2][ms[2].size() / 2U] * 1000.0
                  << " us (F32, median of 7 cold passes)" << (same ? ", same bits" : ", BITS DIFFER") << '\n';
    }
    check(hipEventDestroy(start), "event");
    check(hipEventDestroy(stop), "event");
    check(hipStreamDestroy(stream), "stream");
}

int run_ep_node(const std::string& q2, std::uint32_t rank, std::unique_ptr<strix::dist::Transport> transport, unsigned iterations) {
    const auto layer = load_layer(q2, "blk.20.", 384U, 6U);
    const auto hidden = layer->routed->host.hidden;
    strix::dist::Communicator comm({rank, 2U, 2U, 20U}, std::move(transport), 8U * hidden * 4U);
    std::cout << "rank " << rank << " over " << comm.transport_name() << '\n';
    hipStream_t stream{};
    check(hipStreamCreate(&stream), "stream");
    Device<std::uint64_t> local(1), remote(1);
    const auto barrier = [&] {
        check(hipMemsetAsync(local.data, 0, 8U, stream), "barrier");
        comm.exchange(stream, local.data, remote.data, 8U, strix::dist::Tag{4U, 0U, 0U});
    };
    barrier();
    // Clock offset to the peer (peer clock - own clock), from the shortest of 64 symmetric exchanges of
    // the entry time; the error is at most half of that exchange.
    const auto ns = [](std::chrono::steady_clock::time_point t) {
        return static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count());
    };
    std::int64_t offset = 0, best = INT64_MAX;
    Device<std::int64_t> stamps_local(8), stamps_remote(8);
    for (unsigned k = 0; k < 64U; ++k) {
        const auto entry = std::chrono::steady_clock::now();
        const std::int64_t mine = ns(entry);
        check(hipMemcpyAsync(stamps_local.data, &mine, 8U, hipMemcpyHostToDevice, stream), "stamp");
        comm.exchange(stream, stamps_local.data, stamps_remote.data, 8U, strix::dist::Tag{4U, 2U, k});
        const auto done = std::chrono::steady_clock::now();
        std::int64_t peer = 0;
        check(hipMemcpyAsync(&peer, stamps_remote.data, 8U, hipMemcpyDeviceToHost, stream), "stamp");
        check(hipStreamSynchronize(stream), "stamp");
        const auto span = ns(done) - ns(entry);
        if (span < best) { best = span; offset = peer - (ns(entry) + ns(done)) / 2; }
    }
    std::cout << "clock offset to peer " << offset / 1000 << " us (uncertainty " << best / 2000 << " us)\n";
    Flush flush;
    bool all_exact = true;
    for (const std::uint32_t rows : {1U, 6U}) {
        LayerRun single(*layer, rows, stream), ep(*layer, rows, stream);
        const std::vector<std::uint8_t> image_rows(rows, 0U);
        std::vector<double> single_ms, single_after_ms, ep_ms, exchange_ms, transport_ms, arrive_ms, stage_ms, wire_ms, check_ms;
        std::size_t exact = 0, agree = 0;
        for (unsigned it = 0; it < iterations + 3U; ++it) {
            const auto x = activations(rows, hidden, 1000U + it * 13U + rows);
            flush();
            check(hipStreamSynchronize(stream), "flush");
            auto start = std::chrono::steady_clock::now();
            single.enqueue(x, image_rows, -1, nullptr);
            check(hipStreamSynchronize(stream), "single layer");
            const double t_single = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            const auto y_single = single.result().y;
            flush();
            const auto barrier_start = std::chrono::steady_clock::now();
            barrier();
            const double t_barrier = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - barrier_start).count();
            const auto layer_stats = comm.stats();
            const double before = comm.stats().seconds;
            start = std::chrono::steady_clock::now();
            const auto start_point = start;
            ep.enqueue(x, image_rows, static_cast<int>(rank), &comm);
            check(hipStreamSynchronize(stream), "expert-parallel layer");
            const double t_ep = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            const auto done_point = std::chrono::steady_clock::now();
            const auto y_ep = ep.result().y;
            if (std::getenv("STRIX_DS41_EP_TIMELINE")) {
                // Rank 1 sends its five event times; rank 0 prints both on its own clock (us from its EP start).
                std::int64_t events[8] = {ns(barrier_start), ns(start_point), ns(comm.stats().last_entry), ns(comm.stats().last_staged),
                                          ns(comm.stats().last_received), ns(done_point), 0, 0};
                check(hipMemcpyAsync(stamps_local.data, events, sizeof(events), hipMemcpyHostToDevice, stream), "events");
                comm.exchange(stream, stamps_local.data, stamps_remote.data, sizeof(events), strix::dist::Tag{4U, 3U, it});
                std::int64_t peer[8]{};
                check(hipMemcpyAsync(peer, stamps_remote.data, sizeof(peer), hipMemcpyDeviceToHost, stream), "events");
                check(hipStreamSynchronize(stream), "events");
                if (rank == 0U) {
                    const auto rel = [&](std::int64_t t, bool remote_clock) { return static_cast<double>((remote_clock ? t - offset : t) - events[1]) / 1000.0; };
                    std::cout << "  timeline it " << it << " rows " << rows << " (us; barrier_start, ep_start, exchange_entry, staged, received, done)"
                              << " rank0: " << rel(events[0], false) << ' ' << rel(events[1], false) << ' ' << rel(events[2], false) << ' '
                              << rel(events[3], false) << ' ' << rel(events[4], false) << ' ' << rel(events[5], false)
                              << " | rank1: " << rel(peer[0], true) << ' ' << rel(peer[1], true) << ' ' << rel(peer[2], true) << ' '
                              << rel(peer[3], true) << ' ' << rel(peer[4], true) << ' ' << rel(peer[5], true) << '\n';
                }
            }
            if (std::getenv("STRIX_DS41_EP_VERBOSE"))
                std::cout << "  it " << it << " rows " << rows << " barrier_ms " << t_barrier << " ep_ms " << t_ep << " stage_ms "
                          << (comm.stats().stage - layer_stats.stage) * 1e3 << " wire_ms " << (comm.stats().wire - layer_stats.wire) * 1e3
                          << " overlap_ms " << (comm.stats().overlap - layer_stats.overlap) * 1e3 << " finish_wait_ms "
                          << (comm.stats().finish_wait - layer_stats.finish_wait) * 1e3 << '\n';
            const double t_exchange = (comm.stats().seconds - before) * 1e3;
            // The single-node layer again, after the expert-parallel one (same flush before it).
            flush();
            check(hipStreamSynchronize(stream), "flush");
            const auto again = std::chrono::steady_clock::now();
            single.enqueue(x, image_rows, -1, nullptr);
            check(hipStreamSynchronize(stream), "single layer");
            const double t_single_after = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - again).count();
            const double t_arrive = std::chrono::duration<double, std::milli>(comm.stats().last_entry - start_point).count();
            // Pure exchange of the same size right after a barrier: the transport cost of one layer.
            barrier();
            const auto pure = comm.stats();
            comm.exchange(stream, ep.own.data, ep.other.data, ep.own.count * 4U, strix::dist::Tag{1U, 21U, it});
            const double t_transport = (comm.stats().seconds - pure.seconds) * 1e3;
            const double t_stage = (comm.stats().stage - pure.stage) * 1e3, t_wire = (comm.stats().wire - pure.wire) * 1e3,
                         t_check = (comm.stats().check - pure.check) * 1e3;
            // Both ranks' outputs agree with each other too.
            const auto hash = fnv1a(y_ep);
            check(hipMemcpyAsync(local.data, &hash, 8U, hipMemcpyHostToDevice, stream), "hash");
            comm.exchange(stream, local.data, remote.data, 8U, strix::dist::Tag{4U, 1U, it});
            std::uint64_t peer = 0;
            check(hipMemcpyAsync(&peer, remote.data, 8U, hipMemcpyDeviceToHost, stream), "peer hash");
            check(hipStreamSynchronize(stream), "peer hash");
            if (it < 3U) continue;
            exact += mismatches(y_ep, y_single) == 0U ? 1U : 0U;
            agree += peer == hash ? 1U : 0U;
            single_ms.push_back(t_single); single_after_ms.push_back(t_single_after); ep_ms.push_back(t_ep); exchange_ms.push_back(t_exchange); transport_ms.push_back(t_transport);
            arrive_ms.push_back(t_arrive); stage_ms.push_back(t_stage); wire_ms.push_back(t_wire); check_ms.push_back(t_check);
        }
        all_exact = all_exact && exact == iterations && agree == iterations;
        // Back-to-back exchanges of one layer's half sums: the steady per-exchange cost on both ranks.
        barrier();
        const auto loop = comm.stats();
        constexpr unsigned repeats = 200;
        for (unsigned k = 0; k < repeats; ++k)
            comm.exchange(stream, ep.own.data, ep.other.data, ep.own.count * 4U, strix::dist::Tag{1U, 22U, k});
        check(hipStreamSynchronize(stream), "exchange loop");
        const auto after = comm.stats();
        std::cout << "{\"rank\":" << rank << ",\"rows\":" << rows << ",\"back_to_back_exchanges\":" << repeats
                  << ",\"exchange_us_mean\":" << (after.seconds - loop.seconds) * 1e6 / repeats << ",\"stage_us_mean\":"
                  << (after.stage - loop.stage) * 1e6 / repeats << ",\"wire_us_mean\":" << (after.wire - loop.wire) * 1e6 / repeats
                  << ",\"check_us_mean\":" << (after.check - loop.check) * 1e6 / repeats << "}\n";
        std::cout << "{\"rank\":" << rank << ",\"rows\":" << rows << ",\"iterations\":" << iterations << ",\"bit_exact_vs_single\":" << exact
                  << ",\"ranks_agree\":" << agree << ",\"single_ms_p50\":" << percentile(single_ms, 0.5) << ",\"single_after_ep_ms_p50\":" << percentile(single_after_ms, 0.5) << ",\"ep_ms_p50\":"
                  << percentile(ep_ms, 0.5) << ",\"exchange_in_layer_ms_p50\":" << percentile(exchange_ms, 0.5)
                  << ",\"exchange_alone_ms_p50\":" << percentile(transport_ms, 0.5) << ",\"exchange_alone_ms_p90\":"
                  << percentile(transport_ms, 0.9) << ",\"arrive_ms_p50\":" << percentile(arrive_ms, 0.5) << ",\"alone_stage_ms_p50\":"
                  << percentile(stage_ms, 0.5) << ",\"alone_wire_ms_p50\":" << percentile(wire_ms, 0.5) << ",\"alone_check_ms_p50\":"
                  << percentile(check_ms, 0.5) << ",\"bytes\":" << static_cast<std::size_t>(rows) * hidden * 4U << "}\n";
    }
    check(hipStreamDestroy(stream), "stream");
    std::cout << (all_exact ? "two-node expert parallelism: bit-exact\n" : "two-node expert parallelism: MISMATCH\n");
    return all_exact ? 0 : 1;
}

} // namespace

int main() try {
    if (const char* rank = std::getenv("STRIX_DS41_EP_RANK"); rank && *rank) {
        // Two-node mode: STRIX_DS41_EP_DEVICES=auto (every live USB4 stream, paired by handshake) or
        // STRIX_DS41_EP_DEVICES=/dev/tbstreamA,/dev/tbstreamB (USB4 host path, explicit list) or
        // STRIX_DS41_EP_TCP=host:port (rank 0 listens on port), STRIX_DS41_Q2_GGUF, STRIX_DS41_EP_ITERATIONS.
        const auto r = static_cast<std::uint32_t>(std::stoul(rank));
        std::unique_ptr<strix::dist::Transport> transport;
        if (const char* devices = std::getenv("STRIX_DS41_EP_DEVICES"); devices && std::string(devices) == "auto") {
            // Every live USB4 stream, paired by handshake.
            strix::dist::Usb4AutoOptions options;
            options.rank = r;
            strix::dist::Usb4Topology topology;
            transport = strix::dist::usb4_host_auto(options, &topology);
            std::cout << topology.receipt() << '\n';
        } else if (devices && *devices) {
            std::vector<std::string> list;
            std::string all = devices;
            for (std::size_t at = 0; at <= all.size();) {
                const auto comma = all.find(',', at);
                list.push_back(all.substr(at, comma == std::string::npos ? std::string::npos : comma - at));
                if (comma == std::string::npos) break;
                at = comma + 1U;
            }
            transport = strix::dist::usb4_host(list);
        } else {
            const std::string where = std::getenv("STRIX_DS41_EP_TCP") ? std::getenv("STRIX_DS41_EP_TCP") : "127.0.0.1:47600";
            const auto colon = where.rfind(':');
            const auto port = static_cast<std::uint16_t>(std::stoi(where.substr(colon + 1U)));
            transport = r == 0U ? strix::dist::tcp_listen(port, std::chrono::milliseconds(600000))
                                : strix::dist::tcp_connect(where.substr(0, colon), port, std::chrono::milliseconds(600000));
        }
        const char* iterations = std::getenv("STRIX_DS41_EP_ITERATIONS");
        return run_ep_node(std::getenv("STRIX_DS41_Q2_GGUF"), r, std::move(transport), iterations ? static_cast<unsigned>(std::stoul(iterations)) : 20U);
    }
    test_scalar_tables();
    test_declared_functions();
    test_synthetic();
    const char* q2 = std::getenv("STRIX_DS41_Q2_GGUF");
    const char* dspark = std::getenv("STRIX_DS41_DSPARK_GGUF");
    if (const char* only = std::getenv("STRIX_DS41_ROUTER_BENCH"); only && *only == '1' && q2 && *q2) {
        bench_router(q2);
        return failures == 0 ? 0 : 1;
    }
    if (q2 && *q2) {
        std::vector<std::string> names;
        for (const auto layer : {0, 1, 20, 39})
            for (const char* kind : {"gate", "up", "down"})
                names.push_back("blk." + std::to_string(layer) + ".ffn_" + kind + "_exps.weight");
        test_real_experts(q2, names, {0U, 191U, 383U});
        test_real_experts(q2, {"blk.20.ffn_gate_shexp.weight", "blk.20.ffn_up_shexp.weight", "blk.20.ffn_down_shexp.weight"}, {0U});
        test_real_engram(q2);
    }
    if (dspark && *dspark) {
        std::vector<std::string> names;
        for (const auto stage : {0, 1, 2})
            for (const char* kind : {"gate", "up", "down"})
                names.push_back("mtp." + std::to_string(stage) + ".ffn_" + kind + "_exps.weight");
        test_real_experts(dspark, names, {0U, 64U, 127U});
    }
    test_synthetic_windows();
    test_device_routes();
    test_real_windows(q2 ? q2 : "", dspark ? dspark : "");
    test_real_layers(q2 ? q2 : "", dspark ? dspark : "");
    if (const char* bench = std::getenv("STRIX_DS41_BENCH"); bench && *bench == '1') {
        std::vector<std::uint8_t> storage[3];
        if (q2 && *q2) {
            const DeviceExperts e(file_experts(q2, "blk.20.", storage));
            bench_windows(e, 6U, 384U, 192U, "blk.20");
        }
        if (dspark && *dspark) {
            const DeviceExperts e(file_experts(dspark, "mtp.0.", storage));
            bench_windows(e, 3U, 128U, 64U, "mtp.0");
        }
    }
    if (failures == 0) std::cout << "deepseek41 device format tests passed\n";
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "deepseek41 device format tests: " << error.what() << '\n';
    return 2;
}
