// Host-only properties of weight::with_native_source (src/cuda/weight/gsq_sources.cuh), the binding of a loaded
// expert storage type to the native expert operand source. No device code runs and no GPU is needed; the bitwise
// device identity of the typed decoder is covered by the native wfmt capture (tests/cuda/qwen_moe_wfmt_capture.inc).
//  1. the typed set is exactly the six GSQ storage types, and each binds the source typed for that same type;
//  2. every other type binds the runtime-type source (the fallback), so no type the shared decoder covers is lost;
//  3. a typed source is only ever bound to a type the shared decoder covers (geometry != {0,0});
//  4. typed and runtime sources agree on row_bytes for every covered type and the widths the native route meets;
//  5. the DeepSeek-4.1 expert storage types (Q2_K 10, IQ2_XXS 16) have no shared decoder, so the native route's
//     geometry guard (qwen_moe.cu enqueue_routed) rejects them instead of decoding the runtime switch's zero default.
#include "cuda/weight/gsq_sources.cuh"

#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace w = strix::cuda::weight;
namespace q = strix::cuda::quant;

namespace {

struct TestArch {};
using Runtime = w::OperandSource<w::RuntimeGgmlTag, w::NativeScalarTag, TestArch, w::CudaNativeFp32>;

int failures = 0;
void expect(bool ok, const char* what, std::uint32_t type) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL type %u: %s\n", type, what); }
}

struct Bound { bool typed = false, runtime = false; std::uint32_t storage = ~0U; std::uint64_t bytes[4] = {}; };
constexpr unsigned kWidths[4] = {256, 640, 2560, 5120};   // native route: hidden 2560/5120 and ffn 640, plus one block

Bound bind(std::uint32_t type) {
    Bound b;
    w::with_native_source<TestArch>(type, [&]<class Source>() {
        if constexpr (requires { Source::storage_type; }) { b.typed = true; b.storage = Source::storage_type; }
        b.runtime = std::is_same_v<Source, Runtime>;
        if (q::geometry(type).block_elements != 0)
            for (int i = 0; i < 4; ++i) b.bytes[i] = Source::row_bytes(type, kWidths[i]);
    });
    return b;
}

bool in_typed_set(std::uint32_t t) {
    return t == q::t_iq3_xxs || t == q::t_iq4_nl || t == q::t_iq3_s || t == q::t_iq2_s || t == q::t_iq4_xs || t == q::t_q2_0;
}

} // namespace

int main() {
    unsigned typed = 0, covered = 0;
    for (std::uint32_t t = 0; t < 64; ++t) {
        const Bound b = bind(t);
        const bool decodable = q::geometry(t).block_elements != 0;
        covered += decodable;
        expect(b.typed != b.runtime, "exactly one of typed / runtime source", t);
        expect(b.typed == in_typed_set(t), "typed set is exactly the six GSQ storage types", t);
        if (b.typed) {
            ++typed;
            expect(b.storage == t, "typed source fixed to the dispatched type", t);
            expect(decodable, "typed source bound only to a covered type", t);
        }
        if (decodable && b.typed)
            for (int i = 0; i < 4; ++i)
                if (kWidths[i] % q::geometry(t).block_elements == 0)
                    expect(b.bytes[i] == Runtime::row_bytes(t, kWidths[i]), "typed row_bytes == runtime row_bytes", t);
    }
    expect(typed == 6, "six typed bindings", typed);
    expect(covered == 16, "shared decoder covers 16 storage types (10 by the runtime fallback)", covered);
    expect(q::geometry(10).block_elements == 0, "Q2_K has no shared decoder (DeepSeek-4.1 down)", 10);
    expect(q::geometry(16).block_elements == 0, "IQ2_XXS has no shared decoder (DeepSeek-4.1 gate/up)", 16);
    expect(bind(10).runtime && bind(16).runtime, "uncovered types reach only the guarded runtime source", 16);
    if (failures) return 1;
    std::printf("native_source_binding: %u typed, %u covered by the runtime fallback, 2 DeepSeek-4.1 types guarded\n",
                typed, covered - typed);
    return 0;
}
