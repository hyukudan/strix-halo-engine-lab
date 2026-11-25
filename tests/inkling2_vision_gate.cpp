// Inkling 2 vision gate on the real checkpoint against the dumps of tools/vision/inkling2_vision_ref.py (transformers
// 5.24 Inkling2VisionModel run in FP32 on the checkpoint's own vision tensors).
//
//   strix-inkling2-vision-gate[-hip] --checkpoint DIR --ref DIR [--backend host|hip] [--max-rel-l2 X] NAME...
//
// Per NAME (DIR/NAME.png with DIR/NAME.{grid.json,pixels.f32,hidden.f32,rows.f32}):
//   patches   decode + adapter.patches() against the reference pixel_values (exact)
//   encoder   adapter positions -> tower -> hidden rows (class row first) against the reference hidden states
//   rows      adapter.project() -> image rows against the reference merger output (relative L2, worst row cosine)
// Exit status 0 when patches match and both hidden and projected rows stay within --max-rel-l2 (default 0.02).
#include "inkling2/inkling2_vision.hpp"
#include "strix/serve/json.hpp"
#include "inkling2_vision_gate_checks.hpp"
#include "strix/vision/encoder.hpp"
#include "strix/vision/image.hpp"
#include "strix/vision/lookup_weights.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace strix;

namespace gate = inkling2::gate;

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) try {
    std::string checkpoint, ref, backend_name = "host";
    double max_rel_l2 = 0.02;
    std::vector<std::string> names;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return argv[++i];
        };
        if (a == "--checkpoint") checkpoint = value();
        else if (a == "--ref") ref = value();
        else if (a == "--backend") backend_name = value();
        else if (a == "--max-rel-l2") {
            const auto arg = value();
            std::size_t consumed = 0;
            max_rel_l2 = std::stod(arg, &consumed);
            if (consumed != arg.size() || !std::isfinite(max_rel_l2) || max_rel_l2 < 0.0)
                throw std::invalid_argument("--max-rel-l2 requires a finite nonnegative number");
        }
        else names.push_back(a);
    }
    if (checkpoint.empty() || ref.empty() || names.empty()) {
        std::fprintf(stderr, "usage: %s --checkpoint DIR --ref DIR [--backend host|hip] [--max-rel-l2 X] NAME...\n", argv[0]);
        return 2;
    }
    for (const auto& name : names)
        if (!gate::simple_name(name)) throw std::invalid_argument("NAME must be a simple case name: " + name);
    const weight::ConfinedDirectory dir(checkpoint), refdir(ref);
    const auto spec = inkling2::parse_vision_spec(*dir.read("config.json", 1U << 20U));
    const auto specs = inkling2::vision_tensor_specs(spec);
    weight::TensorLookup lookup;
    const weight::TensorSchema schema{weight::Container::safetensors, "safetensors@1", specs, inkling2::kVisionIgnoredPrefix};
    auto t0 = std::chrono::steady_clock::now();
    if (const auto st = lookup.open_safetensors(dir.open("model-00001-of-00001.safetensors"), schema); !st)
        throw std::runtime_error("vision tensors do not match the schema");
    vision::LookupWeights source(lookup);
    inkling2::VisionAdapter adapter(spec, "gate", inkling2::VisionIdentity::axial_v1);
    vision::EncoderWeights weights;
    adapter.bind(source, weights);
    const auto cfg = adapter.config();
    std::printf("bind %.0f ms\n", ms_since(t0));
    std::unique_ptr<vision::EncoderBackend> backend;
#if defined(STRIX_INKLING2_VISION_GATE_HIP)
    if (backend_name == "hip") {
        t0 = std::chrono::steady_clock::now();
        auto enc = std::make_unique<vision::Encoder>(cfg, adapter.device_options());
        enc->load(weights);
        std::printf("hip load %.0f ms (%s)\n", ms_since(t0), enc->arithmetic());
        backend = std::move(enc);
    }
#endif
    if (!backend) {
        if (backend_name != "host") throw std::invalid_argument("backend " + backend_name + " is not built in this tool");
        backend = std::make_unique<vision::ReferenceEncoder>(cfg, weights);
    }
    bool ok = true;
    for (const auto& name : names) {
        const auto grid = serve::parse_json_unique(*refdir.read(name + ".grid.json", 64U << 10U), 64U << 10U, 256U);
        const auto shape = gate::shape(grid, spec);
        const auto limits = vision::DecodeLimits{};
        const auto bytes = refdir.read(name + ".png", limits.max_bytes);
        const auto image = vision::decode({reinterpret_cast<const std::uint8_t*>(bytes->data()), bytes->size()},
                                          limits, adapter.color_policy());
        const auto cell = spec.patch * spec.merge;
        const unsigned image_h = static_cast<unsigned>(image.height), image_w = static_cast<unsigned>(image.width);
        if ((image_h + cell - 1U) / cell != shape.rows || (image_w + cell - 1U) / cell != shape.cols)
            throw std::runtime_error("decoded raster differs from reference grid: " + name);
        const auto g = adapter.patches(image);
        const auto tg = adapter.token_grid(g);
        shape.check_patches(g, tg.rows, tg.cols);
        const auto pixels = gate::read_f32(refdir, name + ".pixels.f32", shape.pixels());
        const auto reference_hidden = gate::read_f32(refdir, name + ".hidden.f32", shape.hidden());
        const auto reference_rows = gate::read_f32(refdir, name + ".rows.f32", shape.projected());

        // Reference pixels are interleaved by scanline within each padded cell.
        double patch_max = 0.0;
        for (unsigned t = 0; t < shape.tokens(); ++t)
            for (unsigned dy = 0; dy < spec.merge; ++dy)
                for (unsigned dx = 0; dx < spec.merge; ++dx) {
                    const auto patch = 1U + (t * spec.merge + dy) * spec.merge + dx;
                    const float* row = g.data.data() + std::size_t{patch} * shape.patch_dim;
                    for (unsigned py = 0; py < spec.patch; ++py)
                        for (unsigned px = 0; px < spec.patch; ++px)
                            for (unsigned c = 0; c < 3U; ++c) {
                                const auto y = std::size_t{t} * cell + dy * spec.patch + py;
                                const auto x = dx * spec.patch + px;
                                const float want = pixels[(y * cell + x) * 3U + c];
                                const float got = row[(py * spec.patch + px) * 3U + c];
                                if (!std::isfinite(got)) throw std::runtime_error("non-finite patch: " + name);
                                patch_max = std::max(patch_max, std::fabs(double(got) - want));
                            }
                }
        t0 = std::chrono::steady_clock::now();
        const auto pos = adapter.positions(cfg, weights, g);
        auto out = backend->encode(g, pos);
        const double encode_ms = ms_since(t0);
        shape.check_hidden(out);
        const auto hid = gate::compare_rows(out.hidden, reference_hidden, shape.hidden_width);
        t0 = std::chrono::steady_clock::now();
        vision::ProjectedRows rows;
        adapter.project(out, g, *backend, rows);
        const double project_ms = ms_since(t0);
        shape.check_projected(rows);
        std::vector<float> got(shape.projected());
        for (std::size_t i = 0; i < got.size(); ++i) got[i] = vision::from_bf16(rows.bf16[i]);
        const auto rs = gate::compare_rows(got, reference_rows, shape.projected_width);
        const bool pass = gate::passes(patch_max, hid, rs, max_rel_l2);
        ok = ok && pass;
        std::printf("%-22s %ux%u tokens %u | patches max|d| %.3g | hidden relL2 %.3e worst cos %.6f | rows relL2 %.3e "
                    "worst cos %.6f max|d| %.3g | encode %.1f ms project %.1f ms | %s\n",
                    name.c_str(), tg.rows, tg.cols, rows.rows, patch_max, hid.rel_l2, hid.worst_cos,
                    rs.rel_l2, rs.worst_cos, rs.max_abs, encode_ms, project_ms, pass ? "PASS" : "FAIL");
    }
    return ok ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "inkling2 vision gate: %s\n", e.what());
    return 2;
}
