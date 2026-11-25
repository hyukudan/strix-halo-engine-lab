// strix-glm5-vision: GLM-5.3 Flash image input on the host (no GPU). The image tower and its projector run on the
// shared FP32 reference encoder; the result is an embed rows file that strix-glm5-rank splices into the prompt
// (--embed-rows), the same bytes on both ranks.
//
//   strix-glm5-vision describe MMPROJ.gguf
//   strix-glm5-vision encode MMPROJ.gguf IMAGE OUT.rows [--f32 FILE] [--patches FILE] [--reference-f16-conv]
//       IMAGE: PNG or JPEG. --f32: the rows before their BF16 rounding, [rows][width] FP32. --patches: the tower
//       input, [patches][3 * 2 * patch * patch] FP32. --reference-f16-conv: diagnostic arm, the FP16 convolution
//       inputs of the file format's reference implementation (other rows, another identity).
//   strix-glm5-vision expand IN.u32 OUT.u32 A.rows[,B.rows,...]
//       The n-th <|image|> token of the prompt becomes <|begin_of_image|>, one <|image|> per row of the n-th file,
//       <|end_of_image|> (a marker already next to the token is kept, not doubled).
#include "glm/vision/glm_vision.hpp"

#include "strix/model_content_hash.hpp"
#include "strix/session/embed_rows_file.hpp"
#include "strix/vision/embed_span.hpp"
#include "strix/vision/encoder.hpp"
#include "strix/vision/gguf_weights.hpp"
#include "strix/vision/image.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace vis = strix::vision;
namespace glm = strix::glm;

std::vector<std::byte> read_bytes(const std::string& path, std::uint64_t max_bytes) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + path);
    const auto size = static_cast<std::uint64_t>(in.tellg());
    if (size > max_bytes) throw std::invalid_argument(path + " is larger than " + std::to_string(max_bytes) + " bytes");
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in) throw std::runtime_error("short read of " + path);
    return bytes;
}

template <class T>
void write_values(const std::string& path, const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
    out.flush();
    if (!out) throw std::runtime_error("cannot write " + path);
}

struct Tower {
    vis::GgufWeights source;
    glm::VisionConfig model;
    std::string sha;
    explicit Tower(const std::string& path) : source(std::filesystem::path(path), glm::vision_gguf_limits()) {
        model = glm::validate_vision(source.info());
        sha = strix::model_content_identity_hex(strix::sha256_bytes_fast(std::span<const std::byte>(source.bytes())));
    }
};

int describe(const std::string& path) {
    const Tower t(path);
    const auto& c = t.model;
    std::printf("glm5 vision: %u blocks, width %u, %u heads, mlp %u, patch %u, merge %u, %s qkv\n", c.layers, c.width, c.heads,
                c.mlp_width, c.patch, c.merge, c.fused_qkv ? "fused" : "separate");
    std::printf("projector: merge %u -> %u -> %u, mlp %u, swiglu clamp %g, norm eps %g\n", c.width * c.merge * c.merge,
                c.merged_width, c.projection, c.projector_mlp, static_cast<double>(c.swiglu_limit), static_cast<double>(c.eps));
    std::printf("image: %llu to %llu pixels, at most %u rows\n", static_cast<unsigned long long>(c.min_pixels),
                static_cast<unsigned long long>(c.max_pixels), c.max_image_tokens());
    std::printf("sha256 %s\n", t.sha.c_str());
    return 0;
}

int encode(int argc, char** argv) {
    if (argc < 5) throw std::invalid_argument("encode MMPROJ.gguf IMAGE OUT.rows [--f32 FILE] [--patches FILE] [--reference-f16-conv]");
    std::string f32_out, patches_out;
    glm::VisionRecipe recipe;
    for (int i = 5; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--f32" && i + 1 < argc) f32_out = argv[++i];
        else if (a == "--patches" && i + 1 < argc) patches_out = argv[++i];
        else if (a == "--reference-f16-conv") recipe.conv_inputs_f16 = true;
        else throw std::invalid_argument("unknown option " + a);
    }
    using clock = std::chrono::steady_clock;
    const auto seconds = [](clock::time_point a, clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    const auto t0 = clock::now();
    Tower tower(argv[2]);
    glm::require_vision_matches_language(tower.model, glm::kLanguageHidden, glm::kLanguageVocab, glm::kImageToken);
    glm::VisionAdapter adapter(tower.model, tower.sha, glm::kImageToken, recipe);
    vis::EncoderWeights weights;
    adapter.bind(tower.source, weights);
    vis::ReferenceEncoder backend(adapter.config(), weights);
    const auto t1 = clock::now();

    const vis::DecodeLimits limits;
    const std::vector<std::byte> file = read_bytes(argv[3], limits.max_bytes);
    const vis::Image image = vis::decode(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(file.data()), file.size()), limits);
    const vis::PatchGrid grid = adapter.patches(image);
    if (!patches_out.empty()) write_values(patches_out, grid.data);
    const vis::SpanLayout layout = adapter.layout(grid);
    std::fprintf(stderr, "image %dx%d -> %ux%u patches, %u rows; tower on the host reference (FP32), this takes a while\n", image.width,
                 image.height, grid.grid_w, grid.grid_h, layout.rows());
    const auto t2 = clock::now();
    // The steps of vision::encode_cached, with the FP32 rows kept for --f32.
    const vis::EncoderConfig config = adapter.config();
    const vis::EncoderOutput hidden = backend.encode(grid, adapter.positions(config, weights, grid));
    const std::vector<float> wide = adapter.project_f32(hidden, grid, backend);
    vis::ProjectedRows projected;
    projected.width = tower.model.projection;
    projected.rows = layout.rows();
    projected.bf16 = vis::to_bf16_rows(wide.data(), wide.size());
    const auto t3 = clock::now();

    // The span digest names the producer and the content: adapter identity, arithmetic, image file bytes.
    const std::string producer = adapter.identity() + "|" + backend.arithmetic();
    strix::Sha256Fast sha;
    sha.update(std::as_bytes(std::span<const char>(producer.data(), producer.size() + 1U)));
    sha.update(std::span<const std::byte>(file));
    const strix::session::EmbedRows rows = vis::span_embed_rows(layout, projected, sha.finalize().bytes);
    strix::session::write_embed_rows_file(argv[4], rows);
    if (!f32_out.empty()) write_values(f32_out, wide);
    std::printf("rows %u width %u grid %ux%u token %u -> %s (load %.2f s, preprocess %.2f s, encode %.2f s)\n", rows.rows, rows.width,
                layout.grid_w, layout.grid_h, rows.real_token_id, argv[4], seconds(t0, t1), seconds(t1, t2), seconds(t2, t3));
    std::printf("identity %s\n", producer.c_str());
    return 0;
}

int expand(int argc, char** argv) {
    if (argc != 5) throw std::invalid_argument("expand IN.u32 OUT.u32 A.rows[,B.rows,...]");
    // A prompt of at most 16 M ids in and out, and at most 1024 rows files.
    constexpr std::size_t kMaxTokens = std::size_t{1} << 24U;
    const std::vector<std::byte> raw = read_bytes(argv[2], std::uint64_t{kMaxTokens} * 4U);
    if (raw.size() % 4U != 0) throw std::invalid_argument("the prompt is not a list of 32-bit ids");
    std::vector<std::uint32_t> in(raw.size() / 4U);
    std::memcpy(in.data(), raw.data(), raw.size());
    std::vector<std::uint32_t> rows;
    const std::string list = argv[4];
    for (std::size_t a = 0; a <= list.size();) {
        const std::size_t b = std::min(list.find(',', a), list.size());
        if (rows.size() == 1024U) throw std::invalid_argument("more than 1024 rows files");
        const auto span = strix::session::read_embed_rows_file(list.substr(a, b - a), glm::kLanguageHidden);
        if (span.real_token_id != glm::kImageToken) throw std::invalid_argument("a rows file was made for another image token");
        rows.push_back(span.rows);
        a = b + 1U;
    }
    std::vector<std::uint32_t> out;
    std::size_t image = 0;
    for (std::size_t i = 0; i < in.size(); ++i) {
        if (in[i] != glm::kImageToken) {
            if (out.size() == kMaxTokens) throw std::invalid_argument("the expanded prompt exceeds 16 M ids");
            out.push_back(in[i]);
            continue;
        }
        if (i + 1U < in.size() && in[i + 1U] == glm::kImageToken)
            throw std::invalid_argument("the prompt holds consecutive image tokens: it is already expanded");
        if (image == rows.size()) throw std::invalid_argument("the prompt holds more image tokens than rows files were given");
        if (out.size() + rows[image] + 2U > kMaxTokens) throw std::invalid_argument("the expanded prompt exceeds 16 M ids");
        if (out.empty() || out.back() != glm::kImageBeginToken) out.push_back(glm::kImageBeginToken);
        out.insert(out.end(), rows[image++], glm::kImageToken);
        if (i + 1U == in.size() || in[i + 1U] != glm::kImageEndToken) out.push_back(glm::kImageEndToken);
    }
    if (image != rows.size()) throw std::invalid_argument("the prompt holds fewer image tokens than rows files were given");
    write_values(argv[3], out);
    std::printf("prompt %zu -> %zu tokens, %zu image(s)\n", in.size(), out.size(), image);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "describe" && argc == 3) return describe(argv[2]);
        if (mode == "encode") return encode(argc, argv);
        if (mode == "expand") return expand(argc, argv);
        std::fprintf(stderr, "usage: %s describe MMPROJ.gguf\n       %s encode MMPROJ.gguf IMAGE OUT.rows [--f32 FILE] [--patches FILE] [--reference-f16-conv]\n"
                             "       %s expand IN.u32 OUT.u32 A.rows[,B.rows,...]\n", argv[0], argv[0], argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-glm5-vision: %s\n", e.what());
        return 1;
    }
}
