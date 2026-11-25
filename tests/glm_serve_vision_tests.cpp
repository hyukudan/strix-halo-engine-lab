// Host test of the image input of strix-glm5-serve (src/glm/serve/glm_serve_vision.hpp): the projector file through
// the shared media bridge to interned rows, the pair embed frame from the lead to the follower, and both ranks'
// resolution of prompt chunks into vocabulary ids and the engine's splice. Synthetic tower, no device, no engine.
#include "../src/glm/serve/glm_serve_vision.hpp"

#include "glm_vision_toy.hpp"
#include "private_test_dir.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace strix;
namespace gs = strix::glm::serve;

int failures = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            ++failures;                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                               \
    } while (false)

bool throws(const std::function<void()>& fn, const char* needle) {
    try {
        fn();
    } catch (const std::exception& e) {
        if (std::strstr(e.what(), needle) != nullptr) return true;
        std::printf("  unexpected message: %s\n", e.what());
    }
    return false;
}

// Any checksum works for the frame as long as both ends use the same one.
std::uint32_t test_crc(std::span<const std::byte> bytes) noexcept {
    std::uint32_t h = 2166136261U;
    for (const std::byte b : bytes) h = (h ^ static_cast<std::uint32_t>(b)) * 16777619U;
    return h;
}

// The tokenizer of the stack: only the three image tokens matter.
std::vector<std::uint32_t> encode(const std::string& text) {
    if (text == gs::kImageText) return {glm::kImageToken};
    if (text == gs::kImageBeginText) return {glm::kImageBeginToken};
    if (text == gs::kImageEndText) return {glm::kImageEndToken};
    return {1, 2, 3};
}

// A new file in the run's private directory (never an existing path).
std::filesystem::path write_file(const strix::test::PrivateTestDir& dir, const glm_vision_toy::File& f, const char* name) {
    const auto path = dir.fresh(name);
    const std::string bytes = f.bytes();
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("cannot write " + path.string());
    return path;
}

serve::media::MediaItem item_of(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    serve::media::MediaItem m;
    m.bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (m.bytes.empty()) throw std::runtime_error("cannot read " + path);
    m.digest = serve::media::sha256(m.bytes);
    m.param = "messages[0].content[1]";
    return m;
}

// What an engine's embed stage would hold for a chunk: per row the first element of its source row, or -1 for a
// table row.
std::vector<std::int64_t> sources(const session::ResolvedEmbedRun& run) {
    std::vector<std::int64_t> out(run.tokens.size(), -1);
    for (const auto& seg : run.splice.segments)
        for (std::uint32_t j = 0; j < seg.rows; ++j) out[seg.run_row + j] = seg.data[std::size_t{j} * run.splice.width];
    return out;
}
bool same_rows(const session::ResolvedEmbedRun& a, const session::ResolvedEmbedRun& b) {
    if (a.tokens != b.tokens || a.splice.width != b.splice.width || a.splice.segments.size() != b.splice.segments.size()) return false;
    for (std::size_t i = 0; i < a.splice.segments.size(); ++i) {
        const auto& x = a.splice.segments[i];
        const auto& y = b.splice.segments[i];
        if (x.run_row != y.run_row || x.rows != y.rows ||
            std::memcmp(x.data, y.data, std::size_t{x.rows} * a.splice.width * 2U) != 0)
            return false;
    }
    return true;
}

void test_stack_and_frames(const std::string& fixtures) {
    const strix::test::PrivateTestDir dir("strix-glm-serve-vision");
    const auto path = write_file(dir, glm_vision_toy::small_file(false, true, glm::kLanguageHidden), "vision.gguf");
    serve::media::Limits limits;
    auto stack = gs::make_vision_stack(path, encode, limits);
    CHECK(stack->bridge->placeholder() == "<|image|>");
    CHECK(stack->bridge->part_text(0) == "<|begin_of_image|><|image|><|end_of_image|>");
    CHECK(stack->bridge->max_rows() == 36 && stack->bridge->identity().find("glm5-vision/") == 0);
    CHECK(stack->bridge->identity().find("vision-ref-f32") != std::string::npos);

    // Two images through the bridge: rows of the language width, every row an image row.
    const auto a = stack->bridge->encode(item_of(fixtures + "/exif1_noise.png"), 36);
    const auto b = stack->bridge->encode(item_of(fixtures + "/exif6_noise.png"), 36);
    const std::uint32_t ra = a.lease.rows().rows, rb = b.lease.rows().rows;
    CHECK(ra >= 4 && ra <= 36 && rb >= 4 && a.layout.size() == ra && a.lease.rows().width == glm::kLanguageHidden);
    CHECK(a.lease.rows().real_token_id == glm::kImageToken && a.lease.handle() != b.lease.handle());
    const auto again = stack->bridge->encode(item_of(fixtures + "/exif1_noise.png"), 36);
    CHECK(again.cache_hit && again.lease.handle() == a.lease.handle());
    CHECK(throws([&] { (void)stack->bridge->encode(item_of(fixtures + "/exif1_noise.png"), ra - 1U); }, "rows"));

    // The prompt as the API layer expands it: text, <begin>, the rows of image a, <end>, text, <begin>, b, <end>, text.
    std::vector<std::uint32_t> prompt{11, 12, glm::kImageBeginToken};
    prompt.insert(prompt.end(), a.layout.begin(), a.layout.end());
    prompt.insert(prompt.end(), {glm::kImageEndToken, 13, glm::kImageBeginToken});
    prompt.insert(prompt.end(), b.layout.begin(), b.layout.end());
    prompt.insert(prompt.end(), {glm::kImageEndToken, 14, 15});
    const auto table = std::make_shared<const session::EmbedTable>(std::vector<session::EmbedLease>{a.lease, b.lease});

    gs::PromptEmbeds lead(&test_crc), follower(&test_crc);
    lead.bind(table);
    // Chunks as the scheduler cuts them: inside the first image, across the gap, to the end; then the whole prompt.
    const std::size_t cuts[] = {0, 3U + ra / 2U, 3U + ra + 4U, prompt.size()};
    std::vector<std::size_t> frame_bytes;
    std::vector<std::int64_t> lead_all, follower_all;
    for (std::size_t i = 0; i + 1 < std::size(cuts); ++i) {
        const std::span<const std::uint32_t> chunk(prompt.data() + cuts[i], cuts[i + 1] - cuts[i]);
        CHECK(gs::PromptEmbeds::holds_virtual_ids(chunk));
        const auto frame = lead.frame(chunk);
        frame_bytes.push_back(frame.size());
        std::vector<std::uint32_t> remote(chunk.begin(), chunk.end());
        follower.received(frame, remote);
        const auto l = lead.resolve(chunk), f = follower.resolve(remote);
        CHECK(same_rows(l, f));
        CHECK(l.splice.width == glm::kLanguageHidden);
        for (const std::uint32_t t : l.tokens) CHECK(t < glm::kLanguageVocab);
        const auto ls = sources(l), fs = sources(f);
        lead_all.insert(lead_all.end(), ls.begin(), ls.end());
        follower_all.insert(follower_all.end(), fs.begin(), fs.end());
    }
    CHECK(lead_all == follower_all && lead_all.size() == prompt.size());
    // Every image row is provided, in span order; every other row reads the table.
    std::size_t provided = 0;
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        if (!session::is_embed_id(prompt[i])) {
            CHECK(lead_all[i] == -1);
            continue;
        }
        const auto* span = table->find(session::embed_handle(prompt[i]));
        CHECK(span != nullptr && lead_all[i] == span->row(session::embed_row(prompt[i]))[0]);
        ++provided;
    }
    CHECK(provided == ra + rb);
    // The rows of an image travel once: the chunk that continues image a carries a manifest only.
    CHECK(frame_bytes[0] > std::size_t{ra} * glm::kLanguageHidden * 2U);
    CHECK(frame_bytes[1] > std::size_t{rb} * glm::kLanguageHidden * 2U && frame_bytes[1] < frame_bytes[0] + std::size_t{rb} * glm::kLanguageHidden * 2U);
    const auto whole = lead.frame(prompt);
    CHECK(whole.size() < 1024);
    std::vector<std::uint32_t> remote(prompt);
    follower.received(whole, remote);
    CHECK(same_rows(lead.resolve(prompt), follower.resolve(remote)));

    // A text chunk resolves to itself with no splice and needs no table.
    const std::vector<std::uint32_t> text{5, 6, 7};
    CHECK(!gs::PromptEmbeds::holds_virtual_ids(text));
    const auto plain = gs::PromptEmbeds(&test_crc).resolve(text);
    CHECK(plain.tokens == text && plain.splice.empty());
    // Refusals: a frame before a table, virtual ids without a table, a damaged frame.
    CHECK(throws([&] { (void)gs::PromptEmbeds(&test_crc).frame(prompt); }, "no table"));
    CHECK(throws([&] { (void)gs::PromptEmbeds(&test_crc).resolve(prompt); }, "without a bound table"));
    gs::PromptEmbeds lead2(&test_crc), follower2(&test_crc);
    lead2.bind(table);
    auto damaged = lead2.frame(prompt);
    damaged[damaged.size() / 2U] ^= std::byte{1};
    std::vector<std::uint32_t> ids(prompt);
    CHECK(throws([&] { follower2.received(damaged, ids); }, "embed frame"));

    // A tokenizer without the model's image tokens, or a file of another language width, is refused at start.
    CHECK(throws([&] { (void)gs::make_vision_stack(path, [](const std::string&) { return std::vector<std::uint32_t>{1}; }, limits); },
                 "image tokens"));
    const auto narrow = write_file(dir, glm_vision_toy::small_file(false, true), "narrow.gguf");
    CHECK(throws([&] { (void)gs::make_vision_stack(narrow, encode, limits); }, "projection width"));
}

}  // namespace

int main(int argc, char** argv) {
#ifdef STRIX_VISION_FIXTURE_DIR
    const std::string fixtures = argc >= 2 ? argv[1] : STRIX_VISION_FIXTURE_DIR;
#else
    if (argc < 2) {
        std::fprintf(stderr, "usage: glm5-serve-vision FIXTURE_DIR\n");
        return 2;
    }
    const std::string fixtures = argv[1];
#endif
    try {
        test_stack_and_frames(fixtures);
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    if (failures) return 1;
    std::puts("glm5-serve-vision: PASS");
    return 0;
}
