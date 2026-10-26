// glm-rank-plan: CPU checks of the TP2 rank-slice plan (design notes).
//   glm-rank-plan sizes TENSORS_JSON SPLIT_JSON
//       Saved headers of the full model ({"tensors": {name: {shape, dtype, bytes}}}, design notes): both ranks'
//       slice bytes must equal the split contract's rank_bytes exactly; prints the read statistics (pieces, reads
//       after coalescing, piece-size histogram).
//   glm-rank-plan pieces MODEL_DIR SPLIT_JSON OUT_JSON
//       Safetensors in MODEL_DIR (e.g. the EXL3 fixture): writes every slice's pieces for both ranks, for
//       tools/glm/check_rank_plan.py (numpy slicing reference, test_split.py's convention).
#include "glm/load/glm_rank_plan.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>

namespace gl = strix::glm::load;

namespace {

// The saved-headers JSON is large (150k tensors) and flat: a small scanner for name -> {shape, dtype, bytes}.
std::map<std::string, gl::TensorInfo> read_saved(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::map<std::string, gl::TensorInfo> out;
    std::size_t p = s.find("\"tensors\"");
    if (p == std::string::npos) throw std::runtime_error("no tensors");
    p = s.find('{', p) + 1;
    std::uint64_t at = 0;
    for (;;) {
        const std::size_t q = s.find('"', p);
        if (q == std::string::npos) break;
        const std::size_t qe = s.find('"', q + 1);
        const std::string name = s.substr(q + 1, qe - q - 1);
        const std::size_t ob = s.find('{', qe), oe = s.find('}', ob);
        const std::string body = s.substr(ob, oe - ob);
        gl::TensorInfo t;
        const std::size_t sh = body.find('['), she = body.find(']', sh);
        const std::string dims = body.substr(sh + 1, she - sh - 1);
        for (std::size_t a = 0; a < dims.size();) {
            const std::size_t c = dims.find(',', a);
            const std::string d = dims.substr(a, c == std::string::npos ? std::string::npos : c - a);
            if (d.find_first_not_of(' ') != std::string::npos) t.shape.push_back(std::stoull(d));
            if (c == std::string::npos) break;
            a = c + 1;
        }
        const std::size_t dt = body.find("\"dtype\"");
        const std::size_t d1 = body.find('"', body.find(':', dt) + 1), d2 = body.find('"', d1 + 1);
        t.dtype = body.substr(d1 + 1, d2 - d1 - 1);
        const std::size_t by = body.find("\"bytes\"");
        t.bytes = std::stoull(body.substr(body.find(':', by) + 1));
        t.offset = at;
        at += t.bytes;
        out.emplace(name, std::move(t));
        p = oe + 1;
        while (p < s.size() && (s[p] == ' ' || s[p] == ',' || s[p] == '\n')) ++p;
        if (p >= s.size() || s[p] == '}') break;
    }
    return out;
}

unsigned long long ull(std::uint64_t v) { return static_cast<unsigned long long>(v); }

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "sizes" && argc == 4) {
            const gl::Index index(read_saved(argv[2]));
            const gl::Split split(argv[3]);
            int fail = 0;
            for (unsigned r = 0; r < 2; ++r) {
                std::size_t missing = 0;
                const auto plan = gl::rank_plan(index, split, r, &missing);
                std::uint64_t bytes = 0, pieces = 0, reads = 0, small = 0;
                std::map<std::uint64_t, std::uint64_t> hist;
                for (const auto& sl : plan) {
                    bytes += sl.bytes;
                    pieces += sl.pieces.size();
                    reads += gl::coalesce(sl.pieces).size();
                    for (const auto& pc : sl.pieces) {
                        std::uint64_t b = 4096;
                        while (b < pc.len && b < (std::uint64_t{1} << 30U)) b <<= 1U;
                        ++hist[b];
                        small += pc.len <= 16384 ? 1 : 0;
                    }
                }
                const bool ok = missing == 0 && static_cast<double>(bytes) == split.rank_bytes(r);
                fail += ok ? 0 : 1;
                std::printf("rank %u: %zu slices, %llu B (contract %.0f) %s, missing %zu, pieces %llu (<= 16 KiB: %llu), reads after coalescing %llu\n", r,
                            plan.size(), ull(bytes), split.rank_bytes(r), ok ? "EQUAL" : "DIFFER", missing, ull(pieces), ull(small), ull(reads));
                for (const auto& [b, c] : hist) std::printf("   pieces <= %10llu B: %llu\n", ull(b), ull(c));
            }
            std::printf("sizes: %s\n", fail == 0 ? "PASS" : "FAIL");
            return fail == 0 ? 0 : 1;
        }
        if (mode == "pieces" && argc == 5) {
            const gl::Index index(argv[2]);
            const gl::Split split(argv[3]);
            std::FILE* o = std::fopen(argv[4], "w");
            if (o == nullptr) throw std::runtime_error("cannot write " + std::string(argv[4]));
            std::fprintf(o, "{\"files\": [");
            for (std::size_t i = 0; i < index.files().size(); ++i) std::fprintf(o, "%s\"%s\"", i ? ", " : "", index.files()[i].c_str());
            std::fprintf(o, "], \"ranks\": [");
            for (unsigned r = 0; r < 2; ++r) {
                std::size_t missing = 0;
                const auto plan = gl::rank_plan(index, split, r, &missing);
                std::fprintf(o, "%s{", r ? ", " : "");
                bool first = true;
                for (const auto& sl : plan) {
                    std::fprintf(o, "%s\"%s\": {\"exl3\": %s, \"bytes\": %llu, \"tr\": %llu, \"suh\": %llu, \"svh\": %llu, \"mul1\": %llu, \"rows\": %llu, \"cols\": %llu, \"pieces\": [",
                                 first ? "" : ", ", sl.name.c_str(), sl.exl3 ? "true" : "false", ull(sl.bytes), ull(sl.tr_bytes), ull(sl.suh_bytes),
                                 ull(sl.svh_bytes), ull(sl.mul1_bytes), ull(sl.rows), ull(sl.cols));
                    const auto co = gl::coalesce(sl.pieces);
                    for (std::size_t i = 0; i < co.size(); ++i) std::fprintf(o, "%s[%u, %llu, %llu]", i ? ", " : "", co[i].file, ull(co[i].src), ull(co[i].len));
                    std::fprintf(o, "]}");
                    first = false;
                }
                std::fprintf(o, "}");
                std::printf("rank %u: %zu slices from the files, %zu split entries not in this directory\n", r, plan.size(), missing);
            }
            std::fprintf(o, "]}\n");
            std::fclose(o);
            return 0;
        }
        std::fprintf(stderr, "usage: %s sizes TENSORS_JSON SPLIT_JSON | pieces MODEL_DIR SPLIT_JSON OUT_JSON\n", argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "glm-rank-plan error: %s\n", e.what());
        return 3;
    }
}
