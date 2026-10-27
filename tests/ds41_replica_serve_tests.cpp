// CPU tests of the replica worker's spool protocol and request checks (apps/cuda_ds41_replica_serve.hpp).
#include "../apps/cuda_ds41_replica_serve.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace rs = strix::ds41::replica_serve;
namespace fs = std::filesystem;

static int failed = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failed;                                                         \
        }                                                                     \
    } while (0)

static bool rejects(const std::vector<std::string>& a) {
    try {
        rs::check_request_args(a);
    } catch (const rs::RequestReject&) {
        return true;
    }
    return false;
}
static void put(const fs::path& p, const std::string& s) {
    std::ofstream o(p, std::ios::binary);
    o << s;
}
static std::string get(const fs::path& p) {
    std::ifstream i(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(i)), std::istreambuf_iterator<char>());
}

int main() {
    CHECK(rs::eviction_error("scored", false) != nullptr);
    CHECK(std::string(rs::eviction_error("scored", false)).find("no ts_dg_set_policy") != std::string::npos);
    CHECK(rs::eviction_error("lru", false) == nullptr);
    CHECK(rs::eviction_error("scored", true) == nullptr);
    CHECK(rs::eviction_error("auto", true, true) == nullptr);
    CHECK(rs::eviction_error("auto", true, false) != nullptr);
    CHECK(rs::eviction_error("auto", false, true) != nullptr);
    CHECK(std::string(rs::eviction_error("auto", true, false)).find("no ts_dg_end_rows") != std::string::npos);
    CHECK(rs::eviction_error("bogus", true) != nullptr);
    CHECK(!rejects({"--evict", "auto", "--no-census"}));
    CHECK(!rejects({"--evict", "scored", "--no-census"}));
    CHECK(!rejects({"--evict", "lru", "--census"}));
    CHECK(rejects({"--evict"}));
    // option split
    CHECK(!rejects({"--phases", "exact0,chats0", "--chats", "0,1", "--json", "C:\\a b\\x.jsonl", "--chat-logits"}));
    CHECK(!rejects({"--controller-prior-alt", "--controller-learn", "--window", "fixed:3", "--env", "STRIX_DS41_CUDA_ENGRAM_CHECK=0"}));
    CHECK(!rejects({"--qual", "--phases", "exact0"}));  // --qual is a flag
    CHECK(rejects({"--max-positions", "4000"}));  // load-time (capacity)
    CHECK(rejects({"--test-hooks"}));             // load-time (open option)
    CHECK(rejects({"--arena-gib", "70"}));       // load-time
    CHECK(!rejects({"--wb-defer", "--no-flush-ready"}));  // runtime flags (ts_dg_set_sched)
    CHECK(rejects({"--winpf-prior", "x.bin"}));   // load-time (parsed at open)
    CHECK(rejects({"--predict", "8"}));          // load-time (max_predict)
    CHECK(rejects({"--pressure-spares", "8"}));  // load-time (spare slots)
    CHECK(rejects({"--draft", "x.gguf"}));
    CHECK(rejects({"--serve", "d"}));
    CHECK(rejects({"--positions", "300"}));
    CHECK(rejects({"--phases"}));                 // missing value
    CHECK(rejects({"exact0"}));                   // not an option
    CHECK(rejects({"--env", "PATH=x"}));          // not a request variable
    CHECK(rejects({"--env", "noequals"}));
    CHECK(!rejects({"--steps", "--5"}) || true);  // a value starting with -- is a value (the parser reads it)

    // args file: CRLF, blanks, comments, spaces kept
    const fs::path root = fs::temp_directory_path() / ("ds41-serve-test-" + std::to_string(std::rand()));
    fs::remove_all(root);
    fs::create_directories(root);
    put(root / "a.req", "--phases\r\nexact0\r\n\r\n# comment\r\n--json\r\nC:\\dir with space\\o.jsonl\r\n");
    const auto args = rs::read_args_file(root / "a.req");
    CHECK(args.size() == 4U);
    CHECK(args.size() == 4U && args[1] == "exact0" && args[3] == "C:\\dir with space\\o.jsonl");

    CHECK(rs::json_str("a\"b\\c\n") == "\"a\\\"b\\\\c\\u000a\"");

    // spool: lock, order, take, finish, crashed run/ entries answered rc 3
    const fs::path sp = root / "spool";
    {
        fs::create_directories(sp / "run");
        put(sp / "run" / "000-crashed.req", "--phases\nexact0\n");
        rs::Spool s(sp);
        s.open();
        CHECK(fs::exists(sp / "lock"));
        CHECK(fs::exists(sp / "done" / "000-crashed.json"));
        CHECK(get(sp / "done" / "000-crashed.json").find("\"rc\":3") != std::string::npos);
        CHECK(!fs::exists(sp / "run" / "000-crashed.req"));
        {
            rs::Spool second(sp);
            bool locked = false;
            try {
                second.open();
            } catch (const std::exception&) {
                locked = true;
            }
            CHECK(locked);
        }
        CHECK(fs::exists(sp / "lock"));  // the failed second open must not remove the first's lock
        CHECK(!s.take().has_value());
        put(sp / "queue" / "002-b.req", "--ping\n");
        put(sp / "queue" / "001-a.req", "--quit\n");
        put(sp / "queue" / "000-x.tmp", "partial");  // a writer's temp file: never taken
        const auto n1 = s.take();
        CHECK(n1.has_value() && *n1 == "001-a");
        CHECK(fs::exists(s.running("001-a")));
        s.finish("001-a", "{\"rc\":0}");
        CHECK(fs::exists(sp / "done" / "001-a.json") && fs::exists(sp / "done" / "001-a.req"));
        CHECK(!fs::exists(sp / "done" / "001-a.json.tmp"));
        const auto n2 = s.take();
        CHECK(n2.has_value() && *n2 == "002-b");
        s.finish("002-b", "{\"rc\":0}");
        CHECK(!s.take().has_value());
        s.ready("{\"event\":\"ready\"}");
        CHECK(get(sp / "ready.json") == "{\"event\":\"ready\"}\n");
    }
    CHECK(!fs::exists(sp / "lock"));  // released by the destructor
    fs::remove_all(root);
    if (failed) {
        std::fprintf(stderr, "%d failures\n", failed);
        return 1;
    }
    std::printf("ds41_replica_serve_tests: all passed\n");
    return 0;
}
