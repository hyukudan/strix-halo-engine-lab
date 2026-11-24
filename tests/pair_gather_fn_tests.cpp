#include "strix/pair/gather_fn.hpp"

#include <cstddef>
#include <cstdio>
#include <stdexcept>

namespace {

struct Seen {
    int calls = 0;
    void* stream = nullptr;
    void* slots = nullptr;
    std::size_t slot_bytes = 0, bytes = 0;
};

void fake(void* ctx, void* stream, void* slots, std::size_t slot_bytes, std::size_t bytes) {
    auto* s = static_cast<Seen*>(ctx);
    ++s->calls;
    s->stream = stream;
    s->slots = slots;
    s->slot_bytes = slot_bytes;
    s->bytes = bytes;
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int main() {
    Seen seen;
    int stream_tag = 0, slot_tag = 0;
    const strix::pair::GatherFn g{&seen, &fake};
    g(&stream_tag, &slot_tag, 64U, 48U);
    expect(seen.calls == 1, "one call");
    expect(seen.stream == &stream_tag && seen.slots == &slot_tag, "stream and slots forwarded");
    expect(seen.slot_bytes == 64U && seen.bytes == 48U, "sizes forwarded in order");

    unsigned short buf[40] = {};
    strix::pair::WireRing<unsigned short> ring;
    expect(ring.base() == nullptr && ring.slot() == 0U, "unbound ring");
    ring.bind(buf, 10U);
    unsigned short* a = ring.next(10U);
    unsigned short* b = ring.next(4U);
    unsigned short* c = ring.next(10U);
    expect(a == buf + 20, "first join uses the second buffer (flip 1)");
    expect(b == buf, "second join uses the first buffer (flip 0)");
    expect(c == buf + 20, "third join alternates again");
    bool refused = false;
    try {
        (void)ring.next(11U);
    } catch (const std::logic_error&) {
        refused = true;
    }
    expect(refused, "per_rank above the slot is refused");

    std::printf(failures == 0 ? "pair_gather_fn_tests: PASS\n" : "pair_gather_fn_tests: FAIL\n");
    return failures == 0 ? 0 : 1;
}
