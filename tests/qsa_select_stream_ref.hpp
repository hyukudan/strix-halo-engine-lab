// Shared host selection semantics and seeded case generators.
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace qsa_select_stream_ref {

using U = std::uint32_t;
constexpr U K = 512, R = 4, NEG_INF = 0xff800000U, CANARY = 0xa5a5a5a5U;
constexpr std::array<U, 14> sizes{0,1,511,512,513,514,1023,1024,1025,2047,2048,4096,16384,65536};
constexpr std::array<U, 8> slabs{1,7,32,64,255,256,257,1024};
constexpr std::array<U, 3> partitions{512,1024,4096};

inline U select_key(U bits) { // hip_indexed_attention_launcher.hip:1693-1697
    if ((bits & 0x7fffffffU) == 0) bits = 0;
    return bits & 0x80000000U ? ~bits : bits | 0x80000000U;
}
inline bool eligible(U bits) { return std::bit_cast<float>(bits) > -INFINITY; }
struct Image {
    std::vector<U> selected, scores;
    explicit Image(const std::vector<U>& input)
        : selected(K * R + 3 + 17, CANARY), scores(input) {}
};
struct Cut { U prefix = 0, mask = 0, need = K, zeros = 0; bool enough = true; };

// Literal 11/11/10 radix decisions. Scalar sums replace ballots, atomics and
// select_scan: all operands are exact non-overflowing unsigned counts. The
// high/low digit-pair scan is kept, including early exit with a partial mask.
template<class Value> Cut radix(U n, Value value, bool fallback) {
    Cut c;
    for (U pass = 0; pass < 3; ++pass) {
        const U shift = pass == 0 ? 21 : pass == 1 ? 10 : 0;
        const U digits = pass == 2 ? 1024 : 2048;
        std::array<U, 2048> histogram{};
        for (U i = 0; i < n; ++i) {
            U bits = value(i), key = select_key(bits);
            if (eligible(bits) && (key & c.mask) == c.prefix)
                ++histogram[(key >> shift) & (digits - 1)];
        }
        U total = std::accumulate(histogram.begin(), histogram.end(), 0U);
        if (pass == 0 && total <= c.need) {
            c.enough = total == c.need;
            if (fallback) { c.zeros = c.need - total; c.need = total; }
            return c;
        }
        U inclusive = 0, digit = 0, need = 0, count = 0;
        for (U tid = 0; tid < 1024; ++tid) {
            U high = 2 * tid + 1 < digits ? histogram[2 * tid + 1] : 0;
            U low = 2 * tid < digits ? histogram[2 * tid] : 0;
            inclusive += high + low;
            U above = total - inclusive;
            if (above < c.need && c.need <= above + high) {
                digit = 2 * tid + 1; need = c.need - above; count = high;
            } else if (above + high < c.need && c.need <= above + high + low) {
                digit = 2 * tid; need = c.need - above - high; count = low;
            }
        }
        c.prefix |= digit << shift; c.mask |= (digits - 1) << shift; c.need = need;
        if (count == c.need) break;
    }
    return c;
}

struct Paths { std::uint64_t dense=0, few=0, capacity=0, groups=0; };
inline Paths paths;
// Ordered compaction of either all blocks or the ascending candidate groups.
// g/e are the exclusive counts produced by the packed workgroup scans. Each
// 4096-block (fallback) or 1024-candidate tile is <=65535, so its two fields
// cannot carry into each other. Scores are read from the immutable input, as
// each device lane classifies its own block before overwriting that block.
inline void compact(Image& out, const std::vector<U>& input, const std::vector<U>& ids,
             Cut c, U tail) {
    for (U token = 0; token < c.zeros * R; ++token) out.selected[token] = token % R;
    U g = 0, e = 0;
    for (U b : ids) {
        U bits = b < input.size() ? input[b] : NEG_INF;
        U masked = select_key(bits) & c.mask;
        U kind = !eligible(bits) ? 0 : masked > c.prefix ? 1 : masked == c.prefix ? 2 : 0;
        if (kind == 1 || (kind == 2 && e < c.need)) {
            U slot = c.zeros + g + std::min(e, c.need);
            for (U o = 0; o < R; ++o) out.selected.at(slot * R + o) = b * R + o;
            out.scores.at(b) = NEG_INF;
        }
        g += kind == 1; e += kind == 2;
    }
    if (c.zeros) out.scores.at(0) = NEG_INF;
    for (U t = 0; t < tail; ++t) out.selected[K * R + t] = U(input.size()) * R + t;
}

// select_row_body:247-351, with select_blocks_parallel_body:1715-1793
// fallback. Dynamic graph arguments and stats atomics are outside this fixture.
inline Image reference(const std::vector<U>& input, U tail, bool group_shortcut = true) {
    Image out(input);
    U n = U(input.size());
    if (n <= K) {
        if (group_shortcut) ++paths.dense;
        for (U i = 0; i < n * R + tail; ++i) out.selected[i] = i;
        return out;
    }
    std::vector<U> ids;
    if (group_shortcut) {
        std::vector<U> maxima((n + 3) / 4);
        // Selection-equivalent numeric group maxima for the producer's xor-1
        // then xor-2 fmaxf tree (lines 226-229). Do not use host std::fmax:
        // some host libraries propagate signaling NaNs even with a numeric
        // operand. The device numeric-max semantics suppress NaN operands.
        // Maxima bits are not an observed fixture output; all-NaN may use -inf
        // and either zero sign is equivalent under eligibility/select_key.
        auto numeric_max = [](U a, U b) {
            if (!eligible(a)) return eligible(b) ? b : NEG_INF;
            if (!eligible(b)) return a;
            return select_key(a) >= select_key(b) ? a : b;
        };
        for (U g = 0; g < maxima.size(); ++g) {
            std::array<U,4> x;
            for (U j = 0; j < 4; ++j) x[j] = g*4+j < n ? input[g*4+j] : NEG_INF;
            maxima[g] = numeric_max(numeric_max(x[0],x[1]),numeric_max(x[2],x[3]));
        }
        Cut c = radix(U(maxima.size()), [&](U i){ return maxima[i]; }, false);
        if (c.enough) {
            for (U g = 0; g < maxima.size(); ++g)
                if (eligible(maxima[g]) && (select_key(maxima[g]) & c.mask) >= c.prefix)
                    for (U j = 0; j < 4; ++j) ids.push_back(g*4+j);
        }
        if (c.enough && ids.size() <= 6144) {
            ++paths.groups;
            Cut block_cut = radix(U(ids.size()), [&](U i){ return ids[i] < n ? input[ids[i]] : NEG_INF; }, false);
            compact(out, input, ids, block_cut, tail);
            return out;
        }
        if (!c.enough) ++paths.few; else ++paths.capacity;
    }
    ids.resize(n); std::iota(ids.begin(), ids.end(), 0U);
    Cut c = radix(n, [&](U i){ return input[i]; }, true);
    compact(out, input, ids, c, tail);
    return out;
}

struct Record { U key, block; };
static_assert(sizeof(Record) == 8);
inline bool better(const Record& a, const Record& b) {
    return a.key != b.key ? a.key > b.key : a.block < b.block;
}
// Integer rank merge: slab candidates are sorted by integer total order, then
// the first <=512 ranks of the two ordered lists survive. No score comparator.
inline void merge_winners(std::vector<Record>& winners, std::vector<Record> slab) {
    std::sort(slab.begin(), slab.end(), better);
    std::vector<Record> next;
    next.reserve(K);
    std::size_t a=0, b=0;
    while (next.size() < K && (a < winners.size() || b < slab.size())) {
        if (b == slab.size() || (a < winners.size() && better(winners[a], slab[b])))
            next.push_back(winners[a++]);
        else next.push_back(slab[b++]);
    }
    winners.swap(next);
}
inline std::vector<Record> stream(const std::vector<U>& input, U begin, U end, U slab_size = 256) {
    if (!slab_size) throw std::runtime_error("zero slab size");
    std::vector<Record> winners;
    for (U base = begin; base < end;) {
        U stop = std::min(end, base + slab_size);
        std::vector<Record> slab;
        slab.reserve(stop-base);
        // Honor the original absolute 32-block unit boundaries even when slabs
        // or partitions split a unit. No floating arithmetic is regrouped here.
        for (U unit = base; unit < stop;) {
            U unit_end = std::min(stop, (unit / 32 + 1) * 32);
            for (U b = unit; b < unit_end; ++b)
                if (eligible(input[b])) slab.push_back({select_key(input[b]), b});
            unit = unit_end;
        }
        // A singleton losing to rank 512 cannot affect the invariant.
        if (!(slab.size()==1 && winners.size()==K && !better(slab[0], winners.back())))
            merge_winners(winners, std::move(slab));
        base = stop;
    }
    return winners;
}
inline Image emit(const std::vector<U>& input, U tail, std::vector<Record> winners) {
    Image out(input);
    U n = U(input.size());
    if (n <= K) {
        for (U t=0; t<n*R+tail; ++t) out.selected[t]=t;
        return out;
    }
    std::sort(winners.begin(), winners.end(), [](Record a, Record b){ return a.block < b.block; });
    U missing = K - U(winners.size()), slot = 0;
    for (; slot < missing; ++slot)
        for (U o=0; o<R; ++o) out.selected[slot*R+o]=o;
    for (Record w : winners) {
        for (U o=0; o<R; ++o) out.selected[slot*R+o]=w.block*R+o;
        out.scores[w.block]=NEG_INF;
        ++slot;
    }
    if (missing) out.scores[0]=NEG_INF;
    for (U t=0; t<tail; ++t) out.selected[K*R+t]=n*R+t;
    return out;
}
struct TwoLaunch { Image image; std::uint64_t bytes; };
inline TwoLaunch two_launch(const std::vector<U>& input, U tail, U slab, U partition) {
    if (!partition) throw std::runtime_error("zero partition size");
    // Retain all first-launch lists before starting the second launch.
    std::vector<std::vector<Record>> partials;
    std::uint64_t bytes=0;
    if (input.size() > K) {
        for (U base=0; base<input.size(); base+=partition) {
            partials.push_back(stream(input,base,std::min(U(input.size()),base+partition),slab));
            bytes += partials.back().size()*sizeof(Record);
        }
    }
    std::vector<Record> winners;
    for (auto& partial : partials) merge_winners(winners,std::move(partial));
    return {emit(input,tail,std::move(winners)),bytes};
}
inline void same(const Image& a, const Image& b, const std::string& context) {
    auto check = [&](const std::vector<U>& x, const std::vector<U>& y, const char* field) {
        if (x.size()!=y.size() || (!x.empty() && std::memcmp(x.data(),y.data(),x.size()*sizeof(U)))) {
            U i=0; while (i<x.size() && i<y.size() && x[i]==y[i]) ++i;
            throw std::runtime_error(context+" "+field+" mismatch at "+std::to_string(i)
                +" lhs="+std::to_string(i<x.size()?x[i]:0)+" rhs="+std::to_string(i<y.size()?y[i]:0));
        }
    };
    check(a.selected,b.selected,"selected"); check(a.scores,b.scores,"scores");
}

inline std::vector<U> make_case(U n, U kind, std::mt19937& rng) {
    std::vector<U> v(n);
    constexpr std::array<U,12> special{0,0x80000000U,0x7f800000U,NEG_INF,
        0x7fc00001U,0x7fc12345U,0xffc00001U,0xffdabcdeU,
        0x7f800001U,0xff800123U,0x3f800000U,0xbf800000U};
    for (U b=0;b<n;++b) {
        switch(kind) {
        case 0: v[b]=0x3f800000U; break; // all equal
        case 1: v[b]=b%11==0 ? 0x40000000U : b%7==0 ? 0 : 0x3f800000U; break;
        case 2: v[b]=b%2 ? 0 : 0x80000000U; break;
        case 3: v[b]=special[b%special.size()]; break;
        case 4: v[b]=NEG_INF; break;
        case 5: v[b]=special[4+b%6]; break; // NaNs, both signs, signaling/quiet
        case 6: case 7: v[b]=special[3+b%7]; break; // replaced below
        case 8: v[b]=(b%2 ? 0x80000000U : 0) | (1+U(rng()%0x7fffffU)); break;
        case 9: v[b]=(U(rng())&0x807fffffU) | ((1+U(rng()%254))<<23); break;
        default: v[b]=U(rng()); break;
        }
    }
    if (kind==6 || kind==7) {
        std::vector<U> order(n); std::iota(order.begin(),order.end(),0U);
        std::shuffle(order.begin(),order.end(),rng);
        for (U b=0;b<n;++b) v[b]=b%2 ? NEG_INF : special[4+b%6];
        U count=std::min(n,kind==6 ? 173U : K);
        // Keep block 0 ineligible when possible, exposing its extra overwrite.
        order.erase(std::remove(order.begin(),order.end(),0U),order.end());
        if (count>order.size()) order.push_back(0);
        for (U i=0;i<count;++i) v[order[i]]=i%3 ? 0x3f800000U : 0x80000000U;
    }
    return v;
}

constexpr std::array<const char*,11> labels{"all-equal","boundary-ties","signed-zero",
    "infinity-and-NaN","all-negative-infinity","all-NaN","fewer-eligible",
    "exactly-512-eligible","denormals","random-normal","random-bits"};

} // namespace qsa_select_stream_ref
