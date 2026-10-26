#include "strix/ds41_prefill_tuning.hpp"
#include <cstdio>

using namespace strix::ds41::prefill_tuning;
static_assert(attention_rows(516, 1024, 64) == 64);
static_assert(attention_rows(2048, 1024, 64) == 128);
static_assert(attention_rows(8192, 1024, 64) == 256);
static_assert(attention_rows(8192, 192, 64) == 64);
static_assert(attention_rows(8192, 24, 64) == 8);
static_assert(attention_rows(8192, 1024, 64, 1048576) == 32);
static_assert(attention_rows(8192, 1024, 64, 1048576, 16) == 8);
static_assert(dense_outputs(5120, 1792, 1024) == 16);
static_assert(dense_outputs(1280, 32768, 1024) == 32);
static_assert(moe_rows(5120, 1280, 4096, 6, 384) == 16);
static_assert(moe_rows(1024, 512, 4096, 6, 128) == 32);
static_assert(down_stage(1280, 5120));
static_assert(!down_stage(2304, 5120));
int main() {
    for (unsigned chunk = 1; chunk <= 2048; ++chunk) {
        const unsigned rows = attention_rows(32768, chunk, 64);
        if (!rows || rows > 256 || chunk % rows) return 1;
    }
    std::puts("prefill resource selector checks pass");
}
