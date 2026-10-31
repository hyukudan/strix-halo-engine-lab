// Separate translation unit, without LTO: keep the legacy payload copies in the
// benchmark without hashing, scanning or writing any model bytes elsewhere.
void glm_io_bench_observe(const void*, unsigned long long) {}
