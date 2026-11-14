#include "strix/deepseek41_prefix_cache.hpp"
#include "strix/ds41_fast_profile.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void env(const char* name, const char* value) {
#ifdef _WIN32
    require(_putenv_s(name, value) == 0, "set environment");
#else
    require((*value ? setenv(name, value, 1) : unsetenv(name)) == 0, "set environment");
#endif
}
int main() {
    try {
        namespace ds = strix::deepseek41;
        namespace fast = strix::ds41::fast_profile;
        env("STRIX_DS41_FAST_PREFILL", "0");
        env("STRIX_DS41_FAST_ATTN", "");
        env("STRIX_DS41_FAST_MOE", "");
        env("STRIX_DS41_FAST_DENSE", "1");
        env("STRIX_DS41_FAST_DENSE_MIN_ROWS", "");
        const auto initial = ds::prefix_namespace_from_env(false);
        require(fast::dense_min_rows() == 128U, "default dispatch threshold");
        env("STRIX_DS41_FAST_DENSE_MIN_ROWS", "4096");
        const auto large = ds::prefix_namespace_from_env(false);
        require(fast::dense_min_rows() == 4096U, "explicit dispatch threshold");
        require(initial.tag != large.tag, "different dense arithmetic aliases the prefix namespace");
        require(large.text.find("dense_min_rows=4096;") != std::string::npos, "effective threshold not reported");
        const std::filesystem::path model = "pr584-model.bin";
        { std::ofstream fixture(model, std::ios::binary); fixture << "prefix identity fixture"; }
        const bool distinct_models = ds::prefix_model_id(model, 1U, initial) != ds::prefix_model_id(model, 1U, large);
        std::filesystem::remove(model);
        require(distinct_models, "persistent model IDs alias different thresholds");
        env("STRIX_DS41_FAST_DENSE_MIN_ROWS", "004096");
        require(ds::prefix_namespace_from_env(false).tag == large.tag, "threshold not normalized");
        env("STRIX_DS41_FAST_DENSE_MIN_ROWS", "0");
        require(fast::dense_min_rows() == 0U, "explicit zero threshold");
        env("STRIX_DS41_FAST_DENSE_MIN_ROWS", "-1");
        require(fast::dense_min_rows() == std::numeric_limits<unsigned>::max(), "legacy unsigned threshold");
        env("STRIX_DS41_FAST_DENSE", "0");
        const auto off = ds::prefix_namespace_from_env(false);
        env("STRIX_DS41_FAST_DENSE_MIN_ROWS", "128");
        require(ds::prefix_namespace_from_env(false).tag == off.tag, "inactive threshold changes exact namespace");
        env("STRIX_DS41_FAST_PREFILL", "1");
        require(!fast::dense_on() && fast::attention_flash() && fast::moe_on(), "explicit OFF must beat bundle");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
