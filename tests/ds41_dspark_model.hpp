#pragma once
// Test helper: the drafter's weights from the two model files, synthetic inputs and raw dumps.

#include "strix/ds41_dspark.hpp"
#include "strix/ds41_dspark_files.hpp"

#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace ds41_dspark_test {

using namespace strix::ds41::dspark;
using strix::ds41::WeightFormat;

struct Files {
    std::string q2, dspark;
};
struct Model {
    std::unique_ptr<LoadedWeights> loaded;
    Weights w;
};
inline std::unique_ptr<Model> load_model(const Files& f) {
    auto m = std::make_unique<Model>();
    m->loaded = load_weights(f.q2, f.dspark);
    m->w = m->loaded->weights;
    return m;
}

inline std::vector<float> normal_values(std::size_t n, float sigma, unsigned seed) {
    std::mt19937 random(seed);
    std::normal_distribution<float> normal(0.0F, sigma);
    std::vector<float> v(n);
    for (auto& x : v) x = normal(random);
    return v;
}
inline void dump(const std::string& dir, const std::string& name, const void* data, std::size_t bytes) {
    if (dir.empty()) return;
    std::ofstream out(dir + "/" + name, std::ios::binary);
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

} // namespace ds41_dspark_test
