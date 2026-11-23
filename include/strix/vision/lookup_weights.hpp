#pragma once
// WeightSource over a strix::weight::TensorLookup (safetensors and any container the weight library indexes): each
// requested tensor is read once to FP32 host storage that this source owns, viewed as [extent 0][product of the rest]
// (the framework's [out][in...] weight; a 1-D tensor is [n][1]). Views stay valid while the source lives.
#include "strix/vision/adapter.hpp"
#include "strix/weight/tensor_access.hpp"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::vision {

class LookupWeights final : public WeightSource {
public:
    explicit LookupWeights(const weight::TensorLookup& lookup) : lookup_(lookup) {}

    bool has(std::string_view name) const override { return lookup_.find(name) != nullptr; }

    Tensor tensor(std::string_view name) override {
        const auto hit = owned_.find(name);
        if (hit != owned_.end()) return hit->second.view;
        const weight::RawTensor* raw = lookup_.find(name);
        if (!raw) throw std::out_of_range("vision weights: no tensor " + std::string(name));
        if (raw->extents.empty()) throw std::invalid_argument("vision weights: scalar tensor " + std::string(name));
        std::uint64_t cols = 1;
        for (std::size_t i = 1; i < raw->extents.size(); ++i) cols *= raw->extents[i];
        const std::uint64_t rows = raw->extents[0];
        Entry e;
        e.data.resize(static_cast<std::size_t>(rows * cols));
        const weight::Status st = lookup_.read_floats(name, e.data);
        if (!st) throw std::invalid_argument("vision weights: " + std::string(name) + " is not readable as floats");
        auto& stored = owned_.emplace(std::string(name), std::move(e)).first->second;
        stored.view = Tensor{stored.data.data(), Tensor::Type::f32, static_cast<std::size_t>(rows), static_cast<std::size_t>(cols)};
        return stored.view;
    }

private:
    struct Entry {
        std::vector<float> data;
        Tensor view;
    };
    const weight::TensorLookup& lookup_;
    std::map<std::string, Entry, std::less<>> owned_;
};

}  // namespace strix::vision
