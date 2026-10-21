#include "strix/model_family_registry.hpp"

#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace strix {
namespace {

struct Registry {
    std::mutex mutex;
    std::vector<std::unique_ptr<ModelFamily>> families;
};

Registry& registry() {
    static Registry r;
    return r;
}

}  // namespace

void register_model_family(ModelFamily family) {
    if (family.name.empty() || family.architectures.empty() || !family.validate)
        throw std::invalid_argument("model family registration needs a name, architectures and a validator");
    auto& r = registry();
    const std::lock_guard lock(r.mutex);
    for (const auto& f : r.families) {
        if (f->name == family.name) {
            if (f->architectures == family.architectures) return;  // the same family registered again
            throw std::invalid_argument("model family " + family.name + " is already registered differently");
        }
        for (const auto& a : family.architectures)
            if (std::find(f->architectures.begin(), f->architectures.end(), a) != f->architectures.end())
                throw std::invalid_argument("architecture " + a + " is already admitted by family " + f->name);
    }
    r.families.push_back(std::make_unique<ModelFamily>(std::move(family)));
}

bool model_family_registered(std::string_view name) {
    auto& r = registry();
    const std::lock_guard lock(r.mutex);
    return std::any_of(r.families.begin(), r.families.end(), [&](const auto& f) { return f->name == name; });
}

const ModelFamily* find_model_family(std::string_view architecture) {
    auto& r = registry();
    const std::lock_guard lock(r.mutex);
    for (const auto& f : r.families)
        if (std::find(f->architectures.begin(), f->architectures.end(), architecture) != f->architectures.end()) return f.get();
    return nullptr;
}

std::vector<std::string> registered_model_families() {
    auto& r = registry();
    const std::lock_guard lock(r.mutex);
    std::vector<std::string> names;
    for (const auto& f : r.families) names.push_back(f->name);
    return names;
}

AdmittedFile admit_model_file(const std::filesystem::path& path) {
    const auto probe = gguf::read_file(path);
    const auto* arch = probe.find_metadata("general.architecture");
    if (arch == nullptr || !arch->string_value.has_value()) throw std::invalid_argument("model file has no general.architecture");
    const auto* family = find_model_family(*arch->string_value);
    if (family == nullptr) throw std::invalid_argument("no registered model family admits architecture " + *arch->string_value);
    AdmittedFile out;
    out.family = family;
    out.info = family->reader_limits ? gguf::read_file(path, family->reader_limits()) : probe;
    family->validate(out.info);
    return out;
}

}  // namespace strix
