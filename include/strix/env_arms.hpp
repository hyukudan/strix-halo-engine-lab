#pragma once
// In-process A/B arms (shared host layer, every rank app can bind it): one process loads the weights once and runs
// several switch sets ("arms") in a strict ABAB order, so a bundle ticket measures all arms of an ABAB without one
// reload per arm. Spec: "LABEL:K=V+K=V;LABEL2:-" ('/' also separates arms: shell-safe; "-" = the launch environment). Before each arm every key named by
// any arm is restored to its launch value, then that arm's pairs are set. Only switches read per call (getenv /
// envcfg, not cached at startup) are valid arm keys; both ranks of a pair must get the same spec (same CLI).
#include <algorithm>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::env_arms {

struct Arm {
    std::string label;
    std::vector<std::pair<std::string, std::string>> kv;
};

struct Arms {
    std::vector<Arm> arms;
    std::vector<std::pair<std::string, std::optional<std::string>>> launch;  // every key named by any arm
};

inline void set_env(const std::string& k, const std::optional<std::string>& v) {
#ifdef _WIN32
    _putenv_s(k.c_str(), v ? v->c_str() : "");
#else
    if (v) setenv(k.c_str(), v->c_str(), 1);
    else unsetenv(k.c_str());
#endif
}

inline Arms parse(const std::string& spec) {
    Arms out;
    std::size_t a = 0;
    while (a < spec.size()) {
        const std::size_t b = std::min(spec.find_first_of(";/", a), spec.size());
        const std::string item = spec.substr(a, b - a);
        const std::size_t c = item.find(':');
        if (c == std::string::npos || c == 0) throw std::invalid_argument("arms: LABEL:K=V+K=V;LABEL:- expected, got '" + item + "'");
        Arm arm{item.substr(0, c), {}};
        const std::string body = item.substr(c + 1);
        if (body != "-") {
            std::size_t p = 0;
            while (p <= body.size()) {
                const std::size_t q = std::min(body.find('+', p), body.size());
                const std::string kv = body.substr(p, q - p);
                const std::size_t e = kv.find('=');
                if (e == std::string::npos || e == 0) throw std::invalid_argument("arms: K=V expected, got '" + kv + "'");
                arm.kv.emplace_back(kv.substr(0, e), kv.substr(e + 1));
                p = q + 1;
            }
        }
        for (const auto& [k, v] : arm.kv) {
            bool seen = false;
            for (const auto& l : out.launch) seen = seen || l.first == k;
            if (!seen) {
                const char* raw = std::getenv(k.c_str());
                out.launch.emplace_back(k, raw ? std::optional<std::string>(raw) : std::nullopt);
            }
        }
        out.arms.push_back(std::move(arm));
        a = b + 1;
    }
    if (out.arms.empty()) throw std::invalid_argument("arms: empty spec");
    return out;
}

inline void apply(const Arms& all, const Arm& arm) {
    for (const auto& [k, v] : all.launch) set_env(k, v);
    for (const auto& [k, v] : arm.kv) set_env(k, v);
}

inline void restore(const Arms& all) {
    for (const auto& [k, v] : all.launch) set_env(k, v);
}

}  // namespace strix::env_arms
