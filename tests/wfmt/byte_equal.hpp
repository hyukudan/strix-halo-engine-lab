#pragma once
#include <cstring>
#include <vector>
namespace strix::weight::test {
inline bool byte_equal(const std::vector<float>& a,const std::vector<float>& b) {
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);
}
}
