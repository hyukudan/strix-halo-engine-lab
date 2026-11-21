#pragma once

#include "resolve.hpp"

#include <functional>
#include <map>
#include <string>

namespace strix::weight {

struct TensorLaunchPlan {
    Plan launch;
    ByteRange payload;
    Geometry layout;
};
using TensorLaunchPlans = std::map<std::string, TensorLaunchPlan, std::less<>>;

std::string tensor_launch_plans_json(const TensorLaunchPlans&);

} // namespace strix::weight
