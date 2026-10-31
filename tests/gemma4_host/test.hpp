#pragma once
#include <iostream>
#include <stdexcept>
#include <string>
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("check failed: ") + #x + " line " + std::to_string(__LINE__)); } while (false)
template<class F> void refuses(F f) { bool failed = false; try { f(); } catch (const std::exception&) { failed = true; } CHECK(failed); }
