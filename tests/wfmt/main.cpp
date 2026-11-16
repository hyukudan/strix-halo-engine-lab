#include "cases.hpp"
#include <iostream>
#include <string_view>
int wfmt_census_test();
bool wfmt_contracts_test();
int main(int argc, char** argv) {
    if (argc == 5 && std::string_view(argv[1]) == "--wfmt-capture" &&
        std::string_view(argv[3]) == "--wfmt-matrix")
        return strix::weight::test::capture_case(argv[2]) ? 0 : 1;
    if (argc != 2) return 2;
    const std::string_view name(argv[1]);
    bool ok = false;
    if (name == "contracts") ok = wfmt_contracts_test();
    else if (name == "census") ok = wfmt_census_test() == 0;
    else if (name == "compare") ok = strix::weight::test::compare_case();
    else return 2;
    std::cout << name << (ok ? " PASS\n" : " FAIL\n");
    return ok ? 0 : 1;
}
