// MiniMax M3.1 defaults-on resolver, CPU only: the plain invocation resolves to exactly the measured production
// receipt (argv + env, tests/fixtures/minimax_production_receipt.txt), every off switch restores its item's former
// value, STRIX_MM_BASELINE=1 restores all of them, and apply() exports what --print-effective-config prints.
#include "minimax/engine/mm_defaults.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace md = strix::minimax::defaults;

int g_failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        std::cerr << "FAIL: " << what << "\n";
    }
}

using EnvMap = std::map<std::string, std::string>;
md::Lookup lookup(const EnvMap& m) {
    return [&m](const char* name) -> const char* {
        const auto it = m.find(name);
        return it == m.end() ? nullptr : it->second.c_str();
    };
}

std::vector<std::string> words(const std::string& s) {
    std::istringstream in(s);
    std::vector<std::string> out;
    for (std::string w; in >> w;) out.push_back(w);
    return out;
}

struct Receipt {
    std::string app;
    std::vector<std::string> argv, plain;
    EnvMap env;
};

std::vector<Receipt> load(const char* path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    std::vector<Receipt> out;
    Receipt cur;
    bool open = false;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        const auto sp = line.find(' ');
        const std::string key = line.substr(0, sp), rest = sp == std::string::npos ? "" : line.substr(sp + 1);
        if (key == "receipt") {
            cur = Receipt{};
            cur.app = rest;
            open = true;
        } else if (key == "argv") cur.argv = words(rest);
        else if (key == "plain") cur.plain = words(rest);
        else if (key == "env") {
            const auto eq = rest.find('=');
            if (eq == std::string::npos) throw std::runtime_error("bad env line: " + line);
            cur.env[rest.substr(0, eq)] = rest.substr(eq + 1);
        } else if (key == "end" && open) {
            out.push_back(cur);
            open = false;
        } else throw std::runtime_error("bad fixture line: " + line);
    }
    return out;
}

std::set<std::string> lines(const std::string& text) {
    std::set<std::string> s;
    std::istringstream in(text);
    for (std::string l; std::getline(in, l);) s.insert(l);
    return s;
}

// The value each item replaced (measurement evidence), written out independently of the table.
const std::map<std::string, std::string> kFormer = {
    {"STRIX_MMX_MOE_WIDE", "0"},           {"STRIX_MM_INDEX_CONC", "0"},       {"STRIX_MMX_CRC", "1"},
    {"STRIX_MM_XB_WAIT", "0"},             {"STRIX_MM_XB_INPLACE", "0"},       {"STRIX_MM_XB_PUBLISH", "0"},
    {"STRIX_MM_XB_POSTILP", "0"},          {"STRIX_MM_XB_WARM", "0"},          {"STRIX_MM_DSPARK_ATTN", "0"},
    {"STRIX_MM_DRAFT_LIN", "lib"},         {"STRIX_MMX_PREFILL_CORE", "lib"},  {"STRIX_MM_PF_MSA_BATCH", "0"},
    {"STRIX_MMX_LINEAR_V11", "0"},
    {"STRIX_MM_MARKOV", "full"},           {"STRIX_MM_MARKOV_C", "1024"},      {"STRIX_MM_MARKOV_H", "2048"},
    {"STRIX_MM_MARKOV_TOPC", "v1"},
    {"STRIX_MM_PF_ATTEND4", "0"},          {"STRIX_MM_PF_ROUTER_BATCH", "0"},  {"STRIX_MMX_PREFILL_MTFIT", "0"},
    {"STRIX_MM_PF_MSA_REWORK", "0"},       {"STRIX_MMX_PREFILL_PIPE", "0"},    {"STRIX_MM_PF_KV_TILE", "off"},
    {"STRIX_MM_PF_TX_UNCACHED", "0"},      {"STRIX_DIST_CRC_PARALLEL", "0"},   {"STRIX_DIST_HOST_CRC_OVERLAP", "0"},
    {"STRIX_DIST_HOST_CHUNK_BYTES", "0"},  {"STRIX_MM_PF_IDX_TILE", "0"},
    {"STRIX_MM_PF_IDX_ROWS", "32"},
};

void test_table() {
    std::set<std::string> envs, names;
    for (const md::Item& it : md::items()) {
        check(envs.insert(it.env).second, std::string("unique env ") + it.env);
        check(names.insert(it.name).second, std::string("unique name ") + it.name);
        check(std::strcmp(it.on, it.off) != 0, std::string("on != off for ") + it.env);
        check(std::strlen(it.receipt) > 10, std::string("receipt for ") + it.env);
        const auto f = kFormer.find(it.env);
        check(f != kFormer.end() && f->second == it.off, std::string("off value is the former path for ") + it.env);
    }
    check(envs.size() == kFormer.size(), "every former value has an item");
}

void test_receipts(const char* path) {
    const auto receipts = load(path);
    check(receipts.size() == 2, "fixture has the rank and serve receipts");
    const EnvMap none;
    for (const Receipt& r : receipts) {
        const std::string want = md::effective_config(md::parse_cli(r.app, r.argv), lookup(r.env));
        const std::string got = md::effective_config(md::parse_cli(r.app, r.plain), lookup(none));
        check(got == want, r.app + ": plain invocation == receipt\n--- receipt\n" + want + "--- plain\n" + got);
        // Every receipt env value is one of the resolved items (no receipt switch is left unmanaged).
        for (const auto& [k, v] : r.env) {
            bool found = false;
            for (const md::Item& it : md::items()) found = found || (k == it.env && v == it.on);
            check(found, r.app + ": receipt env " + k + "=" + v + " is an item's default");
        }
    }
}

void test_off_switches() {
    const EnvMap none;
    const auto on_lines = lines(md::effective_config(md::Cli{}, lookup(none)));
    for (const md::Item& it : md::items()) {
        const EnvMap off = {{it.env, it.off}};
        const auto r = md::resolve(lookup(off));
        for (const md::Resolved& x : r) {
            if (x.item == &it) check(x.value == it.off && x.source == md::Source::env, std::string(it.env) + " off");
            else check(x.value == x.item->on, std::string(it.env) + " off leaves " + x.item->env + " on");
        }
        const auto off_lines = lines(md::effective_config(md::Cli{}, lookup(off)));
        std::size_t diff = 0;
        for (const auto& l : off_lines) diff += on_lines.count(l) == 0 ? 1 : 0;
        check(diff == 1 && off_lines.count(std::string(it.env) + "=" + it.off) == 1, std::string(it.env) + " off changes one line");
    }
}

void test_baseline() {
    for (const char* v : {"1", "yes"}) {
        const EnvMap base = {{md::kBaselineEnv, v}};
        check(md::baseline(lookup(base)), std::string("baseline on for ") + v);
        for (const md::Resolved& x : md::resolve(lookup(base)))
            check(x.value == x.item->off && x.source == md::Source::baseline, std::string("baseline: ") + x.item->env + " off");
    }
    for (const char* v : {"0", ""}) {
        const EnvMap base = {{md::kBaselineEnv, v}};
        check(!md::baseline(lookup(base)), std::string("baseline off for '") + v + "'");
        for (const md::Resolved& x : md::resolve(lookup(base))) check(x.value == x.item->on, "baseline=0 keeps defaults");
    }
    // An explicit item value wins over the master switch.
    const md::Item& first = md::items().front();
    const EnvMap mixed = {{md::kBaselineEnv, "1"}, {first.env, first.on}};
    for (const md::Resolved& x : md::resolve(lookup(mixed)))
        check(x.item == &first ? x.value == first.on : x.value == x.item->off, std::string("baseline + explicit ") + x.item->env);
}

void test_cli() {
    const auto r = md::parse_cli("rank", {"M", "1", "tcp-listen:9000", "OUT", "--slots", "2", "--capacity", "100", "--max-rows", "16"});
    check(r.transport == "tcp-listen:9000" && r.slots == 2 && r.capacity == 100 && r.max_rows == 16, "rank cli");
    const auto s = md::parse_cli("serve", {"M", "--transport", "usb4gpu", "--slots", "4"});
    check(s.app == "serve" && s.transport == "usb4gpu" && s.slots == 4 && s.capacity == 8256 && s.max_rows == 4096, "serve cli");
    const auto d = md::parse_cli("serve", {"M"});
    check(d.transport == "usb4-auto" && d.slots == 8, "serve cli defaults = receipt");
}

// Last: changes the process environment.
void test_apply() {
    for (const md::Item& it : md::items()) (void)::unsetenv(it.env);
    (void)::unsetenv(md::kBaselineEnv);
    (void)::setenv("STRIX_MM_XB_PUBLISH", "0", 1);  // a user value
    char flag[] = "--print-effective-config", app[] = "strix-minimax-rank", m[] = "M", r0[] = "0", tr[] = "usb4-auto", out[] = "OUT";
    char* argv[] = {app, m, r0, tr, out, flag};
    check(md::print_or_apply("rank", 6, argv), "print_or_apply with the flag");
    for (const md::Item& it : md::items()) {
        const char* v = std::getenv(it.env);
        if (std::strcmp(it.env, "STRIX_MM_XB_PUBLISH") == 0) {
            check(v != nullptr && std::strcmp(v, "0") == 0, "user value kept");
            check(!md::from_defaults(it.env), "user value is not from the defaults");
        } else {
            check(v != nullptr && std::strcmp(v, it.on) == 0, std::string("apply exported ") + it.env);
            check(md::from_defaults(it.env), std::string("from_defaults ") + it.env);
        }
    }
    char* argv2[] = {app, m, r0, tr, out};
    check(!md::print_or_apply("rank", 5, argv2), "print_or_apply without the flag");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " minimax_production_receipt.txt\n";
        return 2;
    }
    try {
        test_table();
        test_receipts(argv[1]);
        test_off_switches();
        test_baseline();
        test_cli();
        test_apply();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "minimax defaults tests: ALL PASS (" << md::items().size() << " items)\n";
    return 0;
}
