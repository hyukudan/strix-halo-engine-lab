#!/usr/bin/env python3
"""Apply identical read-only witnesses to disposable baseline/candidate build trees.
Usage: session_default_probe.py TREE [REVISION]. Build rank and serve targets afterwards.
For source exports, supply the source revision explicitly.
Never apply this overlay to the working source under review.
"""
import pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1]).resolve()
revision = sys.argv[2] if len(sys.argv) > 2 else subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
header = r'''#pragma once
#include <cstdlib>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace mm_probe {
inline bool enabled() { return std::getenv("MM_DEFAULT_PROBE") != nullptr; }
inline std::atomic<bool> startup{true};
inline void bytes(unsigned rank, const char* ext, const void* data, std::size_t n, bool append=true) {
    if (!enabled()) return;
    std::string prefix = std::string(std::getenv("MM_DEFAULT_PROBE")) + ".r" + std::to_string(rank);
    static bool recorded = false;
    if (!recorded) { std::ofstream metadata(prefix+".revision"); metadata << "PROBE_REVISION"; if (!metadata) throw std::runtime_error("probe metadata"); recorded=true; }
    std::string path = prefix + "." + ext;
    std::ofstream f(path, std::ios::binary | (append ? std::ios::app : std::ios::trunc));
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    if (!f) throw std::runtime_error("default-off witness write");
}
inline void startup_bytes(unsigned rank, const char* ext, const void* data, std::size_t n) {
    if (!enabled() || !startup.load()) return;
    const auto size = static_cast<std::uint64_t>(n);
    bytes(rank, ext, &size, sizeof(size));
    bytes(rank, ext, data, n);
}
template<class E> void row(E& eng, unsigned slot, std::uint64_t positions, std::size_t rows=1) {
    if (!enabled()) return;
    auto logits=eng.last_logits(rows);
    bytes(eng.rank(), "f32", logits.data(), logits.size()*4);
    std::vector<unsigned char> state(static_cast<std::size_t>(positions)*76800);
    eng.export_rows(slot, 0, positions, state.data());
    bytes(eng.rank(), "state", state.data(), state.size(), false);
}
}
'''
(root/'apps/mm_default_probe.hpp').write_text(header.replace('PROBE_REVISION', revision))
def patch(name, replacements):
    path=root/name
    source=path.read_text()
    if 'mm_default_probe.hpp' in source: raise RuntimeError('overlay already applied')
    include = 'mm_default_probe.hpp' if name.startswith('apps/') else '../../../apps/mm_default_probe.hpp'
    source='#include "'+include+'"\n'+source
    prefix = suffix = ''
    if name == 'apps/minimax_serve_main.cpp':
        # Observe every transport call from construction through session admission,
        # including peer scalars, hello, votes and any future startup messages.
        source=source.replace('explicit Borrowed(strix::dist::Transport& t) : t_(t) {}',
                              'explicit Borrowed(strix::dist::Transport& t, unsigned rank) : t_(t), rank_(rank) {}')
        source=source.replace('std::make_unique<Borrowed>(*transport)', 'std::make_unique<Borrowed>(*transport, opt.rank)')
        # Reuse the production constructor verbatim, including runtime/tasks.
        # Older baselines construct it later; hoist that same declaration.
        start=source.index('        Lead lead(')
        end=source.index(');\n', start)+2  # the statement end; a constructor argument may hold a lambda body
        declaration=source[start:end]
        source=source[:start]+source[end:]
        if 'tokenizer' in declaration:
            # The prompt callback captures the tokenizer; move its (unchanged) load ahead of the hoisted declaration.
            load='        const auto tokenizer = ms::MinimaxTokenizer::load(opt.model_dir + "/tokenizer.json");\n'
            if source.count(load)!=1: raise RuntimeError('ambiguous/missing observer anchor: tokenizer load')
            source=source.replace(load,'')
            declaration=load+declaration
        source=source.replace('        ms::ObserverInfo oi;', declaration+'\n        ms::ObserverInfo oi;')
        if 'session_runtime' in declaration:
            source=source.replace('        if (opt.rank != 0) {', '''        if (mm_probe::enabled() && std::getenv("MM_DEFAULT_REQUIRE_SESSION") && !session_runtime)
            throw std::runtime_error("probe requires a live session runtime");
        if (opt.rank != 0) {''')
    if name == 'apps/minimax_rank_main.cpp':
        start=source.index('std::vector<std::uint32_t> session(')
        end=source.index('// RTX-prefill', start)
        prefix, source, suffix = source[:start], source[start:end], source[end:]
    for before, after in replacements:
        if source.count(before)!=1: raise RuntimeError('ambiguous/missing observer anchor: '+before)
        source=source.replace(before,after)
    path.write_text(prefix+source+suffix)
patch('apps/minimax_rank_main.cpp', [
 ('    const double ttft = secs(t0, Clock::now());', '    mm_probe::row(eng, 0, prompt.size());\n    const double ttft = secs(t0, Clock::now());'),
 ('        gen.push_back(eng.step({&row, 1})[0]);', '        gen.push_back(eng.step({&row, 1})[0]);\n        mm_probe::row(eng, 0, row.pos+1);'),
 ('    return gen;\n}', '    mm_probe::bytes(eng.rank(), "u32", gen.data(), gen.size()*4);\n    return gen;\n}'),
])
patch('src/minimax/serve/minimax_control.hpp', [
 # Startup has its own complete transport witness. Ordinary controls exclude
 # hello so the explicit session vote does not affect ordinary-request checks.
 ('    eng.exchange_bytes(o.data(), i.data(), o.size());', '    eng.exchange_bytes(o.data(), i.data(), o.size());\n    if (!mm_probe::startup.load()) {\n        mm_probe::bytes(eng.rank(), "send", o.data(), o.size());\n        mm_probe::bytes(eng.rank(), "receive", i.data(), i.size());\n    }'),
])
patch('src/minimax/serve/minimax_lead.hpp', [
 ('            ++r->stats.generated_tokens;', '            ++r->stats.generated_tokens;\n            mm_probe::bytes(eng_.rank(), "emitted", &id, 4);'),
 ('        queue_.finish(r->job, r->stats);', '        queue_.finish(r->job, r->stats);\n        if (mm_probe::enabled()) queue_.shutdown();'),
])
patch('apps/minimax_serve_main.cpp', [
 ('    void send(std::span<const std::byte> m) override { t_.send(m); }',
  '    void send(std::span<const std::byte> m) override { t_.send(m); mm_probe::startup_bytes(rank_, "startup-send", m.data(), m.size()); }'),
 ('    void receive(std::span<std::byte> m) override { t_.receive(m); }',
  '    void receive(std::span<std::byte> m) override { t_.receive(m); mm_probe::startup_bytes(rank_, "startup-receive", m.data(), m.size()); }'),
 ('    void exchange(std::span<const std::byte> o, std::span<std::byte> i) override { t_.exchange(o, i); }',
  '    void exchange(std::span<const std::byte> o, std::span<std::byte> i) override { t_.exchange(o, i); mm_probe::startup_bytes(rank_, "startup-send", o.data(), o.size()); mm_probe::startup_bytes(rank_, "startup-receive", i.data(), i.size()); }'),
 ('    strix::dist::Transport& t_;', '    strix::dist::Transport& t_;\n    unsigned rank_;'),
 ('        if (!check.empty()) {', '        mm_probe::startup.store(false);\n        if (!check.empty()) {'),
 ('        eng.exchange_bytes(out.data(), in.data(), m * 4);', '        eng.exchange_bytes(out.data(), in.data(), m * 4);\n        mm_probe::bytes(eng.rank(), "send", out.data(), m*4);\n        mm_probe::bytes(eng.rank(), "receive", in.data(), m*4);'),
 ('        const bool done = eng.prefill_resume(ms::resume_layers(layers), &last_next);', '        const bool done = eng.prefill_resume(ms::resume_layers(layers), &last_next);\n        if (done && mm_probe::enabled()) { auto lg=eng.last_logits(1); mm_probe::bytes(eng.rank(), "f32", lg.data(), lg.size()*4); mm_probe::bytes(eng.rank(), "u32", &last_next, 4); }'),
 ('        auto g = eng.step(rows);', '        auto g = eng.step(rows);\n        if (mm_probe::enabled()) { mm_probe::row(eng, rows.back().slot, rows.back().pos+1, rows.size()); mm_probe::bytes(eng.rank(), "u32", g.data(), g.size()*4); }'),
 ('        ms::ObserverInfo oi;', r'''        if (mm_probe::enabled()) {
            const char* prompt_path=std::getenv("MM_DEFAULT_PROMPT");
            if (!prompt_path) throw std::runtime_error("MM_DEFAULT_PROMPT required");
            std::ifstream input(prompt_path, std::ios::binary);
            sv::GenerationJob job;
            for (std::uint32_t t; input.read(reinterpret_cast<char*>(&t),4);) job.prompt.push_back(t);
            if (job.prompt.size()!=128) throw std::runtime_error("probe prompt must have 128 ids");
            job.max_tokens=64;
            job.on_tokens=[](std::span<const std::uint32_t>) { return true; };
            (void)queue.submit(std::move(job));
            lead.loop();
            return 0;
        }
        ms::ObserverInfo oi;'''),
])
(root/'mm-default-probe-revision.txt').write_text(revision+'\n')
print('Observer overlay applied to', revision, 'in', root)
