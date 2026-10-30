#!/usr/bin/env python3
"""Verify identical observation anchors on the historical and working source."""
import os, pathlib, shutil, struct, subprocess, sys, tempfile
root=pathlib.Path(__file__).resolve().parents[1]
tool=root/'tools/minimax/session_default_probe.py'
baseline = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('STRIX_MINIMAX_BASE_REVISION') or '05fedd80'
revisions = ['working']
result = subprocess.run(['git', '-C', str(root), 'cat-file', '-e', baseline + '^{commit}'], capture_output=True)
if result.returncode == 0:
    revisions.insert(0, baseline)
else:
    print('SKIP historical anchors: baseline object unavailable')
with tempfile.TemporaryDirectory(prefix='mm-observer-', dir=pathlib.Path.cwd()) as temp:
    tree=pathlib.Path(temp)
    for revision in revisions:
        for relative in ('apps/minimax_rank_main.cpp','apps/minimax_serve_main.cpp',
                         'src/minimax/serve/minimax_control.hpp','src/minimax/serve/minimax_lead.hpp'):
            data=(root/relative).read_bytes() if revision=='working' else subprocess.check_output(['git','-C',str(root),'show',revision+':'+relative])
            (tree/relative).parent.mkdir(parents=True, exist_ok=True)
            (tree/relative).write_bytes(data)
        subprocess.run([sys.executable,str(tool),str(tree),revision],check=True)
        cli=(tree/'apps/minimax_rank_main.cpp').read_text()
        serve=(tree/'apps/minimax_serve_main.cpp').read_text()
        control=(tree/'src/minimax/serve/minimax_control.hpp').read_text()
        lead=(tree/'src/minimax/serve/minimax_lead.hpp').read_text()
        assert cli.count('mm_probe::row(')==2
        assert 'queue.submit(std::move(job))' in serve
        assert serve.count('Lead lead(')==1
        assert serve.index('Lead lead(') < serve.index('queue.submit(std::move(job))')
        if revision=='working':
            assert 'session_runtime ? &session_service : nullptr)' in serve[serve.index('Lead lead('):]
            assert serve.index('ms::SessionService session_service;') < serve.index('Lead lead(')
            assert 'probe requires a live session runtime' in serve
            assert serve.index('probe requires a live session runtime') < serve.index('follow(eng, run, session_runtime.get())')
        assert 'mm_probe::bytes(eng.rank(), "send"' in control
        assert 'mm_probe::bytes(eng.rank(), "receive"' in control
        assert serve.count('mm_probe::startup_bytes(rank_, "startup-send"') == 2
        assert serve.count('mm_probe::startup_bytes(rank_, "startup-receive"') == 2
        assert serve.count('std::make_unique<Borrowed>(*transport, opt.rank)') == 2
        assert serve.index('mm_probe::startup.store(false)') < serve.index('if (!check.empty())')
        if revision == 'working':
            assert serve.index('admit_runtime_pair(') < serve.index('mm_probe::startup.store(false)')
        assert 'mm_probe::bytes(eng_.rank(), "emitted"' in lead
        assert 'if (mm_probe::enabled()) queue_.shutdown();' in lead
        assert 'PROBE_REVISION' not in (tree/'apps/mm_default_probe.hpp').read_text()
    compiler = os.environ.get('CXX') or shutil.which('c++')
    assert compiler, 'C++ compiler required for startup witness check'
    borrowed = serve[serve.index('class Borrowed final'):serve.index('\n};', serve.index('class Borrowed final')) + 3]
    harness = r'''
#include "apps/mm_default_probe.hpp"
#include "src/minimax/serve/minimax_control.hpp"
#include <algorithm>
#include <span>
namespace strix::dist {
struct Transport {
    virtual ~Transport() = default;
    virtual void send(std::span<const std::byte>) = 0;
    virtual void receive(std::span<std::byte>) = 0;
    virtual void exchange(std::span<const std::byte>, std::span<std::byte>) = 0;
    virtual std::string name() const = 0;
    virtual void abort(const std::string&) = 0;
};
}
BORROWED
struct Echo final : strix::dist::Transport {
    void send(std::span<const std::byte>) override {}
    void receive(std::span<std::byte> m) override { std::fill(m.begin(), m.end(), std::byte{}); }
    void exchange(std::span<const std::byte> o, std::span<std::byte> i) override { std::copy(o.begin(), o.end(), i.begin()); }
    std::string name() const override { return "echo"; }
    void abort(const std::string&) override {}
};
struct Engine {
    Borrowed& transport;
    unsigned rank() const { return 0; }
    strix::minimax::engine::PfProgress prefill_progress(bool = false) const { return {}; }
    void abort(const char*) {}
    void exchange_bytes(const void* out, void* in, std::size_t n) {
        transport.exchange({static_cast<const std::byte*>(out), n}, {static_cast<std::byte*>(in), n});
    }
};
int main(int argc, char**) {
    Echo echo; Borrowed transport(echo, 0);
    Engine engine{transport};
    std::array<std::byte, 777> out{}, in{};
    transport.send(std::span(out).first(8));
    transport.receive(std::span(in).first(8));
    (void)strix::minimax::serve::session_hello(engine, argc > 1);
    if (argc > 1) transport.exchange(std::span(out).first(524), std::span(in).first(524));
    mm_probe::startup.store(false);
    transport.exchange(out, in);
    strix::minimax::serve::Control command;
    command.command = 1;
    (void)strix::minimax::serve::exchange(engine, command);
}
'''.replace('BORROWED', borrowed).replace('#include <algorithm>', '#include <algorithm>\n#include <array>')
    (tree/'startup.cpp').write_text(harness)
    executable = tree/'startup-check'
    subprocess.run([compiler, '-std=c++20', '-I' + str(root/'include'), '-I' + str(tree),
                    str(tree/'startup.cpp'), '-o', str(executable)], check=True)
    def witness(tag, extra):
        prefix = tree/tag
        subprocess.run([str(executable)] + extra, env=dict(os.environ, MM_DEFAULT_PROBE=str(prefix)), check=True)
        sent = pathlib.Path(str(prefix) + '.r0.startup-send').read_bytes()
        assert sent == pathlib.Path(str(prefix) + '.r0.startup-receive').read_bytes()
        ordinary = pathlib.Path(str(prefix) + '.r0.send').read_bytes()
        assert len(ordinary) == 512
        assert ordinary == pathlib.Path(str(prefix) + '.r0.receive').read_bytes()
        sizes, at = [], 0
        while at < len(sent):
            size, = struct.unpack_from('=Q', sent, at)
            sizes.append(size); at += 8 + size
        assert at == len(sent)
        return sizes, sent, ordinary
    original_sizes, original, ordinary = witness('ordinary', [])
    extra_sizes, extra, with_sessions = witness('extra', ['vote'])
    assert original_sizes == [8, 512] and extra_sizes == [8, 512, 524]
    assert original != extra, 'startup witness must detect an added non-Control exchange'
    assert ordinary == with_sessions, 'ordinary controls must exclude the session hello vote'
    gate = (root/'tools/minimax/session_cache_gate.sh').read_text()
    assert 'extensions+=(send receive startup-send startup-receive)' in gate
print('PASS observer overlay: available source anchors; ordinary Lead/follow witness path')
