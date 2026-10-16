#!/usr/bin/env python3
"""Source-only tool: describe evaluated build inputs; never run a compiler/Ninja.
Ninja POSIX + CMake FileAPI codemodel-v2 ONLY. Explicit valid `ninja -t deps`
export is required. An absent/ambiguous dependency closure produces NO manifest.
This is a current-input provenance description, NOT observed compiler execution
or attestation that a binary was built from those inputs.
"""
import argparse, hashlib, json, os, re, shlex, struct
from pathlib import Path
MAX_FILES=8192; MAX_UNITS=4096; MAX_TARGETS=256; MAX_BYTES=16*1024*1024
class Incomplete(ValueError): pass
def require(ok, reason):
    if not ok: raise Incomplete(reason)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def absolute(p, base):
    return (base/Path(p)).resolve(strict=True) if not Path(p).is_absolute() else Path(p).resolve(strict=True)
def read_json(p, captured=None):
    require(p.stat().st_size<=MAX_BYTES,'metadata size exceeds bound')
    raw=p.read_bytes();require(len(raw)<=MAX_BYTES,'metadata grew beyond bound')
    if captured is not None:
        pin=(len(raw),hashlib.sha256(raw).hexdigest());key=p.resolve(strict=True)
        require(key not in captured or captured[key]==pin,'metadata changed between parses');captured[key]=pin
    return json.loads(raw)
def parse_deps(path, build, captured=None):
    require(path.stat().st_size<=256*1024*1024,'dependency export exceeds bound')
    raw=path.read_bytes();require(len(raw)<=256*1024*1024,'dependency export grew beyond bound')
    if captured is not None:captured[path.resolve(strict=True)]=(len(raw),hashlib.sha256(raw).hexdigest())
    result={}; current=None; expected=0
    for line in raw.decode('utf-8').splitlines():
        if not line:
            if current is not None:
                require(len(result[current])==expected,'dependency count mismatch')
            current=None; continue
        if not line.startswith(' '):
            require(current is None,'missing dependency block separator')
            m=re.fullmatch(r'(.+): #deps ([0-9]+), deps mtime ([0-9]+) \(VALID\)',line)
            require(m is not None,'unsupported/stale Ninja dependency export')
            current=absolute(m[1],build);expected=int(m[2]);require(0<expected<=MAX_FILES,'invalid dependency extent')
            require(current.stat().st_mtime_ns==int(m[3]),'stale object timestamp in dependency export')
            require(current not in result,'duplicate dependency object');result[current]=[]
        else:
            require(current is not None and line.startswith('    '),'invalid dependency row')
            dep=absolute(line[4:],build);require(dep not in result[current],'duplicate dependency path')
            result[current].append(dep)
    if current is not None:require(len(result[current])==expected,'truncated dependency block')
    require(result,'empty dependency export');return result

def generate(args):
    require(os.name=='posix','only Ninja POSIX is supported; leave flags unknown elsewhere')
    require(args.target=='strix-hip-decode-token','only the decoder target is supported')
    parsed={}
    read=lambda p:read_json(p,parsed)
    build=Path(args.build).resolve(strict=True);reply=Path(args.codemodel).resolve(strict=True).parent
    binary=Path(args.binary).resolve(strict=True); db=(build/'compile_commands.json').resolve(strict=True)
    context_header=Path(args.build_context_header).resolve(strict=True);dep_export=Path(args.ninja_deps_export).resolve(strict=True)
    require(not Path(args.output).exists(),'output must be fresh')
    require((build/'CMakeCache.txt').stat().st_size<=MAX_BYTES,'CMake cache exceeds bound')
    cache_raw=(build/'CMakeCache.txt').read_bytes();require(len(cache_raw)<=MAX_BYTES,'CMake cache grew beyond bound');cache=cache_raw.decode('utf-8')
    parsed[(build/'CMakeCache.txt').resolve(strict=True)]=(len(cache_raw),hashlib.sha256(cache_raw).hexdigest())
    require(re.search(r'^CMAKE_GENERATOR:INTERNAL=Ninja$',cache,re.M),'only single-config Ninja is supported')
    cm=read(Path(args.codemodel).resolve(strict=True));require(cm.get('version',{}).get('major')==2,'codemodel-v2 required')
    configs=[c for c in cm['configurations'] if c['name']==args.configuration]
    require(len(configs)==1,'exact configuration unavailable');config=configs[0]
    source=Path(cm['paths']['source']).resolve(strict=True)
    require(Path(cm['paths']['build']).resolve(strict=True)==build,'codemodel build mismatch')
    require(context_header.stat().st_size<=MAX_BYTES,'context header exceeds bound')
    context_raw=context_header.read_bytes();require(len(context_raw)<=MAX_BYTES,'context header grew beyond bound');parsed[context_header]=(len(context_raw),hashlib.sha256(context_raw).hexdigest())
    match=re.findall(r'offline_tuning_declared_build_context_hex\s*=\s*"([0-9a-fA-F]+)"',context_raw.decode('utf-8'))
    require(len(match)==1 and len(match[0])<=65536,'literal declared build context unavailable');context=match[0]
    files={}; snapshots={}
    def add(p,role):
        p=Path(p).resolve(strict=True);require(p.is_file(),'non-file input')
        # The same path has one strongest role. Sources are added before headers.
        if p in files:return
        st=p.stat();require(st.st_size<=256*1024**2,'single declared input exceeds loader bound')
        files[p]=(role,st.st_size,sha(p));snapshots[p]=files[p][1:]
        require(len(files)<=MAX_FILES,'too many input files')
    for p,role in [(binary,10),(db,7),(dep_export,8),(context_header,9),(build/'CMakeCache.txt',6),(Path(args.codemodel),6)]:add(p,role)
    deps=parse_deps(dep_export,build,parsed)
    toolchain=Path(args.toolchains).resolve(strict=True);add(toolchain,6)
    tj=read(toolchain);require(tj.get('version',{}).get('major')==1,'unsupported toolchain reply')
    compilers={tc['language']:absolute(tc['compiler']['path'],build) for tc in tj['toolchains']}
    declared_linker=re.findall(r'^CMAKE_LINKER:FILEPATH=(.+)$',cache,re.M)
    require(len(declared_linker)==1 and absolute(declared_linker[0],build)==Path(args.linker).resolve(strict=True),
            'linker must match declared CMake linker; not an inferred driver subcommand')
    commands=read(db);require(isinstance(commands,list) and len(commands)<=65536,'invalid command database')
    candidates={}
    for row in commands:
        directory=Path(row['directory']).resolve(strict=True)
        argv=row.get('arguments')
        if argv is None:argv=shlex.split(row['command'],posix=True)
        require(isinstance(argv,list) and len(argv)>=2 and all(isinstance(a,str) and a and '\0' not in a for a in argv),'invalid evaluated argv')
        require(not any(a.startswith('@') for a in argv),'response files require explicit expansion support; no key')
        outputs=[argv[i+1] for i,a in enumerate(argv[:-1]) if a=='-o']
        require(len(outputs)==1,'exact -o object output required')
        obj=(directory/Path(outputs[0])).resolve(strict=False)
        if 'output' in row:require((directory/Path(row['output'])).resolve(strict=False)==obj,'database output disagrees with argv')
        src=(directory/Path(row['file'])).resolve(strict=False)
        candidates.setdefault(src,[]).append((obj,directory,argv))
    refs={t['id']:t for t in config['targets']}; roots=[t for t in config['targets'] if t['name']==args.target]
    require(len(roots)==1,'target unavailable or ambiguous')
    target_map={}; ordered=[]; active=set()
    def visit(identifier):
        require(identifier not in active,'cyclic target dependency')
        if identifier in target_map:return
        active.add(identifier)
        require(identifier in refs,'missing dependent target')
        ref=refs[identifier];path=(reply/ref['jsonFile']).resolve(strict=True)
        require(path.parent==reply,'target metadata escapes reply directory');add(path,6)
        t=read(path);require(t['type'] in ('EXECUTABLE','STATIC_LIBRARY','OBJECT_LIBRARY','SHARED_LIBRARY'),'unsupported target type')
        target_map[identifier]=t
        for dep in t.get('dependencies',[]):visit(dep['id'])
        active.remove(identifier);ordered.append(identifier)
    visit(roots[0]['id']);require(len(ordered)<=MAX_TARGETS,'target closure too large')
    targets=[];units=[];names=set()
    for ident in ordered:
        t=target_map[ident];name=t['name'];require(name not in names,'duplicate closure target name');names.add(name)
        artifact=None
        if t['type']!='OBJECT_LIBRARY':
            a=t.get('artifacts',[]);require(len(a)==1,'ambiguous target artifacts')
            artifact=absolute(a[0]['path'],build);add(artifact,5)
            if ident==roots[0]['id']:require(artifact==binary,'executable artifact mismatch')
        fragments=[]
        for section in ('link','archive'):
            fragments += [str(x['role'])+':'+x['fragment'] for x in t.get(section,{}).get('commandFragments',[])]
        # Pin explicit library/object path fragments as declared link inputs.
        # -l/-L search resolution and compiler driver resources are intentionally
        # NOT claimed as a runtime/toolchain closure; their literal argv stays.
        for section in ('link','archive'):
            for item in t.get(section,{}).get('commandFragments',[]):
                for token in shlex.split(item['fragment'],posix=True):
                    for part in token.split(','):
                        if '/' in part and re.search(r'\.(a|o|so(\.[0-9.]+)?)$',part):add(absolute(part,build),5)
        targets.append((name,t['type'],artifact,[target_map[d['id']]['name'] for d in t.get('dependencies',[])],fragments))
        groups=t.get('compileGroups',[])
        for entry in t.get('sources',[]):
            if 'compileGroupIndex' not in entry:continue
            g=groups[entry['compileGroupIndex']];require(g['language'] in ('CXX','C','HIP'),'unsupported compilation language')
            src=absolute(entry['path'],build if entry.get('isGenerated') else source)
            rows=[x for x in candidates.get(src,[]) if '/CMakeFiles/'+name+'.dir/' in str(x[0])]
            require(len(rows)==1,'missing/ambiguous target compilation command')
            obj,directory,argv=rows[0];require(obj in deps,'missing valid explicit object dependency export')
            require(Path(argv[0]).is_absolute(),'relative compiler path is not a closed toolchain identity')
            compiler=Path(argv[0])
            compiler=absolute(compiler,directory)
            require(compilers.get(g['language'])==compiler,'compile argv compiler differs from FileAPI toolchain')
            # A launcher hides the actual compiler argv: do not treat it as closed.
            require(re.search(r'(clang(\+\+)?|g\+\+|gcc|c\+\+|cc)(-[0-9]+)?$',compiler.name),'unsupported launcher/compiler')
            add(compiler,1);add(src,2);add(obj,4)
            unit_deps=list(deps[obj]);
            if src not in unit_deps:unit_deps.insert(0,src)
            units.append((name,src,obj,directory,compiler,argv,unit_deps))
    require(0<len(units)<=MAX_UNITS,'invalid unit closure')
    require(len({x[2] for x in units})==len(units),'duplicate object ownership')
    # Add headers only after ALL sources so a source appearing in another dep
    # list keeps its source role. Every dependency is represented, not a glob.
    for unit in units:
        for dep in unit[-1]:add(dep,3)
    require(any(v[0]==3 for v in files.values()),'no dependency/header closure')
    # Pin every declared compiler and linker; no driver subcommand is claimed.
    for compiler in compilers.values():add(compiler,1)
    add(Path(args.linker).resolve(strict=True),1)
    buf=bytearray(b'STXTC001')
    def u32(v):buf.extend(struct.pack('<I',v))
    def text(v):
        b=str(v).encode('utf-8');require(0<len(b)<=65536 and b'\0' not in b,'invalid string extent');u32(len(b));buf.extend(b)
    text(context);text(files[binary][2]);u32(1);u32(len(files))
    for path,(role,size,digest) in sorted(files.items(),key=lambda x:str(x[0])):
        u32(role);text(path);buf.extend(struct.pack('<Q',size));text(digest)
    text(args.target);u32(len(targets))
    for name,typ,artifact,dependencies,fragments in targets:
        text(name);text(typ)
        if typ!='OBJECT_LIBRARY':text(artifact)
        u32(len(dependencies))
        for d in dependencies:text(d)
        u32(len(fragments))
        for f in fragments:text(f)
    u32(len(units))
    for name,src,obj,directory,compiler,argv,unit_deps in units:
        for v in (name,src,obj,directory,compiler):text(v)
        u32(len(argv))
        for a in argv:text(a)
        u32(len(unit_deps))
        for d in unit_deps:text(d)
    require(len(buf)<=MAX_BYTES,'manifest exceeds bound')
    require(sum(v[1] for v in files.values())<=16*1024**3,'declared input total exceeds loader bound')
    for path,pin in parsed.items():require(path in files and files[path][1:]==pin,'parsed metadata differs from its pinned bytes')
    for path,old in snapshots.items():require((path.stat().st_size,sha(path))==old,'input changed while describing build')
    # Exclusive creation; a write failure leaves no admitted manifest.
    output=Path(args.output)
    import tempfile
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',prefix='.compile-provenance-',dir=output.parent,delete=False) as f:
            temporary=Path(f.name);f.write(buf);f.flush();os.fsync(f.fileno())
        os.link(temporary,output) # atomic publication, refuses an existing final path
    finally:
        if temporary is not None:temporary.unlink()
    return dict(payload_sha256=hashlib.sha256(buf).hexdigest(),files=len(files),targets=len(targets),units=len(units),attestation=False,scope='evaluated-TU-argv-and-declared-inputs-not-transitive-driver-runtime-closure')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build','codemodel','toolchains','binary','build-context-header','ninja-deps-export','linker','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--target',default='strix-hip-decode-token');p.add_argument('--configuration',default='Release')
    print(json.dumps(generate(p.parse_args()),sort_keys=True))
if __name__=='__main__':main()
