"""Synthetic source fixtures only; the tool never invokes a compiler or Ninja."""
import importlib.util,json,os,tempfile,types,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('provenance',Path(__file__).parents[1]/'tools/qualification/prepare_compile_provenance.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
@unittest.skipUnless(os.name=='posix','Ninja POSIX description only')
class Provenance(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  def file(name,body):
   p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);return p
  self.compiler=file('clang++','synthetic compiler bytes, never executed')
  self.linker=file('ld.lld','synthetic linker bytes, never executed')
  self.src=file('source.cpp','source');self.header=file('header.hpp','header')
  self.obj=file('CMakeFiles/strix-hip-decode-token.dir/source.cpp.o','object');self.binary=file('app','binary')
  file('CMakeCache.txt','CMAKE_GENERATOR:INTERNAL=Ninja\nCMAKE_LINKER:FILEPATH='+str(self.linker)+'\n')
  self.context=file('context.hpp','inline constexpr auto offline_tuning_declared_build_context_hex = "aabb";')
  file('compile_commands.json',json.dumps([{'directory':str(self.root),'file':str(self.src),'arguments':[str(self.compiler),'-c',str(self.src),'-o',str(self.obj)]}]))
  self.target=file('reply/target.json',json.dumps({'name':'strix-hip-decode-token','type':'EXECUTABLE','artifacts':[{'path':str(self.binary)}],'compileGroups':[{'language':'CXX'}],'sources':[{'path':str(self.src),'compileGroupIndex':0}]}))
  self.cm=file('reply/codemodel.json',json.dumps({'version':{'major':2},'paths':{'source':str(self.root),'build':str(self.root)},'configurations':[{'name':'Release','targets':[{'id':'app::id','name':'strix-hip-decode-token','jsonFile':'target.json'}]}]}))
  self.tc=file('reply/toolchains.json',json.dumps({'version':{'major':1},'toolchains':[{'language':'CXX','compiler':{'path':str(self.compiler)}}]}))
  self.deps=file('ninja-deps.txt',f'{self.obj}: #deps 2, deps mtime {self.obj.stat().st_mtime_ns} (VALID)\n    {self.src}\n    {self.header}\n\n')
  self.args=types.SimpleNamespace(build=str(self.root),codemodel=str(self.cm),toolchains=str(self.tc),binary=str(self.binary),build_context_header=str(self.context),ninja_deps_export=str(self.deps),linker=str(self.linker),output=str(self.root/'out.bin'),configuration='Release',target='strix-hip-decode-token')
 def tearDown(self):self.tmp.cleanup()
 def test_complete(self):
  r=module.generate(self.args);self.assertEqual(r['units'],1);self.assertFalse(r['attestation']);self.assertEqual((self.root/'out.bin').read_bytes()[:8],b'STXTC001')
 def test_stale_deps(self):
  self.deps.write_text(self.deps.read_text().replace('(VALID)','(STALE)'))
  with self.assertRaises(module.Incomplete):module.generate(self.args)
  self.assertFalse((self.root/'out.bin').exists())
 def test_changed_object_timestamp(self):
  os.utime(self.obj,ns=(1,1))
  with self.assertRaises(module.Incomplete):module.generate(self.args)
 def test_missing_dependency(self):
  self.header.unlink()
  with self.assertRaises((module.Incomplete,FileNotFoundError)):module.generate(self.args)
 def test_existing_output_preserved(self):
  p=self.root/'out.bin';p.write_bytes(b'preserve')
  with self.assertRaises(module.Incomplete):module.generate(self.args)
  self.assertEqual(p.read_bytes(),b'preserve')
 def test_configuration(self):
  self.args.configuration='Debug'
  with self.assertRaises(module.Incomplete):module.generate(self.args)
 def test_response_not_claimed_complete(self):
  db=self.root/'compile_commands.json';rows=json.loads(db.read_text());rows[0]['arguments'].append('@unknown.rsp');db.write_text(json.dumps(rows))
  with self.assertRaises(module.Incomplete):module.generate(self.args)
if __name__=='__main__':unittest.main()
