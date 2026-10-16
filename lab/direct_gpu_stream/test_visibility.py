#!/usr/bin/env python3
"""Portable regressions for the native visibility integration (no real HIP/DMA).

Compiles the actual new kernels/adapter/probe and modified .hip translation unit
against explicitly marked API doubles. Native HIP compilation remains separate.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('visibility_campaign', ROOT / 'visibility_campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def fixture(role, sequence=1):
    result = dict(schema='strix-direct-gpu-commissioning-v1', production_qualified=False,
        cleanup='released', error='', sequence=sequence, completed_count=1, executed=True,
        gpu_visibility_profile=campaign.PROFILE, gpu_verifier_self_test=True,
        controller_frames_complete=True, consumer_credit=True, connection_epoch=7,
        pool_binding_identity=1, import_generation=1, device_instance_generation=1,
        payload_bytes=4097, campaign_id='synthetic-test', run_id=f's{sequence}', device_identity='test-only',
        gpu_uuid='a'*32 if role=='sender' else 'b'*32, kernel_identity='test-kernel', driver_identity='test-driver',
        driver_build_id='aa', abi_identity='test-abi', physical_link_id='test-link',
        local_endpoint_identity='c'*32 if role=='sender' else 'd'*32,
        peer_endpoint_identity='d'*32 if role=='sender' else 'c'*32,
        observed_payload_sha256=None, role=role, gpu_acquire_complete=True,consumer_complete=True,
        receive_reposted=True, full_payload_verified=True, local_tx_complete=True, transport_ack=True,
        pattern_seed=0, pattern_version=1, model_plan_id=1,request_id=1,generation=1,boundary_id=1,
        shape_fingerprint=101, dtype_id=1, expected_payload_sha256='0'*64,producer_delay_ms=0,consumer_delay_ms=0)
    for name, stage in (('raw_rx_gpu',3),('consumer_gpu',4)):
        nonce=2*sequence+stage-1
        result[name]=dict(executed=role=='receiver',passed=role=='receiver',nonce=nonce,started=nonce,
            finished=nonce ^ campaign.MASK,epoch=7,sequence=sequence,stage=stage,checked_bytes=4097,mismatches=0,first_bad=None)
    return result


class ReceiptTests(unittest.TestCase):
    def test_complete_pair(self):
        result=campaign.compare(fixture('sender'),fixture('receiver'))
        self.assertTrue(result['record_comparison_passed']);self.assertFalse(result['production_qualified'])
    def test_reject_each_gpu_result_mutation(self):
        for key,value in [('executed',False),('passed',False),('nonce',0),('finished',0),('epoch',8),('sequence',2),('stage',4),('checked_bytes',4096),('mismatches',1),('mismatches',False),('first_bad',0)]:
            with self.subTest(key=key,value=value):
                row=fixture('receiver');row['raw_rx_gpu'][key]=value
                with self.assertRaises(ValueError):campaign.compare(fixture('sender'),row)
    def test_reject_each_lifecycle_failure(self):
        for key,value in [('cleanup','retained'),('error','bad'),('executed',False),('controller_frames_complete',False),('consumer_credit',False),('receive_reposted',False),('gpu_acquire_complete',False),('consumer_complete',False),('full_payload_verified',False),('production_qualified',True),('gpu_verifier_self_test',False),('gpu_visibility_profile',None),('observed_payload_sha256','0'*64)]:
            with self.subTest(key=key):
                row=fixture('receiver');row[key]=value
                with self.assertRaises(ValueError):campaign.compare(fixture('sender'),row)
    def test_reject_peer_disagreement(self):
        for key,value in [('connection_epoch',8),('payload_bytes',1),('run_id','other'),('shape_fingerprint',102),('role','sender'),('peer_endpoint_identity','wrong'),('producer_delay_ms',1)]:
            with self.subTest(key=key):
                row=fixture('receiver');row[key]=value
                with self.assertRaises(ValueError):campaign.compare(fixture('sender'),row)
    def test_reject_duplicate_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.json';path.write_text('{"gpu-visibility":true,"gpu-visibility":false}')
            with self.assertRaises(ValueError):campaign.load(path)
    def test_argument_allowlist(self):
        self.assertEqual(campaign.arguments({'gpu-visibility':True,'duplex-pairs':True}),['--experimental-commissioning','--gpu-visibility','--duplex-pairs'])
        for config in ({'gpu-visibility':False},{'gpu-visibility':True,'arbitrary-command':'x'},{'gpu-visibility':True,'epoch':True},{'gpu-visibility':True,'duplex-pairs':'true'}):
            with self.assertRaises(ValueError):campaign.arguments(config)
    def test_complete_session_and_stale_import(self):
        records=[]
        for role in ('sender','receiver'):
            rows=[fixture(role,1),fixture(role,2)]
            for row in rows:row.update(schema='strix-direct-gpu-commissioning-transfer-v2',cleanup='import_held')
            records.append(dict(schema='strix-direct-gpu-commissioning-session-v2',production_qualified=False,cleanup='released',error='',requested_transfers=2,completed_count=2,schedule='sequential',concurrency=1,pool_binding_identity=1,import_generation=1,transfers=rows))
        self.assertEqual(campaign.compare(*records)['transfers'],2)
        mutated=copy.deepcopy(records);mutated[1]['transfers'][1]['import_generation']=2
        with self.assertRaises(ValueError):campaign.compare(*mutated)
        mutated=copy.deepcopy(records);mutated[1]['completed_count']=1
        with self.assertRaises(ValueError):campaign.compare(*mutated)
        mutated=copy.deepcopy(records);mutated[1]['transfers'][1]['raw_rx_gpu']['nonce']=4
        with self.assertRaises(ValueError):campaign.compare(*mutated)

    def observations(self):
        gpu=dict(schema='strix-direct-gpu-device-observation-v1',gpu_architecture='gfx1151',gpu_uuid='a'*32,gpu_index=0,gpu_runtime_version=1,gpu_driver_version=1,
            qualified=False,production_qualified=False,native_ready=False,executed_payload=False,allocation_performed=False,export_performed=False)
        driver=dict(schema='strix-direct-gpu-driver-observation-v1',query_observed=True,status='observed',fd_retained=False,fd_close_requested=True,error='',
            qualified=False,production_qualified=False,native_ready=False,executed_payload=False,canonical_path='/dev/test-only',kernel_identity='test-kernel',driver_build_id='aa',
            query=dict(abi_major=1,abi_minor=0,abi_cookie=0x5354475055303031,frame_bytes=4096,raw_state=0,import_generation=0,pool_binding_identity=0,connection_epoch=0,
                ring_frames=32,device_instance='do-not-reuse',local_endpoint_identity='b'*32,peer_endpoint_identity='c'*32))
        args=SimpleNamespace(epoch=2,request_id=1,iterations=32,payload_bytes=65537,duplex_pairs=False,alternate_directions=False,
            campaign_id='test',run_id='run',role='sender',physical_link='link',seed=0,control_port=39001,producer_delay_ms=0,consumer_delay_ms=0)
        return gpu,driver,args
    def test_prepare_preserves_observations_without_reusing_fd_instance(self):
        config=campaign.prepare_config(*self.observations())
        self.assertEqual(config['local-endpoint'],'b'*32)
        self.assertEqual(config['ring-frames'],32)
        self.assertEqual(config['epoch'],2)
        self.assertNotIn('device_instance',config)
        self.assertNotIn('import_generation',config)
        self.assertEqual(config['control-mode'],'listen')
    def test_prepare_rejects_active_retained_and_incompatible_observations(self):
        for mutation in ('active','retained','arch','epoch','port'):
            gpu,driver,args=self.observations()
            if mutation=='active':driver['query']['import_generation']=1
            elif mutation=='retained':driver['fd_retained']=True
            elif mutation=='arch':gpu['gpu_architecture']='other'
            elif mutation=='epoch':args.epoch=0
            else:args.control_port=0
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):campaign.prepare_config(gpu,driver,args)


class CompiledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='usb4-visibility-tests-');cls.addClassCleanup(cls.temp.cleanup)
        cls.probe=Path(cls.temp.name)/'probe';cls.main_cases=Path(cls.temp.name)/'main-cases'
        compiler=os.environ.get('CXX','c++')
        common=[compiler,'-std=c++20','-g','-Wall','-Wextra','-Wpedantic','-Wconversion','-Wshadow','-Werror',
                '-fno-pie','-no-pie','-I'+str(ROOT/'tests/stubs'),'-I'+str(ROOT)]
        subprocess.run(common+['-fsanitize=address,undefined','-fno-omit-frame-pointer',str(ROOT/'tests/probe_tests.cpp'),'-o',str(cls.probe)],check=True)
        # Full translation-unit syntax/type check, API declarations substituted.
        subprocess.run(common+['-ffunction-sections','-fdata-sections','-Wl,--gc-sections',str(ROOT/'tests/main_cases.cpp'),'-o',str(cls.main_cases)],check=True)
    def execute_probe(self,case):
        result=subprocess.run([str(self.probe),case],capture_output=True,text=True,timeout=30,
                              env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1:abort_on_error=1'})
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
    def test_result_contract(self):self.execute_probe('contract')
    def test_actual_verifier_kernel_bodies(self):self.execute_probe('probe')
    def test_failure_retains_owners(self):self.execute_probe('retention')
    def test_raw_hook_precedes_acquire(self):self.execute_probe('observer')
    def test_actual_main_prewarm_and_negative_control(self):
        subprocess.run([str(self.main_cases),'warm'],check=True,timeout=10)
    def parse(self,extra):
        base=['--experimental-commissioning','--campaign-id','test','--run-id','run','--role','sender','--dma-device','/dev/test-only',
              '--local-endpoint','a'*32,'--peer-endpoint','b'*32,'--physical-link','test-link','--kernel-release','test-kernel',
              '--driver-build-id','aa','--gpu-uuid','c'*32,'--gpu-runtime-version','1','--gpu-driver-version','1',
              '--epoch','1','--request-id','1','--payload-bytes','4097','--control-mode','listen','--control-port','33333']
        return subprocess.run([str(self.main_cases),*base,*extra],capture_output=True,text=True,timeout=5)
    def test_visibility_cli_and_legacy_compatibility(self):
        for extra in ([],['--gpu-visibility'],['--gpu-visibility','--producer-delay-ms','50','--consumer-delay-ms','50'],
                      ['--gpu-visibility','--iterations','2','--payload-sizes','4097,8191'],
                      ['--iterations','2','--payload-sizes','4097,1']):
            result=self.parse(extra);self.assertEqual(result.returncode,0,result.stderr)
    def test_cli_rejects_unimplementable_or_unsafe_requests(self):
        for extra in (['--gpu-visibility','--gpu-visibility'],['--producer-delay-ms','1'],['--gpu-visibility','--consumer-delay-ms','5001'],
                      ['--gpu-visibility','--producer-delay-ms','5000','--consumer-delay-ms','5000'],
                      ['--gpu-visibility','--iterations','2','--payload-sizes','4097,1'],['--gpu-visibility','--queue-depth','8']):
            with self.subTest(args=extra):self.assertNotEqual(self.parse(extra).returncode,0)


if __name__=='__main__':
    unittest.main(verbosity=2)
