"""Synthetic parser fixtures are NOT hardware evidence and never leave tests."""
import copy
import importlib.util
from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('visibility_evidence',ROOT/'tools/qualification/gpu_visibility_evidence.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def fixtures():
    def report(local,peer,gpu,receiver):
        r=dict(schema='strix-gpu-visibility-v1',test_mode='native_gpu',production_qualified=False,
            campaign='fixture-only',run='fixture-only',kernel='fixture',driver_build='abcd',architecture='gfx1151',
            local=local*32,peer=peer*32,gpu_uuid=gpu*32,runtime_version=1,gpu_driver_version=1,
            epoch=7,seed=0,bytes=4097,iterations=4,concurrency=2,operations_per_direction=1,completed=4,
            elapsed_ns=1000000,producer_delay_ms=1,consumer_delay_ms=1,cleanup='released',error='',transfers=[])
        seen=0
        for seq in range(1,5):
            rx=receiver if seq%2 else not receiver
            result=lambda nonce:dict(epoch=7,sequence=seq,nonce=nonce,checked_bytes=4097,mismatches=0,
                first_mismatch=m.U64_MAX,observed_digest=9,consumed_digest=10)
            r['transfers'].append(dict(sequence=seq,receiver=rx,completed=True,controller_complete=True,
                consumer_credit=True,producer_completed=not rx,transport_ack=not rx,receive_reposted=rx,
                raw_gpu_verified=rx,copied_gpu_verified=rx,negative_control_detected=rx,
                raw_previously_read=rx and seen>0,raw=result(seq*2),copied=result(seq*2+1)))
            seen+=int(rx)
        return r
    return report('1','2','3',False),report('2','1','4',True)
class EvidenceTests(unittest.TestCase):
    def test_paired_coverage(self):
        r=m.pair_reports(*fixtures());self.assertTrue(r['duplex_pairs']);self.assertEqual(r['reused_raw_buffers_read'],2);self.assertFalse(r['production_qualified'])
    def test_reject_mismatch(self):
        a,b=fixtures();b['transfers'][0]['raw']['mismatches']=1
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_reject_cpu_or_fake_mode(self):
        for mode in ('cpu','api_double','simulated',''):
            a,b=fixtures();b['test_mode']=mode
            with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_reject_incomplete_bytes(self):
        a,b=fixtures();b['transfers'][0]['copied']['checked_bytes']=4096
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_reject_replayed_nonce(self):
        a,b=fixtures();b['transfers'][2]['raw']['nonce']=b['transfers'][0]['raw']['nonce']
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_reject_epoch(self):
        a,b=fixtures();b['transfers'][0]['raw']['epoch']=6
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_reject_early_credit(self):
        a,b=fixtures();b['transfers'][0]['negative_control_detected']=False
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_reject_foreign_peer(self):
        a,b=fixtures();b['peer']='5'*32
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_integer_types(self):
        for bad in (True,1.0,'1',None):
            a,b=fixtures();b['epoch']=bad
            with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_ambiguous_json(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json'
            for data in ('{"a":1,"a":2}','{"a":NaN}'):
                p.write_text(data)
                with self.assertRaises(m.EvidenceError):m.load_report(p)
    def test_reject_claimed_qualification(self):
        a,b=fixtures();b['production_qualified']=True
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_cli_contract(self):
        with tempfile.TemporaryDirectory() as d:
            exe=Path(d)/'options'
            subprocess.run([os.environ.get('CXX','g++'),'-std=c++20','-Wall','-Wextra','-Werror','-Wconversion','-Wshadow',
                '-fsanitize=address,undefined',str(ROOT/'tests/gpu_visibility_options_test.cpp'),'-o',str(exe)],check=True)
            subprocess.run([str(exe)],check=True)
if __name__=='__main__':unittest.main(verbosity=2)
