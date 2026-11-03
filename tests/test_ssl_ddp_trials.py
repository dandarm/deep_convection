import json
import pytest

from scripts import run_seviri_ssl_ddp_trials as trials


def test_pilot_batches_preserve_global_and_per_gpu_comparisons():
    plan=trials.pilot_batches(32,104)
    assert [row['global_batch'] for row in plan]==[32,128,416]
    assert [row['batch_per_gpu'] for row in plan]==[8,32,104]
    shared=trials.pilot_batches(32,32)
    assert len(shared)==2
    assert shared[1]['comparisons']==['per_gpu_reference','large']
    with pytest.raises(ValueError,match='capacity'):
        trials.pilot_batches(32,24)


def test_capacity_refines_compiled_memory_and_keeps_headroom(tmp_path,monkeypatch):
    def launch(root,name,options,allow_oom=False):
        out=root/name;out.mkdir(parents=True,exist_ok=True)
        if name.startswith('smoke/'):
            (out/'encoder').mkdir();(out/'encoder/config.json').write_text('{}')
            return out
        batch=int(options[options.index('--batch-size-per-gpu')+1])
        mode=options[options.index('--compile-mode')+1]
        used=4+batch*(.7 if mode=='off' else .5)
        if used>64:return None
        report=dict(gpus=[dict(total_gib=64,peak_reserved_gib=used,process_device_used_gib=used,
                               steady_seconds_per_update=.1)]*4)
        (out/'capacity.json').write_text(json.dumps(report));return out
    monkeypatch.setattr(trials,'launch',launch)
    trials.capacity(tmp_path)
    choice=json.loads((tmp_path/'batch_choice.json').read_text())
    assert choice['large_batch_per_gpu']==104
    assert choice['large_global_batch']==416
    assert choice['memory_limit_fraction']==.92
    assert any(r['status']=='out_of_memory' for r in choice['trials'])


def test_ram_trials_measure_the_pipeline_and_prioritize_largest_safe_batch(tmp_path,monkeypatch):
    from scripts import stage_seviri_ssl_ram
    monkeypatch.setattr(stage_seviri_ssl_ram,'stage',lambda root:None)
    monkeypatch.setenv('SEVIRI_STORAGE_ROOTS','test-placeholder')
    (tmp_path/'data_locations.json').write_text(json.dumps(dict(training='/training',extended='/extended',
        canonical_zarr_root='/original/source',canonical_frame_root='/original/frames')))
    (tmp_path/'batch_choice.json').write_text(json.dumps(dict(large_batch_per_gpu=144)))
    launches=[]
    def launch(root,name,options,allow_oom=False):
        batch=int(options[options.index('--batch-size-per-gpu')+1]);launches.append((name,batch))
        out=root/name;out.mkdir(parents=True)
        if name.startswith('benchmarks/'):
            report=dict(ranks=[dict(peak_reserved_gib=62 if batch==144 else 40,
                process_device_used_gib=62 if batch==144 else 40,total_gib=64)]*4)
            (out/'pipeline_benchmark.json').write_text(json.dumps(report))
        else:(out/'progress.json').write_text(json.dumps(dict(status='completed')))
        return out
    monkeypatch.setattr(trials,'launch',launch)
    assert trials.ram_pilots(tmp_path)==0
    assert launches==[('benchmarks/ram_b8',8),('benchmarks/ram_b32',32),('benchmarks/ram_b144',144),
                      ('benchmarks/ram_b136',136),('pilots/large',136),('pilots/per_gpu_reference',32)]
