import json

from scripts import run_seviri_ssl_ddp_trials as trials


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
