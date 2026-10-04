import json

from scripts import run_seviri_v2_mask90_scaling as queue


def test_mask90_extension_preserves_baselines_and_runs_only_two_sizes(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'prepare',lambda *args:None)
    (tmp_path/'preflight.json').write_text('{}')
    for name,count in [('mask75',2000),('large4000',4000),('large8000',8000),('mask90',2000)]:
        folder=tmp_path/name;folder.mkdir()
        (folder/'smoke_summary.json').write_text(json.dumps(dict(epochs=150,samples=count,
            model_version='v2',mask_ratio_requested=.9 if name=='mask90' else .75)))
        (folder/'best_validation.json').write_text(json.dumps(dict(epoch=147,rmse_k=7.)))
    original=tmp_path/'dataset_comparison.json';original.write_text('original')
    calls=[];curves=[];states=[]
    def train(output,root,name,count,ratio,epochs):
        calls.append((name,count,ratio,epochs))
        folder=output/name;folder.mkdir()
        (folder/'best_validation.json').write_text(json.dumps(dict(epoch=149,rmse_k=6.)))
    monkeypatch.setattr(queue,'train',train)
    monkeypatch.setattr(queue,'comparison',lambda output,rows:curves.append(list(rows)))
    queue.execute(tmp_path,tmp_path,150,lambda stage,**kwargs:states.append(stage))
    assert calls==[('large4000_mask90',4000,.9,150),('large8000_mask90',8000,.9,150)]
    assert states==['training','training','completed']
    assert len(curves[-1])==6
    assert original.read_text()=='original'
