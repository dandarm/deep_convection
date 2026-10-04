import json
from pathlib import Path
import sys


def test_prepare_only_policy_prevents_old_40000_training(tmp_path,monkeypatch):
    import scripts.run_seviri_masking_experiments as queue
    calls=[]
    monkeypatch.setattr(queue,'prepare_split',lambda *args:None)
    monkeypatch.setattr(queue,'prepare_manifest',lambda output,root,name,*args,**kwargs:calls.append(('manifest',name)))
    def train(output,root,name,manifest,ratio,epochs):
        calls.append(('train',name))
        destination=output/name
        destination.mkdir()
        # 1% improvement would otherwise trigger the expansion.
        (destination/'best_validation.json').write_text(json.dumps({'epoch':1,'rmse_k':9.9 if ratio==.5 else 10.}))
    monkeypatch.setattr(queue,'train',train)
    (tmp_path/'preflight.json').write_text('{}')
    (tmp_path/'expansion_policy.json').write_text(json.dumps({'mode':'prepare_only'}))
    monkeypatch.setattr(sys,'argv',['queue','--zarr-root',str(tmp_path/'zarr'),'--output-dir',str(tmp_path)])
    queue.main()
    assert [name for kind,name in calls if kind=='train']==['mask50','mask75','mask90']
    assert ('manifest','train40000') not in calls
    assert json.loads((tmp_path/'selection.json').read_text())['expand_to_40000'] is False
    assert json.loads((tmp_path/'queue_status.json').read_text())['stage']=='completed'
