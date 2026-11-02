import json

from scripts.pretrain_seviri_ssl import main


def test_stop_request_commits_epoch_and_resume_continues(tmp_path):
    output=tmp_path/'run'; stop=tmp_path/'stop';stop.touch()
    args=['--synthetic-smoke','--objective','mvm','--device','cpu','--batch-size','2',
          '--epochs','3','--num-workers','0','--cpu-threads','1','--output-dir',str(output)]
    assert main([*args,'--stop-after-epoch-file',str(stop)])==0
    progress=json.loads((output/'progress.json').read_text())
    assert progress['status']=='paused' and progress['completed_epochs']==1
    assert progress['optimizer_steps']==2
    assert (output/'checkpoint_latest.pt').is_file()
    stop.unlink()
    assert main([*args,'--resume',str(output/'checkpoint_latest.pt')])==0
    progress=json.loads((output/'progress.json').read_text())
    assert progress['status']=='completed' and progress['completed_epochs']==3
    assert progress['optimizer_steps']==6
