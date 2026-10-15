import json
from dataclasses import asdict
import pandas as pd
import pytest
from emma_gpm.rdt import RDTTeacherConfig
from emma_gpm.rdt_dashboard import domain_catalog_protocol, RDTClipViewer


def test_parallel_catalog_accepts_sequences_but_rejects_different_teacher(tmp_path, monkeypatch):
    root=tmp_path/'source'
    roots=[]
    for i in range(2):
        directory=tmp_path/f'sequence{i}';directory.mkdir();roots.append(directory)
        protocol=dict(root=str(root.resolve()),config=asdict(RDTTeacherConfig()),config_hash=RDTTeacherConfig().config_hash,
                      sources={'pipeline.py':'same'},shape=[763,1762],orientation='north-up west-left',
                      start_date=f'2020-0{i+1}-01',track_id_offset=(i+1)*(1<<32),frame_index_offset=i*100)
        (directory/'protocol.json').write_text(json.dumps(protocol))
    catalog=pd.DataFrame({'label_root':[str(p) for p in roots],'teacher_version':['multispectral_utls_v0']*2,
                          'start_time':['2020-01-01','2020-02-01'],'origin_y':[0,224],'origin_x':[0,448]})
    actual,_=domain_catalog_protocol(root,catalog)
    assert actual==roots
    monkeypatch.setattr(RDTClipViewer,'update_options',lambda self,*args:None)
    viewer=RDTClipViewer(root,catalog)
    calls=[]
    viewer.load=lambda *args,**kwargs:calls.append((args,kwargs))
    viewer.sample.options=[('Second sequence',1)]
    viewer.sample.value=1
    viewer.load_sample()
    assert calls[-1][1]['label_root']==roots[1]
    viewer.close()
    protocol['config_hash']='different'
    (roots[1]/'protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='Incompatible'):domain_catalog_protocol(root,catalog)
    with pytest.raises(ValueError,match='different satellite'):domain_catalog_protocol(tmp_path/'other',catalog)
