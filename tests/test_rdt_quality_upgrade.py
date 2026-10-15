import hashlib
import importlib.util
import json
from pathlib import Path
import pytest


def test_quality_upgrade_pins_sources_and_rejects_changed_config(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts/build_rdt_pseudolabels.py'
    spec=importlib.util.spec_from_file_location('rdt_builder_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source='src/emma_gpm/rdt/pipeline.py'
    prior=dict(config={'prominence_k':6},shape=[4,4],sources={source:'old'})
    candidate=dict(prior,sources={source:'new'})
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(prior))
    audit=tmp_path/'audit.json'
    data=dict(previous_protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        old_sources=prior['sources'],new_sources=candidate['sources'],
        compatibility_check='previous-good-frame-exact-equality',previous_outputs_unchanged=True,
        bad_pixels_excluded_without_repair=True)
    audit.write_text(json.dumps(data))
    module.accept_quality_mask_upgrade(protocol,candidate,audit)
    with pytest.raises(ValueError,match='new output directory'):
        module.accept_quality_mask_upgrade(protocol,candidate,None)
    changed=dict(candidate,config={'prominence_k':5})
    with pytest.raises(ValueError,match='not compatible'):
        module.accept_quality_mask_upgrade(protocol,changed,audit)
    changed=dict(candidate,sources={source:'unreviewed'})
    with pytest.raises(ValueError,match='not compatible'):
        module.accept_quality_mask_upgrade(protocol,changed,audit)
