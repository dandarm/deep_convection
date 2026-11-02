import json
from pathlib import Path

import pytest

from scripts.prepare_seviri_ssl_hpc import digest, relocated_arguments
from scripts.run_seviri_ssl_hpc import build_command
from emma_gpm.ssl_execution import validate_migration


def test_hpc_arguments_preserve_scientific_controls():
    original = dict(zarr_root='/canonical/raw', frame_cache='/canonical/frames',
                    batch_size=32, gradient_accumulation=1, epochs=150, precision='bf16',
                    learning_rate=.00015, mask_ratio=.9, seed=20261024)
    result = relocated_arguments(original, Path('/fast/data'), Path('/work/run'),
                                 Path('/work/runtime.json'), 'base', 'joint')
    assert all(result[k] == v for k, v in original.items())
    assert result['global_stats'] == '/fast/data/dataset/channel_stats.json'
    assert result['model_size'] == 'base' and result['objective'] == 'joint'


def test_resume_revision_requires_registered_physical_storage():
    parent = dict(batch_size=32, loader_batch_size=32, gradient_accumulation=1,
                  num_workers=4, sources={'trainer':'old'}, objective='mvm')
    current = {**parent, 'sources':{'trainer':'new'},
               'input_pipeline':{'metadata_sha256':'frames','storage_mapping_sha256':'locations'}}
    revision = dict(parent_checkpoints={'checkpoint':'file'}, parent_sources=parent['sources'],
                    sources=current['sources'], clip_metadata_sha256='frames',
                    runtime={k:current[k] for k in ('batch_size','loader_batch_size','gradient_accumulation','num_workers')})
    with pytest.raises(ValueError, match='Storage locations'):
        validate_migration(parent,current,'checkpoint',revision)
    revision['storage_mapping_sha256']='locations'
    validate_migration(parent,current,'checkpoint',revision)


def test_hpc_command_verifies_frozen_inputs_and_resumes(tmp_path):
    archive=tmp_path/'code'; archive.mkdir()
    (archive/'trainer.py').write_text('pass\n')
    dataset=tmp_path/'dataset';dataset.mkdir()
    (dataset/'transfer_plan.json').write_text('{}')
    runtime=tmp_path/'runtime.json';runtime.write_text('{}')
    config=tmp_path/'config.json';config.write_text(json.dumps({'cuda_runtime':str(runtime)}))
    mapping=tmp_path/'locations.json';mapping.write_text('{}')
    output=tmp_path/'run';output.mkdir()
    plan=dict(source_archive=str(archive), frozen_files={'trainer.py':digest(archive/'trainer.py')},
              storage_roots=str(mapping),storage_roots_sha256=digest(mapping),dataset=str(dataset),
              transfer_plan_sha256=digest(dataset/'transfer_plan.json'),cuda_runtime_sha256=digest(runtime),
              runs=[dict(config=str(config),config_sha256=digest(config),output=str(output))])
    assert '--resume' not in build_command(plan,0,'python')
    (output/'checkpoint_latest.pt').write_bytes(b'checkpoint')
    assert build_command(plan,0,'python')[-2:] == ['--resume',str(output/'checkpoint_latest.pt')]
    (archive/'trainer.py').write_text('changed\n')
    with pytest.raises(ValueError,match='source changed'):
        build_command(plan,0,'python')
