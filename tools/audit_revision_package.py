"""Verify the frozen revision control, included weights and revision checksum index."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT/'revision_s1_cbam'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest = json.loads((ROOT/'provenance/revision_bundle_checksums.json').read_text(encoding='utf-8'))
    for row in manifest:
        path = ROOT/row['path']
        assert path.is_file() and sha(path) == row['sha256'], row['path']
    with (ROOT/'weights/included_weights.csv').open(encoding='utf-8', newline='') as stream:
        weights = list(csv.DictReader(stream))
    assert len(weights) == 13
    for row in weights:
        path = ROOT/row['package_path']
        assert path.stat().st_size == int(row['bytes']) and sha(path) == row['sha256']
    for name in ['runner.py', 'protocol.json', 'preflight.json']:
        assert sha(REV/'original_extension'/name) == sha(ROOT/'provenance/revision_s1_cbam'/name)
    original_fingerprints=json.loads((ROOT/'provenance/revision_extension_sources.json').read_text(encoding='utf-8'))
    for row in original_fingerprints:
        assert sha(ROOT/row['bundle_path']) == row['sha256'], row['bundle_path']
    assert len(list((REV/'cache').glob('*.json'))) == 25
    assert len(list((REV/'selections').glob('*.json'))) == 5
    assert len(list((REV/'run_arguments').glob('*.yaml'))) == 5
    logs = list((REV/'epoch_logs').glob('*.csv'))
    assert len(logs) == 5
    for path in logs:
        with path.open(encoding='utf-8', newline='') as stream:
            assert len(list(csv.DictReader(stream))) == 200
    complete = json.loads((ROOT/'benchmark_800/complete.json').read_text(encoding='utf-8'))
    assert complete['status'] == 'complete'
    assert complete['actual_shape'] == [800,800]
    assert complete['forward_samples_per_model'] == 3000
    assert complete['pipeline_samples_per_model'] == 96
    protocol = json.loads((ROOT/'benchmark_800/protocol.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'provenance/benchmark_800_original.py') == protocol['script_sha256']
    assert not (ROOT/'external_resource_audit/DTU_annotations').exists()
    report = dict(passed=True, preserved_revision_files_verified=len(manifest),
                  bundled_weight_files=13, core_epoch200_control_weights=10,
                  original_checkpoint_manifest_scope=80, all_original_weights_bundled=False,
                  new_training_runs=5, epochs_per_run=200, new_cached_evaluations=25,
                  actual_desktop_GPU_benchmark=True, external_model_test_performed=False,
                  raw_DTU_annotations_redistributed=False, GPU_used_by_this_audit=False)
    target=ROOT/'outputs/revision_package_audit.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__=='__main__':
    main()
