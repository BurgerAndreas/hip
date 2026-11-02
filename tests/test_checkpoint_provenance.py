import hashlib
from types import SimpleNamespace

from hip.checkpoint_provenance import CheckpointProvenance, dataset_manifest, safe_remote


def test_dataset_manifest_hashes_file(tmp_path):
    data = tmp_path / "validation.lmdb"
    data.write_bytes(b"sample data")
    result = dataset_manifest(data)
    assert result["files"][0]["sha256"] == hashlib.sha256(b"sample data").hexdigest()
    assert result["files"][0]["size_bytes"] == 11


def test_checkpoint_records_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr("hip.checkpoint_provenance.torch.cuda.device_count", lambda: 0)
    config = {"training": {"bz": 16}, "ckpt_model_path": "model.ckpt"}
    callback = CheckpointProvenance(config, tmp_path)
    trainer = SimpleNamespace(is_global_zero=True, world_size=2, accumulate_grad_batches=4)
    module = SimpleNamespace(potential=SimpleNamespace(max_neighbors=1000), wandb_run_id="run")
    callback.on_fit_start(trainer, module)
    checkpoint = {}
    callback.on_save_checkpoint(trainer, module, checkpoint)
    provenance = checkpoint["provenance"]
    assert provenance["run"]["effective_batch_size"] == 128
    assert provenance["graph"]["max_neighbors"] == 1000
    assert provenance["resolved_config"] == config
    assert provenance["saved_at_utc"]


def test_remote_url_drops_credentials():
    assert safe_remote("https://user:secret@example.com/repo.git?token=secret") == "https://example.com/repo.git"
