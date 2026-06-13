from pathlib import Path

from hip.inference_utils import (
    HIP_CHECKPOINT_REPO_ID,
    resolve_checkpoint_path,
)


def test_resolve_checkpoint_path_downloads_known_missing_checkpoint(tmp_path, monkeypatch):
    cached_checkpoint = tmp_path / "cached.ckpt"
    cached_checkpoint.write_bytes(b"checkpoint")
    calls = []

    def fake_hf_hub_download(repo_id, filename):
        calls.append((repo_id, filename))
        return str(cached_checkpoint)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    checkpoint_path = tmp_path / "models" / "hip_v3.ckpt"
    resolved_path = resolve_checkpoint_path(checkpoint_path)

    assert Path(resolved_path) == checkpoint_path
    assert checkpoint_path.read_bytes() == b"checkpoint"
    assert calls == [(HIP_CHECKPOINT_REPO_ID, "ckpt/hip_v3.ckpt")]


def test_resolve_checkpoint_path_ignores_unknown_missing_checkpoint(
    tmp_path, monkeypatch
):
    def fake_hf_hub_download(repo_id, filename):
        raise AssertionError("unexpected download")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    checkpoint_path = tmp_path / "other.ckpt"

    assert Path(resolve_checkpoint_path(checkpoint_path)) == checkpoint_path
    assert not checkpoint_path.exists()


def test_resolve_checkpoint_path_keeps_existing_checkpoint(tmp_path, monkeypatch):
    def fake_hf_hub_download(repo_id, filename):
        raise AssertionError("unexpected download")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    checkpoint_path = tmp_path / "hip_v2.ckpt"
    checkpoint_path.write_bytes(b"local checkpoint")

    assert Path(resolve_checkpoint_path(checkpoint_path)) == checkpoint_path
    assert checkpoint_path.read_bytes() == b"local checkpoint"
