"""Resume a HIP training checkpoint to 100 epochs and save final validation."""
import json
import os
from pathlib import Path
import runpy
import sys
import torch
import lightning.pytorch as pl

SMOKE = os.environ.get("HIP_SHORT_SMOKE") == "1"

class FinalShortOutput(pl.Callback):
    def __init__(self):
        self.final_validation_seen = False

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking and trainer.current_epoch + 1 == 100:
            self.final_validation_seen = True
            print(f'SHORT100 final validation completed at epoch {trainer.current_epoch + 1}', flush=True)

    def on_fit_end(self, trainer, pl_module):
        if SMOKE:
            print("SHORT100 smoke test completed", flush=True)
            return
        if trainer.current_epoch < 100 or not self.final_validation_seen:
            raise RuntimeError('Training ended before epoch 100 and its final validation completed')
        out = Path(os.environ['HIP_SHORT_OUTPUT'])
        out.mkdir(parents=True, exist_ok=True)
        checkpoint = out / 'final-epoch-100.ckpt'
        trainer.save_checkpoint(str(checkpoint))
        metrics = {}
        for name, value in trainer.callback_metrics.items():
            if name.startswith('val-'):
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    metrics[name] = value.detach().cpu().item()
                elif isinstance(value, (int, float)):
                    metrics[name] = value
        (out / 'final-validation.json').write_text(json.dumps({'completed_epochs': trainer.current_epoch, 'global_step': trainer.global_step, 'metrics': metrics}, indent=2) + '\n')
        print(f'SHORT100 final checkpoint saved: {checkpoint}; validation metrics: {len(metrics)}', flush=True)

class SmokeStop(pl.Callback):
    def __init__(self): self.steps = 0
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.steps += 1
        if self.steps == 5:
            print('SHORT100 smoke resumed checkpoint and completed five batches', flush=True)
            trainer.fit_loop.epoch_loop.max_steps = trainer.global_step

OriginalTrainer = pl.Trainer
class ShortTrainer(OriginalTrainer):
    def __init__(self, *args, **kwargs):
        kwargs['callbacks'] = [*(kwargs.get('callbacks') or []), FinalShortOutput()]
        if SMOKE:
            kwargs['callbacks'].append(SmokeStop())
        super().__init__(*args, **kwargs)

pl.Trainer = ShortTrainer

def _resume_arg():
    """Resume from the newest last.ckpt under ckpt_base_dir (train.py auto-resume mis-resolves absolute paths)."""
    if SMOKE or any(a.lstrip('+').startswith('ckpt_trainer_path=') for a in sys.argv[1:]):
        return
    base = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('ckpt_base_dir=')), None)
    if not base or not Path(base).is_absolute():
        return
    best = None
    for p in Path(base).glob('*/last.ckpt'):
        try:
            step = torch.load(p, map_location='cpu', mmap=True, weights_only=False)['global_step']
        except Exception as e:
            print(f'SHORT100 resume: skipping unreadable {p}: {e}', flush=True)
            continue
        if best is None or step > best[0]:
            best = (step, p)
    if best:
        print(f'SHORT100 resume from {best[1]} (global_step {best[0]})', flush=True)
        sys.argv.append(f'ckpt_trainer_path={best[1]}')
    else:
        print(f'SHORT100 resume: no last.ckpt under {base}, starting fresh', flush=True)

_resume_arg()
sys.path.insert(0, os.environ['HIP_RUN'])
runpy.run_path(os.path.join(os.environ['HIP_RUN'], 'seeded_train.py'), run_name='__main__')
