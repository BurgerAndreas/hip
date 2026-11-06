"""Production defaults for checkpointed HIP training on a single GPU."""
import os
import runpy
import sys
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ProgressBar
import muon
import fullgraph_muon

muon.MuonWithAuxAdam.step = fullgraph_muon.step

class FinalEpochValidation(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        trainer.check_val_every_n_epoch = 1 if trainer.current_epoch + 1 == trainer.max_epochs else 300

class SparseProgress(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step and trainer.global_step % 1000 == 0:
            print(f"train epoch={trainer.current_epoch} step={trainer.global_step}", flush=True)

OriginalTrainer = pl.Trainer
class ScheduledTrainer(OriginalTrainer):
    def __init__(self, *args, **kwargs):
        callbacks = [cb for cb in (kwargs.get('callbacks') or []) if not isinstance(cb, ProgressBar)]
        kwargs['callbacks'] = [*callbacks, FinalEpochValidation(), SparseProgress()]
        kwargs['enable_progress_bar'] = False
        super().__init__(*args, **kwargs)

if __name__ == '__main__':
    pl.seed_everything(20261102, workers=True)
    pl.Trainer = ScheduledTrainer
    sys.argv.extend(['training.bz_val=125', 'pltrainer.limit_val_batches=8', 'pltrainer.check_val_every_n_epoch=300', 'pltrainer.log_every_n_steps=1000'])
    root = os.environ['HIP_TRAIN_ROOT']
    sys.path.insert(0, root)
    runpy.run_path(os.path.join(root, 'scripts', 'train.py'), run_name='__main__')
