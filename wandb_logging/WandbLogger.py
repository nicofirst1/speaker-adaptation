import os
from typing import Any, Dict, Optional

import torch
import wandb


class WandbLogger:
    def __init__(
        self,
        opts: Dict = {},
        group: Optional[str] = None,
        run_id: Optional[str] = None,
        train_logging_step: int = 1,
        val_logging_step: int = 1,
        **kwargs,
    ):
        # This callback logs to wandb the interaction as they are stored in the leader process.
        # When interactions are not aggregated in a multigpu run, each process will store
        # its own Dict[str, Any] object in logs. For now, we leave to the user handling this case by
        # subclassing WandbLogger and implementing a custom logic since we do not know a priori
        # what type of data are to be logged.
        self.opts = opts

        if "wandb_dir" not in opts.keys():
            opts["wandb_dir"] = "wandb_out"
        out_dir = opts["wandb_dir"]

        # create wandb dir if not existing
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        wandb.init(
            group=group,
            entity="adaptive-speaker",
            id=run_id,
            dir=out_dir,
            config=opts,
            mode="disabled" if opts["debug"] else "online",
            **kwargs,
        )
        wandb.config.update(opts)
        self.metrics = {}
        self.train_logging_step = train_logging_step
        self.val_logging_step = val_logging_step
        self.epochs = 0
        self.steps = {}

    def on_batch_end(
        self,
        loss: torch.Tensor,
        data_point: Dict[str, Any],
        aux: Dict[str, Any],
        batch_id: int,
        is_train: bool,
    ):
        raise NotImplemented()

    def on_train_end(self, metrics: Dict[str, Any], epoch_id: int):
        self.epochs = epoch_id
        raise NotImplemented()

    def on_eval_end(self, metrics: Dict[str, Any], epoch_id: int):
        raise NotImplemented()

    @staticmethod
    def log_to_wandb(metrics: Dict[str, Any], commit: bool = False, **kwargs):
        wandb.log(metrics, commit=commit, **kwargs)

    def wandb_close(self):
        """close method.

        it ends the current wandb run
        """
        wandb.finish()

    def save_model(self, path2model, model_name, epoch, description=""):
        model_name = f"epoch_{epoch}_{model_name}"
        artifact = wandb.Artifact(model_name, type='model', description=description)
        artifact.add_file(path2model)
        wandb.log_artifact(artifact)




def delete_run(run_to_remove: str):
    """delete_run method.

    Parameters
    ----------
    run_to_remove : str
        "<entity>/<project>/<run_id>"

    Returns
    -------
    None.

    """
    api = wandb.Api()
    run = api.run(run_to_remove)
    run.delete()
