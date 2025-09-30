from flair.trainers.plugins import BasePlugin, MetricRecord, TrainerPlugin
from typing import Any, Dict, Set

import wandb


class WandbLoggerPlugin(TrainerPlugin):
    """
    A TrainerPlugin that logs training metrics to Weights & Biases (wandb).
    """
    def __init__(self, 
                 entity: str, 
                 project: str, 
                 config: Dict[str, Any] = None, 
                 tracked: Set[str] = None, 
                 reinit: str = "finish_previous"):
        
        """
        Initializes the WandbLoggerPlugin.
        :param entity: The wandb entity (user or team) to log to.
        :param project: The name of the wandb project.
        :param config: Optional configuration dictionary to log to wandb.
        :param tracked: Optional set of metric names to track.
        :param reinit: Strategy for reinitializing wandb runs.
        """
        
        self.entity = entity
        self.project = project
        self.config = dict(config) if config is not None else dict()
        self.tracked = set(tracked) if tracked is not None else None
        self.reinit = reinit
        self._run = None
        self._pluggable = None
        self._hook_handles = []

    @BasePlugin.hook("after_setup")
    def after_setup(self, **kw):
        """
        Initializes the wandb run after the trainer setup.
        """
        self._run = wandb.init(
            entity=self.entity, 
            project=self.project, 
            config=self.config, 
            reinit=self.reinit
        )

    @BasePlugin.hook("metric_recorded")
    def metric_recorded(self, metric: MetricRecord):
        """
        Logs the recorded metric to wandb if it is in the tracked set.
        :param metric: The MetricRecord instance containing the metric details.
        """
        name = metric.joined_name
        value = metric.value
        if (self.tracked and name in self.tracked and not isinstance(value, str)):
            wandb.log({name: float(value)}, step=metric.global_step)

    @BasePlugin.hook("after_training_epoch")
    def after_training_epoch(self, epoch=None):
        """
        Logs the learning rate after each training epoch.
        :param epoch: The current epoch number.
        """
        trainer = self.trainer
        lrs = [group['lr'] for group in trainer.optimizer.param_groups]
        wandb.log({"learning_rate": lrs[0] if len(lrs) == 1 else lrs}, step=epoch)

    @BasePlugin.hook("after_evaluation")
    def after_evaluation(self, epoch=None, current_model_is_best=None, **kw):
        """
        Logs whether the current model is the best model after evaluation.
        :param epoch: The current epoch number.
        :param current_model_is_best: Boolean indicating if the current model is the best.
        """
        if current_model_is_best:
            wandb.log({"best_model_saved": 1}, step=epoch)
        else:
            wandb.log({"best_model_saved": 0}, step=epoch)

    @BasePlugin.hook("after_training")
    def after_training(self, **kw):
        """
        Finalizes the wandb run after training is complete.
        """
        if self._run:
            self._run.finish()
            self._run = None