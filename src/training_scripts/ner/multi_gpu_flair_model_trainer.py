from flair.data import Dictionary, _len_dataset
from flair.datasets import DataLoader
from flair.distributed_utils import (
    aggregate, 
    is_main_process, 
    validate_corpus_same_each_process
)
from flair.samplers import FlairSampler
from flair.trainers import ModelTrainer
from flair.trainers.plugins import (
    AnnealingPlugin,
    CheckpointPlugin,
    LinearSchedulerPlugin,
    LogFilePlugin,
    LossFilePlugin,
    MetricName,
    MetricRecord,
    Pluggable,
    ReduceTransformerVocabPlugin,
    TrainerPlugin,
    TrainingInterrupt,
    WeightExtractorPlugin
)
from flair.training_utils import (
    EmbeddingStorageMode, 
    identify_dynamic_embeddings, 
    log_line, 
    store_embeddings
)
from inspect import signature
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel
from torch.optim.sgd import SGD
from torch.utils.data import DistributedSampler
from typing import Optional, Union

import contextlib
import flair
import logging
import numpy as np
import time
import torch
import warnings

log = logging.getLogger("flair")

class MultiGpuFlairModelTrainer(ModelTrainer):
    """
    A subclass of :class:`flair.trainers.ModelTrainer` that supports 
    multi-GPU training via PyTorch's DistributedDataParallel.
    """
    
    def __init__(self, *args, find_unused_parameters=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.find_unused_parameters = find_unused_parameters

    def train_custom(
        self,
        base_path: Union[Path, str],
        # training parameters
        learning_rate: float = 0.1,
        decoder_learning_rate: Optional[float] = None,
        mini_batch_size: int = 32,
        eval_batch_size: int = 64,
        mini_batch_chunk_size: Optional[int] = None,
        max_epochs: int = 100,
        optimizer: type[torch.optim.Optimizer] = SGD,
        train_with_dev: bool = False,
        train_with_test: bool = False,
        max_grad_norm: Optional[float] = 5.0,
        reduce_transformer_vocab: bool = False,
        # evaluation and monitoring
        main_evaluation_metric: tuple[str, str] = ("micro avg", "f1-score"),
        monitor_test: bool = False,
        monitor_train_sample: float = 0.0,
        use_final_model_for_eval: bool = False,
        gold_label_dictionary_for_eval: Optional[Dictionary] = None,
        exclude_labels: Optional[list[str]] = None,
        # sampling and shuffling
        sampler: Optional[Union[FlairSampler, type[FlairSampler]]] = None,
        shuffle: bool = True,
        shuffle_first_epoch: bool = True,
        # evaluation and monitoring
        embeddings_storage_mode: EmbeddingStorageMode = "cpu",
        epoch: int = 0,
        # when and what to save
        save_final_model: bool = True,
        save_optimizer_state: bool = False,
        save_model_each_k_epochs: int = 0,
        # logging parameters
        create_file_logs: bool = True,
        create_loss_file: bool = True,
        write_weights: bool = False,
        # acceleration
        use_amp: bool = False,
        multi_gpu: bool = False,
        # plugins
        plugins: Optional[list[TrainerPlugin]] = None,
        **kwargs,
    ) -> dict:
        """Trains any class that implements the :class:`flair.nn.Model` interface.

        Args:
            base_path: Main path to which all output during training is logged and models are saved
            learning_rate: The learning rate of the optimizer
            decoder_learning_rate: Optional, if set, the decoder is trained with a separate learning rate
            mini_batch_size: Size of mini-batches during training
            eval_batch_size: Size of mini-batches during evaluation
            mini_batch_chunk_size: If mini-batches are larger than this number, they get broken down into chunks of
                this size for processing purposes
            max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.
            optimizer: The optimizer to use (typically SGD or Adam)
            train_with_dev: If True, the data from dev split is added to the training data
            train_with_test: If True, the data from test split is added to the training data
            reduce_transformer_vocab (bool): If True, temporary reduce the vocab size to limit ram usage during training.
            main_evaluation_metric: The metric to optimize (often micro-average or macro-average F1-score, or accuracy)
            monitor_test: If True, test data is evaluated at end of each epoch
            monitor_train_sample: Set this to evaluate on a sample of the train data at the end of each epoch.
                If you set an int, it will sample this many sentences to evaluate on. If you set a float, it will sample
                a percentage of data points from train.
            max_grad_norm: If not None, gradients are clipped to this value before an optimizer.step is called.
            use_final_model_for_eval: If True, the final model is used for the final evaluation. If False, the
                model from the best epoch as determined by main_evaluation_metric is used for the final evaluation.
            gold_label_dictionary_for_eval: Set to force evaluation to use a particular label dictionary
            exclude_labels: Optionally define a list of labels to exclude from the evaluation
            sampler: You can pass a data sampler here for special sampling of data.
            shuffle: If True, data is shuffled during training
            shuffle_first_epoch: If True, data is shuffled during the first epoch of training
            embeddings_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),
                'cpu' (embeddings stored on CPU) or 'gpu' (embeddings stored on GPU)
            epoch: The starting epoch (normally 0 but could be higher if you continue training model)
            save_final_model: If True, the final model is saved at the end of training.
            save_optimizer_state: If True, the optimizer state is saved alongside the model
            save_model_each_k_epochs: Each k epochs, a model state will be written out. If set to '5', a model will
                be saved each 5 epochs. Default is 0 which means no model saving.
            create_file_logs: If True, logging output is written to a file
            create_loss_file: If True, a loss file logging output is created
            use_amp: If True, uses the torch automatic mixed precision
            multi_gpu: If True, distributes training across local GPUs
            write_weights: If True, write weights to weights.txt on each batch logging event.
            plugins: Any additional plugins you want to pass to the trainer
            **kwargs: Additional arguments, for instance for the optimizer

        Returns:
            A dictionary with at least the key "test_score" containing the final evaluation score. Some plugins add
            additional information to this dictionary, such as the :class:`flair.trainers.plugins.MetricHistoryPlugin`
        """
        exclude_labels = exclude_labels if exclude_labels is not None else []
        plugins = plugins if plugins is not None else []

        # Create output folder
        base_path = Path(base_path)
        base_path.mkdir(exist_ok=True, parents=True)

        # === START BLOCK: ACTIVATE PLUGINS === #
        # We first activate all optional plugins. These take care of optional functionality such as various
        # logging techniques and checkpointing

        for plugin in plugins:
            plugin.attach_to(self)

        # log file plugin
        if create_file_logs:
            LogFilePlugin(base_path=base_path).attach_to(self)

        # loss file plugin
        if create_loss_file:
            LossFilePlugin(base_path=base_path, epoch=epoch).attach_to(self)

        # plugin for writing weights
        if write_weights:
            WeightExtractorPlugin(base_path=base_path).attach_to(self)

        if reduce_transformer_vocab:
            ReduceTransformerVocabPlugin(base_path=base_path, save_optimizer_state=save_optimizer_state).attach_to(self)

        # plugin for checkpointing
        if save_model_each_k_epochs > 0:
            CheckpointPlugin(
                save_model_each_k_epochs=save_model_each_k_epochs,
                save_optimizer_state=save_optimizer_state,
                base_path=base_path,
            ).attach_to(self)

        # === END BLOCK: ACTIVATE PLUGINS === #

        # derive parameters the function was called with (or defaults)
        local_variables = locals()
        training_parameters = {
            parameter: local_variables[parameter] for parameter in signature(self.train_custom).parameters
        }
        training_parameters.update(kwargs)

        # initialize model card with these parameters
        self.model.model_card = self._initialize_model_card(**training_parameters)

        # -- Store tokenizer used in corpus on the model --
        # Check if the corpus has a specific tokenizer associated with it
        corpus_tokenizer = getattr(self.corpus, "corpus_tokenizer", None)
        if corpus_tokenizer:
            # Add the tokenizer to the model instance
            self.model.tokenizer = corpus_tokenizer
            log.info(f"Storing tokenizer used in training: {corpus_tokenizer}")
        else:
            # Ensure the attribute exists, even if no specific tokenizer was found
            self.model.tokenizer = None
        # -- End tokenizer storage --

        # Prepare training data and get dataset size
        train_data = self._get_train_data(train_with_dev=train_with_dev, train_with_test=train_with_test)
        dataset_size = _len_dataset(train_data)
        parameters = {"dataset_size": dataset_size, **training_parameters}

        # determine what splits (train, dev, test) to evaluate
        evaluation_splits = {}
        if not train_with_dev and self.corpus.dev:
            evaluation_splits["dev"] = self.corpus.dev
        if self.corpus.test and monitor_test:
            evaluation_splits["test"] = self.corpus.test
        if monitor_train_sample > 0.0:
            evaluation_splits["train_sample"] = self._sample_train_split(monitor_train_sample)

        # determine how to determine best model and whether to save it
        determine_best_epoch_using_dev_score = not train_with_dev and self.corpus.dev
        best_epoch_score = 0 if determine_best_epoch_using_dev_score else float("inf")
        save_best_model = not train_with_dev and not use_final_model_for_eval

        # instantiate the optimizer
        kwargs["lr"] = learning_rate
        if decoder_learning_rate:
            params = [
                {
                    "params": [param for name, param in self.model.named_parameters() if "embeddings" not in name],
                    "lr": decoder_learning_rate,
                },
                {
                    "params": [param for name, param in self.model.named_parameters() if "embeddings" in name],
                    "lr": learning_rate,
                },
            ]
            self.optimizer = optimizer(params=params, **kwargs)
            log.info(
                f"Modifying learning rate to {decoder_learning_rate} for the following "
                f"parameters: {[name for name, param in self.model.named_parameters() if 'embeddings' not in name]}"
            )
        else:
            self.optimizer = optimizer(params=self.model.parameters(), **kwargs)

        # load optimizer state if it exists
        optimizer_state_loaded = False
        if hasattr(self.model, "optimizer_state_dict") and self.model.optimizer_state_dict is not None:
            try:
                self.optimizer.load_state_dict(self.model.optimizer_state_dict)
                optimizer_state_loaded = True
            except Exception as e:
                log.warning(f"Found saved optimizer state from previous training but coult not load: {e}")

        # initialize sampler if provided
        if sampler is not None:
            # init with default values if only class is provided
            if isinstance(sampler, type):
                sampler = sampler()
            # set dataset to sample from
            sampler.set_dataset(train_data)
            shuffle = False

        # configure special behavior to use multiple GPUs
        if multi_gpu:
            if not torch.distributed.is_initialized():
                raise RuntimeError("multi_gpu=True can only used inside flair.distributed_utils.launch_distributed()")
            # Guard against each process initializing corpus differently due to e.g. different random seeds
            validate_corpus_same_each_process(self.corpus)
            self.ddp_model = DistributedDataParallel(
                self.model, device_ids=[flair.device.index], find_unused_parameters=self.find_unused_parameters
            )
            log.disabled = not is_main_process()  # Only print logs once
            original_forward = self.model.forward

        # this field stores the names of all dynamic embeddings in the model (determined after first forward pass)
        dynamic_embeddings = None

        # Sanity checks
        assert len(train_data) > 0
        if epoch >= max_epochs:
            log.warning(f"Starting at epoch {epoch + 1}/{max_epochs}. No training will be done.")
        if epoch == 0:
            self.check_for_and_delete_previous_best_models(base_path)

        # Sanity conversion: if flair.device was set as a string, convert to torch.device
        if isinstance(flair.device, str):
            flair.device = torch.device(flair.device)

        # -- AmpPlugin -> wraps with AMP
        # -- AnnealingPlugin -> initialize schedulers (requires instantiated optimizer)
        with contextlib.ExitStack() as context_stack:
            self.context_stack = context_stack
            self.dispatch("after_setup", **parameters)

            scaler = torch.cuda.amp.GradScaler(enabled=use_amp and flair.device.type != "cpu")

            final_eval_info = (
                "model after last epoch (final-model.pt)"
                if use_final_model_for_eval
                else "model from best epoch (best-model.pt)"
            )
            computation_device_info = aggregate(
                flair.device, lambda devices: ", ".join([str(device) for device in devices])
            )

            log_line(log)
            log.info(f'Model: "{self.model}"')
            log_line(log)
            log.info(f"{self.corpus}")
            log_line(log)
            log.info(f"Train:  {len(train_data)} sentences")
            log.info(f"        (train_with_dev={train_with_dev}, train_with_test={train_with_test})")
            log_line(log)
            log.info("Training Params:")
            log.info(f' - optimizer: "{optimizer}" ')
            log.info(
                f' - learning_rate: "{learning_rate}" '
                f'{"(decoder: " + str(decoder_learning_rate) + ")" if decoder_learning_rate else ""}'
            )
            log.info(f' - mini_batch_size: "{mini_batch_size}"')
            log.info(f' - max_epochs: "{max_epochs}"')
            log.info(f' - shuffle: "{shuffle}"')
            if optimizer_state_loaded:
                log_line(log)
                log.info("Optimizer state loaded from from previous training!")
            log_line(log)
            log.info("Plugins:")
            for plugin in plugins:
                log.info(" - " + str(plugin))
            log_line(log)
            log.info(f"Final evaluation on {final_eval_info}")
            log.info(f' - metric: "{main_evaluation_metric}"')
            log_line(log)
            log.info("Computation:")
            log.info(f" - compute on device: {computation_device_info}")
            log.info(f" - embedding storage: {embeddings_storage_mode}")
            log_line(log)
            log.info(f'Model training base path: "{base_path}"')
            log_line(log)

            # At any point you can hit Ctrl + C to break out of training early.
            try:
                total_train_samples = 0
                batch_count = 0

                for epoch in range(epoch + 1, max_epochs + 1):
                    log_line(log)

                    # - SchedulerPlugin -> load state for anneal_with_restarts, batch_growth_annealing, logic for early stopping
                    # - LossFilePlugin -> get the current epoch for loss file logging
                    self.dispatch("before_training_epoch", epoch=epoch)
                    self.model.model_card["training_parameters"]["epoch"] = epoch

                    lr_info, momentum_info = self._get_current_lr_and_momentum(batch_count)

                    # if shuffle_first_epoch==False, the first epoch is not shuffled
                    shuffle_data_this_epoch = shuffle
                    if not shuffle_first_epoch and epoch == 1:
                        shuffle_data_this_epoch = False

                    if multi_gpu:
                        distributed_sampler: DistributedSampler = DistributedSampler(
                            train_data, shuffle=shuffle_data_this_epoch
                        )
                        distributed_sampler.set_epoch(epoch - 1)
                        batch_loader = DataLoader(
                            train_data,
                            batch_size=mini_batch_size,
                            shuffle=False,
                            sampler=distributed_sampler,
                        )
                    else:
                        batch_loader = DataLoader(
                            train_data,
                            batch_size=mini_batch_size,
                            shuffle=shuffle_data_this_epoch,
                            sampler=sampler,
                        )

                    self.model.train()

                    epoch_train_loss: float = 0.0
                    epoch_train_samples: int = 0

                    epoch_start_time = time.time()

                    # log infos on training progress every `log_modulo` batches
                    log_modulo = max(1, int(len(batch_loader) / 10))

                    # process mini-batches
                    for batch_no, batch in enumerate(batch_loader):
                        # zero the gradients on the model and optimizer
                        self.model.zero_grad()
                        self.optimizer.zero_grad()

                        batch_train_loss = 0.0
                        batch_train_samples = 0
                        batch_count += 1

                        batch_kw = {
                            "batch_no": batch_no,
                            "batch": batch,
                            "total_number_of_batches": len(batch_loader),
                            "epoch": epoch,
                            "batch_count": batch_count,
                        }

                        self.dispatch("before_training_batch", **batch_kw)

                        batch_steps = self.get_batch_steps(batch, mini_batch_chunk_size=mini_batch_chunk_size)

                        # forward and backward for batch
                        for batch_step_no, batch_step in enumerate(batch_steps):
                            disable_gradient_sync = multi_gpu and batch_step_no < len(batch_steps) - 1
                            grad_sync = self.ddp_model.no_sync() if disable_gradient_sync else contextlib.nullcontext()
                            with grad_sync:
                                # forward pass
                                with torch.autocast(device_type=flair.device.type, enabled=use_amp):
                                    if multi_gpu:
                                        # We need to __call__ ddp_model() because this triggers hooks that sync gradients.
                                        # But that calls forward rather than forward_loss. So we patch forward to redirect
                                        # to forward_loss. Then undo the patch in case forward_loss itself calls forward.
                                        def wrapped_forward_loss(*args, **kwargs2):
                                            self.model.forward = original_forward
                                            return self.model.forward_loss(*args, **kwargs2)

                                        self.model.forward = wrapped_forward_loss
                                        loss, datapoint_count = self.ddp_model(batch_step)
                                    else:
                                        loss, datapoint_count = self.model.forward_loss(batch_step)

                                batch_train_samples += datapoint_count
                                batch_train_loss += loss.item()

                                self._backward(scaler.scale(loss))

                                # identify dynamic embeddings (always deleted) on first sentence
                                if dynamic_embeddings is None:
                                    dynamic_embeddings = identify_dynamic_embeddings(batch)

                                # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                                store_embeddings(batch_step, embeddings_storage_mode, dynamic_embeddings)

                        self.dispatch("before_training_optimizer_step", **batch_kw)

                        # do the optimizer step
                        if multi_gpu:
                            # DDP averages across processes but we want the sum
                            self._scale_gradients(torch.distributed.get_world_size())
                        scaler.unscale_(self.optimizer)
                        if max_grad_norm is not None:
                            gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        else:
                            gradient_norm = None
                        scale_before = scaler.get_scale()
                        scaler.step(self.optimizer)
                        scaler.update()
                        scale_after = scaler.get_scale()
                        batch_kw["optimizer_was_run"] = scale_before <= scale_after

                        if batch_train_samples > 0:
                            total_train_samples += batch_train_samples
                            train_loss = batch_train_loss / batch_train_samples
                            self._record(MetricRecord.scalar(("train", "batch_loss"), train_loss, batch_count))
                            if gradient_norm is not None:
                                self._record(
                                    MetricRecord.scalar(("train", "gradient_norm"), gradient_norm, batch_count)
                                )

                            epoch_train_loss += batch_train_loss
                            epoch_train_samples += batch_train_samples

                        if (batch_no + 1) % log_modulo == 0:
                            intermittent_loss = (
                                epoch_train_loss / epoch_train_samples
                                if epoch_train_samples > 0
                                else epoch_train_samples / (batch_no + 1)
                            )
                            intermittent_loss = aggregate(intermittent_loss)

                            current_time = time.time()
                            samples_per_second = epoch_train_samples / (current_time - epoch_start_time)
                            samples_per_second = aggregate(samples_per_second, np.sum)

                            lr_info, momentum_info = self._get_current_lr_and_momentum(batch_count)
                            log.info(
                                f"epoch {epoch}"
                                f" - iter {batch_no + 1}/{len(batch_loader)}"
                                f" - loss {intermittent_loss:.8f}"
                                f" - time (sec): {(current_time - epoch_start_time):.2f}"
                                f" - samples/sec: {samples_per_second:.2f}"
                                f"{lr_info}{momentum_info}"
                            )

                        # - SchedulerPlugin -> do the scheduler step if one-cycle or linear decay
                        # - WeightExtractorPlugin -> extracts weights
                        self.dispatch("after_training_batch", **batch_kw)

                    train_loss = epoch_train_loss / epoch_train_samples
                    train_loss = aggregate(train_loss)
                    self._record(MetricRecord.scalar(("train", "loss"), train_loss, epoch))

                    total_train_samples += epoch_train_samples

                    log_line(log)
                    log.info(f"EPOCH {epoch} done: loss {train_loss:.4f}{lr_info}")

                    # - CheckpointPlugin -> executes save_model_each_k_epochs
                    # - SchedulerPlugin -> log bad epochs
                    self.dispatch("after_training_epoch", epoch=epoch)

                    self.model.eval()

                    # Determine if this is the best model or if we need to anneal
                    current_epoch_has_best_model_so_far = False
                    validation_scores: tuple = ()

                    for evaluation_split, evaluation_split_data in evaluation_splits.items():
                        eval_result = self.model.evaluate(
                            evaluation_split_data,
                            out_path=base_path / f"{evaluation_split}.tsv",
                            mini_batch_size=eval_batch_size,
                            exclude_labels=exclude_labels,
                            main_evaluation_metric=main_evaluation_metric,
                            gold_label_dictionary=gold_label_dictionary_for_eval,
                            embedding_storage_mode=embeddings_storage_mode,
                            gold_label_type=self.model.label_type,
                            gold_label_dictionary_for_eval=gold_label_dictionary_for_eval,
                        )

                        # log results
                        log.info(
                            f"{evaluation_split.upper()} : loss {eval_result.loss}"
                            f" - {main_evaluation_metric[1]}"
                            f" ({main_evaluation_metric[0]})"
                            f"  {round(eval_result.main_score, 4)}"
                        )

                        # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                        store_embeddings(evaluation_split_data, embeddings_storage_mode)

                        self._publish_eval_result(eval_result, evaluation_split, global_step=epoch)

                        # use DEV split to determine if this is the best model so far
                        if determine_best_epoch_using_dev_score and evaluation_split == "dev":
                            validation_scores = eval_result.main_score, eval_result.loss

                            if eval_result.main_score > best_epoch_score:
                                current_epoch_has_best_model_so_far = True
                                best_epoch_score = eval_result.main_score

                    # if not using DEV score, determine best model using train loss
                    if not determine_best_epoch_using_dev_score:
                        validation_scores = (train_loss,)

                        if train_loss < best_epoch_score:
                            current_epoch_has_best_model_so_far = True
                            best_epoch_score = train_loss

                    # - LossFilePlugin -> somehow prints all relevant metrics
                    # - AnnealPlugin -> scheduler step
                    self.dispatch(
                        "after_evaluation",
                        epoch=epoch,
                        current_model_is_best=current_epoch_has_best_model_so_far,
                        validation_scores=validation_scores,
                    )

                    if save_best_model and current_epoch_has_best_model_so_far:
                        log.info("saving best model")
                        self._save_model(base_path / "best-model.pt")

                # - SWAPlugin -> restores SGD weights from SWA
                self.dispatch("after_training_loop")

                # if we do not use dev data for model selection, save final model
                if save_final_model:
                    self._save_model(base_path / "final-model.pt")

            except KeyboardInterrupt:
                log_line(log)
                log.info("Exiting from training early.")

                self.dispatch("training_interrupt")  # TODO: no plugin calls this event

                if save_final_model:
                    log.info("Saving model ...")
                    self._save_model(base_path / "final-model.pt")
                log.info("Done.")

            except TrainingInterrupt as exc:
                log_line(log)
                log.info(str(exc))
                log_line(log)
                self.dispatch("training_interrupt")  # TODO: no plugin calls this event

                if save_final_model:
                    log.info("Saving model ...")
                    self._save_model(base_path / "final-model.pt")
                log.info("Done.")

            except Exception:
                self.dispatch("_training_exception")
                raise
            finally:
                # TensorboardLogger -> closes writer
                self.dispatch("_training_finally")

            # test best model if test data is present
            if self.corpus.test and not train_with_test:
                log_line(log)

                self.model.eval()

                if (base_path / "best-model.pt").exists():
                    log.info("Loading model from best epoch ...")
                    self._load_model(base_path / "best-model.pt")
                else:
                    log.info("Testing using last state of model ...")

                test_results = self.model.evaluate(
                    self.corpus.test,
                    gold_label_type=self.model.label_type,
                    mini_batch_size=eval_batch_size,
                    out_path=base_path / "test.tsv",
                    embedding_storage_mode="none",
                    main_evaluation_metric=main_evaluation_metric,
                    gold_label_dictionary=gold_label_dictionary_for_eval,
                    exclude_labels=exclude_labels,
                    return_loss=False,
                )

                log.info(test_results.detailed_results)
                log_line(log)

                # get and return the final test score of best model
                self.return_values["test_score"] = test_results.main_score

            else:
                if (base_path / "best-model.pt").exists():
                    log.info("Loading model from best epoch ...")
                    self._load_model(base_path / "best-model.pt")
                self.return_values["test_score"] = 0
                log.info("Test data not provided setting final score to 0")

        # MetricHistoryPlugin -> stores the loss history in return_values
        self.dispatch("after_training")

        # Store return values, as they will be erased by reset_training_attributes
        return_values = self.return_values

        self.reset_training_attributes()

        return return_values