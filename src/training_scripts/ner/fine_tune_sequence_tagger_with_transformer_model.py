from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from data_handlers.codealltag_data_handler import CodealltagDataHandler
from flair.data import Corpus, Dictionary
from flair.datasets import ColumnCorpus
from flair.distributed_utils import launch_distributed
from flair.embeddings import TokenEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from training_scripts.ner.multi_gpu_flair_model_trainer import MultiGpuFlairModelTrainer
from training_scripts.ner.wandb_logger_plugin import WandbLoggerPlugin
from utils.project_utils import ProjectUtils


import numpy as np
import os
import random
import torch
import warnings

warnings.filterwarnings("ignore", message=r".*torch\.cuda\.amp\.GradScaler.*")
warnings.filterwarnings("ignore", message=r"No device id is provided via `init_process_group`.*")

def fine_tune():

    model_checkpoints_root_dir = os.environ.get("MODEL_CHECKPOINTS_ROOT_DIR", None)
    model_checkpoints_root_dir = Path(model_checkpoints_root_dir) if model_checkpoints_root_dir else Path.home() / "model_checkpoints"

    data_dir = os.environ.get("DATA_DIR", None)
    data_dir = Path(data_dir) if data_dir else None

    data_fold_k_value = os.environ.get("DATA_FOLD_K_VALUE", None)
    data_fold_k_value = int(data_fold_k_value) if data_fold_k_value else 1

    use_multi_gpu = os.environ.get("USE_MULTI_GPU", None)
    use_multi_gpu = int(use_multi_gpu) if use_multi_gpu else 0
    use_multi_gpu = bool(use_multi_gpu) if use_multi_gpu else False

    log_to_wandb = os.environ.get("LOG_TO_WANDB", None)
    log_to_wandb = int(log_to_wandb) if log_to_wandb else 0
    log_to_wandb = bool(log_to_wandb) if log_to_wandb else False

    transformer_model_name = os.environ.get("TRANSFORMER_MODEL_NAME", "google-bert/bert-base-german-cased")
    
    use_context = os.environ.get("USE_CONTEXT", None)
    use_context = int(use_context) if use_context else 1
    if use_context == 0 or use_context == 1:
        use_context = bool(use_context)

    learning_rate = os.environ.get("LEARNING_RATE", None)
    learning_rate = float(learning_rate) if learning_rate else 5e-5
    
    max_epochs = os.environ.get("MAX_EPOCHS", None)
    max_epochs = int(max_epochs) if max_epochs else 35
    
    mini_batch_size = os.environ.get("MINI_BATCH_SIZE", None)
    mini_batch_size = int(mini_batch_size) if mini_batch_size else 4
    
    project_root: Path = ProjectUtils.get_project_root()
    data_handler = CodealltagDataHandler(project_root, data_dir=data_dir)
    datasetdict = data_handler.get_train_dev_test_datasetdict(k=data_fold_k_value)
    
    label_order = ['MALE', 'FAMILY', 'ORG', 'CITY', 'DATE', 'URL', 'EMAIL', 
                   'FEMALE', 'UFID', 'PHONE', 'USER', 'STREET', 'STREETNO', 'ZIP']
    fold_stats = data_handler.get_fold_stats(datasetdict, label_order)

    train_df = datasetdict["train"].to_pandas()
    dev_df = datasetdict["dev"].to_pandas()
    test_df = datasetdict["test"].to_pandas()
    sample_size = len(train_df) + len(dev_df) + len(test_df)

    print(f"model_checkpoints_root_dir: {model_checkpoints_root_dir}")
    print(f"data_dir: {data_dir}")
    print(f"data_fold_k_value: {data_fold_k_value}")
    print(f"use_multi_gpu: {use_multi_gpu}")
    print(f"log_to_wandb: {log_to_wandb}")
    print(f"transformer_model_name: {transformer_model_name}")
    print(f"use_context: {use_context}")
    print(f"learning_rate: {learning_rate:.0e}".replace('e-0', 'e-'))
    print(f"max_epochs: {max_epochs}")
    print(f"mini_batch_size: {mini_batch_size}")
    print(f"sample_size: {sample_size}")

    model_dir_name = transformer_model_name.replace("/", "--").replace("_", "-")
    if use_context:
        model_dir_name += "-flert"

    data_dir_path = model_checkpoints_root_dir / "codealltag" / "ner" / model_dir_name
    data_dir_path = data_dir_path / "additional-embeddings-none"

    if isinstance(use_context, bool):
        if use_context:
            data_dir_path = data_dir_path / "use-context-64"
        else:
            data_dir_path = data_dir_path / "use-context-none"
    elif isinstance(use_context, int):
        data_dir_path = data_dir_path / f"use-context-{use_context}"

    data_dir_path =  data_dir_path / f"sample-size-{sample_size}" / f"data-fold-{data_fold_k_value}"
    data_dir_path.mkdir(parents=True, exist_ok=True)

    
    train_text = train_df.bio_text.str.cat(sep="\n\n")
    dev_text = dev_df.bio_text.str.cat(sep="\n\n")
    test_text = test_df.bio_text.str.cat(sep="\n\n")

    if not Path(data_dir_path / "xtrain.txt").exists():
        with (data_dir_path / "xtrain.txt").open("w", encoding="utf-8") as writer:
            writer.write(train_text)
    if not Path(data_dir_path / "xdev.txt").exists():
        with (data_dir_path / "xdev.txt").open("w", encoding="utf-8") as writer:
            writer.write(dev_text)
    if not Path(data_dir_path / "xtest.txt").exists():
        with (data_dir_path / "xtest.txt").open("w", encoding="utf-8") as writer:
            writer.write(test_text)
    
    if (not Path(data_dir_path / "train.txt").exists() or 
        not Path(data_dir_path / "dev.txt").exists() or 
        not Path(data_dir_path / "test.txt").exists()):
        
        # clean up data files
        for split in ["train.txt", "dev.txt", "test.txt"]:
            file_path = data_dir_path / split
            backup_path = data_dir_path / f"x{split}"

            with open(backup_path, encoding="utf-8") as fin, open(file_path, "w", encoding="utf-8") as fout:
                for line in fin:
                    line = line.strip()
                    if not line:
                        fout.write("\n")
                        continue

                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    token, tag = parts[0], parts[-1]

                    if tag == "O":
                        fout.write(f"{token} {tag}\n")
                        continue

                    if "-" in tag:
                        _, entity = tag.split("-", 1)
                        if entity in label_order:
                            fout.write(f"{token} {tag}\n")
                        else:
                            fout.write(f"{token} O\n")
                    else:
                        fout.write(f"{token} O\n")

    corpus: Corpus = ColumnCorpus(data_dir_path, 
                                  {0: 'text', 1: 'ner'}, 
                                  train_file="train.txt", 
                                  dev_file="dev.txt", 
                                  test_file="test.txt")
    
    label_dict: Dictionary = corpus.make_label_dictionary(label_type="ner")

    model_dir_path = data_dir_path / f"learning-rate-{learning_rate:.0e}".replace('e-0', 'e-')
    model_dir_path = model_dir_path / f"max-epochs-{max_epochs}"
    model_dir_path = model_dir_path / f"mini-batch-size-{mini_batch_size}"
    model_dir_path.mkdir(parents=True, exist_ok=True)

    embeddings: TokenEmbeddings = TransformerWordEmbeddings(
        model=transformer_model_name,
        use_context=use_context,
        fine_tune=True
    )

    tagger: SequenceTagger = SequenceTagger(
        hidden_size = 256,
        embeddings=embeddings,
        tag_dictionary=label_dict,
        tag_type="ner",
        use_rnn=False,
        use_crf=False,
        reproject_embeddings=False
    )
    tagger.label_dictionary.add_unk = True

    if use_multi_gpu:
        trainer = MultiGpuFlairModelTrainer(
            tagger, 
            corpus, 
            find_unused_parameters=False if not transformer_model_name.startswith("xlm-roberta") else True
        )
    else:
        trainer = ModelTrainer(tagger, corpus)
    
    wandb_plugin = WandbLoggerPlugin(
        project = project_root.name,
        config = {
            "transformer_model_name": transformer_model_name, 
            "data_fold": data_fold_k_value, 
            "learning_rate": learning_rate, 
            "max_epochs": max_epochs, 
            "mini_batch_size": mini_batch_size, 
            "use_context": use_context, 
            "sample_size": sample_size, 
            "fold_stats": fold_stats
        },
        tracked = {
            "train/loss", 
            "dev/loss", 
            "dev/micro avg/precision", 
            "dev/micro avg/recall", 
            "dev/micro avg/f1-score", 
            "dev/macro avg/precision", 
            "dev/macro avg/recall", 
            "dev/macro avg/f1-score", 
            "dev/accuracy"
        }
    )

    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2025)

    trainer.fine_tune(
        model_dir_path, 
        learning_rate = learning_rate, 
        max_epochs = max_epochs, 
        mini_batch_size = mini_batch_size, 
        eval_batch_size = mini_batch_size, 
        write_weights = True, 
        save_final_model = False, 
        use_final_model_for_eval = False, 
        shuffle=False, 
        shuffle_first_epoch=False, 
        multi_gpu=use_multi_gpu, 
        use_amp=use_multi_gpu, 
        plugins = [wandb_plugin] if log_to_wandb else None
    )

if __name__ == "__main__":

    use_multi_gpu = os.environ.get("USE_MULTI_GPU", None)
    use_multi_gpu = int(use_multi_gpu) if use_multi_gpu else 0
    use_multi_gpu = bool(use_multi_gpu) if use_multi_gpu else False

    if use_multi_gpu:
        launch_distributed(fine_tune)
    else:
        fine_tune()