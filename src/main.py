from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from data_handlers.codealltag_data_handler import CodealltagDataHandler
from utils.project_utils import ProjectUtils
from tqdm import tqdm
from typing import List

import json
import pandas as pd


if __name__ == "__main__":
    project_root = ProjectUtils.get_project_root()
    data_handler = CodealltagDataHandler(project_root=project_root, max_worker_threads=8)
    
    ### Get email files info dataframe
    """
    email_files_info_df = data_handler.get_email_files_info_dataframe()
    from IPython.display import display
    display(email_files_info_df[["file_path_json", "token_count", "entity_count"]].sample(10))
    """

    ### Get sample dataframe
    """
    sample_df = data_handler.get_sample_dataframe()
    from IPython.display import display
    display(sample_df[["ID", "file_path", "category", "token_count", "entity_count"]].head(10))
    """

    ### Get aggregated label counts of sample dataframe
    """
    label_counts = data_handler._aggregate_label_counts(sample_df)
    print("Label counts:", label_counts)
    label_order = [label for label, _ in label_counts.most_common()]
    print(label_order)
    # ['MALE', 'FAMILY', 'ORG', 'CITY', 'DATE', 'URL', 'EMAIL', 'FEMALE', 'UFID', 'PHONE', 'USER', 'STREET', 'STREETNO', 'ZIP']
    """
    
    ### Get fold_stats
    """
    label_order = [
        'MALE', 
        'FAMILY', 
        'ORG', 
        'CITY', 
        'DATE', 
        'URL', 
        'EMAIL', 
        'FEMALE', 
        'UFID', 
        'PHONE', 
        'USER', 
        'STREET', 
        'STREETNO', 
        'ZIP'
    ]
    fold_stats = dict()
    for fold in range(1, 6):
        fold_datasetdict = data_handler.get_train_dev_test_datasetdict(k=fold)
        fold_stats[fold] = data_handler.get_fold_stats(fold_datasetdict, label_order)
    print(json.dumps(fold_stats, indent=4, ensure_ascii=False))

    printable_fold_stats = dict()
    for fold, stats in fold_stats.items():
        printable_fold_stats[fold] = dict()
        for stat, value in stats.items():
            if stat == "total_files":
                printable_fold_stats[fold]["Total Files"] = (
                    f"Train: {value['train']}<br>"
                    f"Dev: {value['dev']}<br>"
                    f"Test: {value['test']}"
                )
            elif stat == "total_tokens":
                printable_fold_stats[fold]["Total Tokens"] = (
                    f"Train: {value['train']}<br>"
                    f"Dev: {value['dev']}<br>"
                    f"Test: {value['test']}"
                )
            elif stat == "total_entities":
                printable_fold_stats[fold]["Total Entities"] = (
                    f"Train: {value['train']}<br>"
                    f"Dev: {value['dev']}<br>"
                    f"Test: {value['test']}"
                )

            for label in label_order:
                if stat == label:
                    printable_fold_stats[fold][label] = (
                        f"Train: {value['train']}<br>"
                        f"Dev: {value['dev']}<br>"
                        f"Test: {value['test']}"
                    )
    print(json.dumps(printable_fold_stats, indent=4, ensure_ascii=False))
    """