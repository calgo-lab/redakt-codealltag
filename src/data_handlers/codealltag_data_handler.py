from __future__ import annotations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import Dataset, DatasetDict
from pandas import DataFrame
from pathlib import Path
from sklearn.model_selection import KFold
from somajo import SoMaJo
from tqdm import tqdm
from typing import Any, Dict, Generator, List, Set, Tuple, Union
from IPython.display import display

import json
import numpy as np
import pandas as pd
import re


class CodealltagDataHandler:
    """
    Data handler for the CodEAlltag dataset.
    """
    def __init__(self, 
                 project_root: Path, 
                 data_dir: Path = None, 
                 version: str = "20220513",
                 max_worker_threads: int = 4) -> None:
        """
        Initialize the CodealltagDataHandler with the path to the data directory.
        :param project_root: Path to the root of the project.
        :param data_dir: Optional path to the data directory. If None, defaults to 'data' directory in the project root.
        :param version: Version of the dataset to use.
        :param max_worker_threads: Maximum number of worker threads for parallel processing.
        """
        self.project_root: Path = project_root
        
        if data_dir:
            self.data_dir: Path = data_dir
        else:
            self.data_dir: Path = self.project_root / "data"

        self.version: str = version
        self.max_worker_threads: int = max_worker_threads

        self._xl_dir_path: Path = self.data_dir / "raw" / f"version-{self.version}" / "CodEAlltag_pXL"
        self._xl_ann_dir_path: Path = self.data_dir / "raw" / f"version-{self.version}" / "CodEAlltag_pXL_ann"
        self._email_file_ext: str = ".txt"
        self._annotation_file_ext: str = ".ann"
        self._encoding_utf8: str = "utf-8"
        self._hyphen: str = "-"
        self._newline: str = "\n"
        self._whitespace_regex: str = r"\s+"
        self._annotation_dataframe_columns: List[str] = ['file_path', 'Token_ID', 'Label', 'Start', 'End', 'Token']
        self._email_files_info_dataframe_file_path: Path = self.data_dir / f"email_files_info_{self.version}.parquet"
        self._email_files_info_dataframe_columns: List[str] = ['ID', 'file_path_json', 'token_count', 'entity_count', 'label_wise_entity_count', 'file_size']
        self._somajo_tokenizer: SoMaJo = None
        self._sample_dataframe_dir = self.data_dir / "sample"
        self._sample_dataframe_file_path_prefix: str = f"sample_dataframe_{self.version}"

    def get_email_files_info_dataframe(self, force_create: bool = False) -> DataFrame:
        """
        Get the email files info dataframe containing summary stat for each and every email file,
        such as token count, entity count, and label-wise entity count.
        :param force_create: If True, forces the creation of the dataframe even if it already exists.
        :return: DataFrame with email files info.
        """
        if not self._email_files_info_dataframe_file_path.exists() or force_create:
            list(self._create_email_files_info_dataframe())
        
        df = pd.read_parquet(self._email_files_info_dataframe_file_path, engine="pyarrow")
        df = df.set_index("ID")
        return df

    def _create_email_files_info_dataframe(self) -> Generator[Tuple[str, int, int, str, int], None, None]:
        """
        Create the email files info dataframe and save it as a Parquet file.
        Yields tuples containing (file_path_json, token_count, entity_count, label_wise_entity_count, file_size).
        """
        email_files_info_tuples: List[Tuple[str, int, int, str, int]] = list()
        email_files = list(self._xl_dir_path.glob("**/*" + self._email_file_ext))

        with tqdm(total=len(email_files), smoothing=0) as progress_bar:
            with ProcessPoolExecutor(max_workers=self.max_worker_threads) as executor:
                futures = [
                    executor.submit(self._get_email_files_info_tuple_for_email_file, email_file_path) 
                    for email_file_path in email_files
                ]
                for future in as_completed(futures):
                    result = future.result()
                    email_files_info_tuples.append(result)
                    progress_bar.update(1)
                    yield result

        email_files_info_tuples.sort(
            key=lambda t: CodealltagDataHandler._full_natural_sort_key(json.loads(t[0]))
        )
        email_files_info_tuples = [
            (idx, *tup) for idx, tup in enumerate(email_files_info_tuples)
        ]
        df = pd.DataFrame(email_files_info_tuples, columns=self._email_files_info_dataframe_columns)
        df.to_parquet(
            self._email_files_info_dataframe_file_path, 
            index=False, 
            engine="pyarrow", 
            compression="snappy"
        )

    def _get_email_files_info_tuple_for_email_file(self, file_path: Path) -> Tuple[str, int, int, str, int]:
        """
        Get the email files info tuple for a given email file.
        :param file_path: Path to the email file.
        :return: Tuple containing (file_path_json, token_count, entity_count, label_wise_entity_count).
        """
        relative_file_path: Path = file_path.relative_to(self._xl_dir_path)
        relative_file_path_json: str = json.dumps(list(relative_file_path.parts))
        file_path, email_text = self.read_email(relative_file_path)
        annotation_df: DataFrame = self.read_annotations_as_dataframe(relative_file_path)
        token_count: int = len(self.tokenize_with_somajo_and_annotation_dataframe(email_text, annotation_df))
        entity_count: int = len(annotation_df) if not annotation_df.empty else 0
        label_wise_entity_count: dict = dict()
        if not annotation_df.empty:
            label_wise_entity_count = annotation_df['Label'].value_counts().to_dict()
            label_wise_entity_count = dict(sorted(label_wise_entity_count.items()))
        file_size = file_path.stat().st_size
        
        return (
            relative_file_path_json, 
            token_count, 
            entity_count, 
            json.dumps(label_wise_entity_count, ensure_ascii=False),
            file_size
        )
    
    @staticmethod
    def _natural_key(part: str) -> List[Any]:
        """
        Helper function to generate a natural sort key.
        Splits the input string into a list of integers and non-integer substrings.
        :param part: The input string to be split.
        :return: A list containing integers and strings for natural sorting.
        """
        return [
            int(text) if text.isdigit() else text 
            for text in re.split(r'(\d+)', part)
        ]

    @staticmethod
    def _full_natural_sort_key(path_parts: List[str]) -> List[Any]:
        """
        Generate a full natural sort key for a list of path parts.
        :param path_parts: List of strings representing parts of a path.
        :return: A flattened list of natural sort keys for the path parts.
        """
        return [
            item
            for part in path_parts
            for item in CodealltagDataHandler._natural_key(part)
        ]

    def get_absolute_email_file_path(self, file_path: Path) -> Path:
        """
        Get the absolute path to an email file given its relative path.
        :param file_path: Relative path to the email file.
        :return: Absolute path to the email file with the correct file extension.
        """
        return (self._xl_dir_path / file_path).with_suffix(self._email_file_ext)
    
    def read_email(self, file_path: Path, show: bool = False) -> Tuple[Path, str]:
        """
        Read the content of an email file.
        :param file_path: Relative path to the email file.
        :param show: If True, prints the file path and content.
        :return: Tuple containing the absolute file path and the content of the email file.
        """
        file_path = self.get_absolute_email_file_path(file_path)
        return file_path, self._read_file(file_path, show)

    def get_absolute_annotations_file_path(self, file_path: Path) -> Path:
        """
        Get the absolute path to an annotations file given its relative path.
        :param file_path: Relative path to the annotations file.
        :return: Absolute path to the annotations file with the correct file extension.
        """
        return (self._xl_ann_dir_path / file_path).with_suffix(self._annotation_file_ext)
     
    def read_annotations(self, file_path: Path, show: bool = False) -> Tuple[Path, str]:
        """
        Read the content of an annotations file.
        :param file_path: Relative path to the annotations file.
        :param show: If True, prints the file path and content.
        :return: Tuple containing the absolute file path and the content of the annotations file.
        """
        file_path = self.get_absolute_annotations_file_path(file_path)
        return file_path, self._read_file(file_path, show)
    
    def _get_annotation_tuples_for_file(self, file_path: Path, show: bool = False) -> List[Tuple]:
        """
        Get annotation tuples for a given annotations file.
        Each tuple contains (file_path, Token_ID, Label, Start, End, Token).
        :param file_path: Relative path to the annotations file.
        :param show: If True, prints the annotation tuples.
        :return: List of annotation tuples.
        """
        _, content = self.read_annotations(file_path)
        annotation_tuples: List[Any] = list()
        if content is not None:
            lines = content.splitlines()
            for item in lines:
                attributes = re.split(self._whitespace_regex, item)
                annotation_tuples.append((
                    json.dumps(file_path.with_suffix(self._annotation_file_ext).parts),
                    attributes[0],
                    attributes[1],
                    attributes[2],
                    attributes[3],
                    attributes[4] if len(attributes) == 4 else " ".join(attributes[4:len(attributes)])
                ))
        if show:
            print(annotation_tuples)
            print()

        return annotation_tuples

    def read_annotations_as_dataframe(self, file_path: Path, show: bool = False) -> DataFrame:
        """
        Read the annotations of a file and return them as a pandas DataFrame.
        :param file_path: Relative path to the annotations file.
        :param show: If True, displays the DataFrame.
        :return: DataFrame containing the annotations.
        """
        df = pd.DataFrame(
            self._get_annotation_tuples_for_file(file_path),
            columns=self._annotation_dataframe_columns
        )
        if show:
            display(df)
            print()

        return df
    
    def _read_file(self, file_path: Path, show: bool = False) -> str:
        """
        Read the content of a file.
        :param file_path: Path to the file.
        :param show: If True, prints the file path and content.
        :return: Content of the file as a string.
        """
        with file_path.open(mode="r", encoding=self._encoding_utf8) as reader:
            content = reader.read()

        if show:
            print(str(file_path))
            print(self._hyphen * len(str(file_path)) + self._newline)
            print(content)
            print()

        return content

    def _get_somajo_tokenizer(self) -> SoMaJo:
        """
        Get a SoMaJo tokenizer for German text.
        :return: SoMaJo tokenizer instance.
        """
        if self._somajo_tokenizer is None:
            self._somajo_tokenizer = SoMaJo("de_CMC", split_camel_case=False)
        return self._somajo_tokenizer

    def tokenize_with_somajo(self, text: str, label: Union[str, None] = None) -> List[str]:
        """
        Tokenize text using SoMaJo tokenizer.
        If a label is provided, each token is labeled using the BIO scheme.
        :param text: Text to be tokenized.
        :param label: Optional label for the tokens. If provided, tokens are labeled using BIO scheme.
        :return: List of tokens or labeled tokens.
        """
        tokenizer = self._get_somajo_tokenizer()
        tokens = [
            token.text
            for sentence in tokenizer.tokenize_text([text])
            for token in sentence
        ]

        if label and tokens:
            labeled_tokens: List[str] = list()
            if label == "O":
                labeled_tokens = [token + " " + label for token in tokens]
            else:
                labeled_tokens = [tokens[0] + " B-" + label] + [
                    token + " I-" + label for token in tokens[1:]
                ]
            return labeled_tokens

        return tokens

    def tokenize_with_somajo_and_annotation_dataframe(self, 
                                                      email_text: str, 
                                                      annotation_df: DataFrame, 
                                                      labeled: bool = False) -> List[str]:
        """
        Tokenize email text using SoMaJo tokenizer, considering the provided annotation DataFrame.
        :param email_text: The email text to be tokenized.
        :param annotation_df: DataFrame containing the annotations.
        :param labeled: If True, tokens are labeled using the BIO scheme.
        :return: List of tokens or labeled tokens.
        """

        non_entity_texts: List[str] = list()
        entity_texts: List[str] = list()

        text_stretch: str = ""
        tokens_or_labeled_tokens: List[str] = list()
        next_start: int = 0
        for _, row in annotation_df.iterrows():
            if int(row.Start) > next_start:
                text_stretch = email_text[next_start:int(row.Start)]
                non_entity_texts.append(text_stretch)
                tokens_or_labeled_tokens.extend(self.tokenize_with_somajo(text_stretch, 
                                                                          "O" if labeled else None))

            text_stretch = email_text[int(row.Start): int(row.End)]
            entity_texts.append(text_stretch)
            tokens_or_labeled_tokens.extend(self.tokenize_with_somajo(text_stretch, 
                                                                      row.Label if labeled else None))
            next_start = int(row.End)

        if len(email_text) > next_start:
            text_stretch = email_text[next_start: len(email_text)]
            non_entity_texts.append(text_stretch)
            tokens_or_labeled_tokens.extend(self.tokenize_with_somajo(text_stretch, 
                                                                      "O" if labeled else None))

        return tokens_or_labeled_tokens
    
    def _create_sample_dataframe(self, 
                                 sample_size: int = 175_000, 
                                 random_state: int = 2025, 
                                 max_file_size: int = 1024) -> None:
        """
        Create a sample dataframe of email files ensuring balanced representation across categories.
        :param sample_size: The maximum number of files to include in the sample.
        :param random_state: Random state for reproducibility.
        :param max_file_size: Maximum file size (in bytes) to include in the sample.
        :return: None
        """
        email_files_info_df = self.get_email_files_info_dataframe()
        email_files_info_df = email_files_info_df.reset_index()

        email_files_info_df = email_files_info_df[email_files_info_df["file_size"] <= max_file_size]
        email_files_info_df = email_files_info_df[email_files_info_df["entity_count"] > 0]
        
        email_files_info_df["file_path"] = email_files_info_df["file_path_json"].apply(
            lambda s: str(Path(*json.loads(s)))
        )
        email_files_info_df["category"] = email_files_info_df["file_path_json"].apply(
            lambda s: json.loads(s)[0].split("_")[-1]
        )

        categories: List[str] = sorted(email_files_info_df["category"].unique())
        samples_per_category = int(round(sample_size / len(categories)))
        sample_ids: List[int] = list()
        for category in categories:
            df_for_category = email_files_info_df[email_files_info_df["category"] == category]
            if len(df_for_category) >= samples_per_category:
                sample_ids.extend(
                    df_for_category.sample(
                        n=samples_per_category, 
                        random_state=random_state
                    )["ID"].tolist()
                )
            else:
                sample_ids.extend(df_for_category["ID"].tolist())

        sample_df = email_files_info_df[email_files_info_df["ID"].isin(sample_ids)]
        sample_df = sample_df[[
            "ID", 
            "file_path", 
            "category", 
            "token_count", 
            "entity_count", 
            "label_wise_entity_count", 
            "file_size"
        ]]

        labeled_tokens_list: List[str] = list()
        for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), leave=True, position=0, smoothing=0):
            file_path = Path(row["file_path"])
            email_text = self.read_email(file_path)[1]
            annotation_df = self.read_annotations_as_dataframe(file_path)
            labeled_tokens = self.tokenize_with_somajo_and_annotation_dataframe(email_text, annotation_df, labeled=True)
            labeled_tokens_list.append("\n".join(labeled_tokens))

        sample_df["bio_text"] = labeled_tokens_list

        sample_df = sample_df.sort_values(by="ID").reset_index(drop=True)
        self._sample_dataframe_dir.mkdir(parents=True, exist_ok=True)
        sample_df.to_parquet(
            self._sample_dataframe_dir / f"{self._sample_dataframe_file_path_prefix}_{sample_size}.parquet", 
            engine="pyarrow", 
            compression="snappy"
        )

    def get_sample_dataframe(self, 
                             sample_size: int = 175_000, 
                             random_state: int = 2025, 
                             max_file_size: int = 1024) -> DataFrame:
        
        """
        Get a sample dataframe of email files ensuring balanced representation across categories.
        If the sample dataframe does not exist, it will be created.
        :param sample_size: The maximum number of files to include in the sample.
        :param random_state: Random state for reproducibility.
        :param max_file_size: Maximum file size (in bytes) to include in the sample.
        :return: Sample DataFrame with email files info.
        """
        sample_df_file_path = self._sample_dataframe_dir / f"{self._sample_dataframe_file_path_prefix}_{sample_size}.parquet"
        if not sample_df_file_path.exists():
            self._create_sample_dataframe(
                sample_size=sample_size, 
                random_state=random_state, 
                max_file_size=max_file_size
            )
        return pd.read_parquet(sample_df_file_path, engine="pyarrow")

    def get_train_dev_test_datasetdict(self, 
                                       sample_size: int = 175_000, 
                                       random_state: int = 2025, 
                                       max_file_size: int = 1024, 
                                       k: int = 1) -> DatasetDict:
        
        """
        Retrieve the train, dev, and test dataframes for the specified fold.
        :param sample_size: The maximum number of files to include.
        :param random_state: Random state for reproducibility.
        :param max_file_size: Maximum file size (in bytes) to include.
        :param k: The fold number to retrieve (1-based index).
        :return: A DatasetDict containing the train, dev, and test datasets.
        """

        sample_df = self.get_sample_dataframe(
            sample_size=sample_size, 
            random_state=random_state, 
            max_file_size=max_file_size
        )
        sample_df.reset_index(drop=True, inplace=True)

        fold_tuples = list()
        splits = list(KFold(n_splits=5, shuffle=True, random_state=random_state).split(sample_df.index.to_numpy()))
        train_dev_test_k_folds = self.get_train_dev_test_folds()
        for index, fold in enumerate(train_dev_test_k_folds):
            train_indices = list()
            fold_train_indices = fold[1]
            for fold_train_index in fold_train_indices:
                train_indices += list(splits[fold_train_index][1])
            dev_indices = list()
            fold_dev_indices = fold[2]
            for fold_dev_index in fold_dev_indices:
                dev_indices += list(splits[fold_dev_index][1])
            test_indices = list()
            fold_test_indices = fold[3]
            for fold_test_index in fold_test_indices:
                test_indices += list(splits[fold_test_index][1])
            fold_tuples.append((
                index + 1,
                sample_df[sample_df.index.isin(train_indices)].ID.tolist(),
                sample_df[sample_df.index.isin(dev_indices)].ID.tolist(),
                sample_df[sample_df.index.isin(test_indices)].ID.tolist()
            ))
        
        kth_tuple = fold_tuples[k-1]

        train_df = sample_df[sample_df.ID.isin(kth_tuple[1])]
        train_df = train_df.sort_values(by="ID").reset_index(drop=True)
        train_ds = Dataset.from_pandas(train_df)

        dev_df = sample_df[sample_df.ID.isin(kth_tuple[2])]
        dev_df = dev_df.sort_values(by="ID").reset_index(drop=True)
        dev_ds = Dataset.from_pandas(dev_df)

        test_df = sample_df[sample_df.ID.isin(kth_tuple[3])]
        test_df = test_df.sort_values(by="ID").reset_index(drop=True)
        test_ds = Dataset.from_pandas(test_df)

        return DatasetDict({
            "train": train_ds, 
            "dev": dev_ds, 
            "test": test_ds
        })
        
    
    def get_fold_stats(self, 
                       fold_datasetdict: DatasetDict, 
                       label_order: List[str]) -> Dict[str, str]:
        """
        Given a DatasetDict with 'train', 'dev', 'test' splits,
        returns a dict with total files, tokens, entities,
        and per-label entity counts for each split.
        :param fold_datasetdict: The DatasetDict containing 'train', 'dev', 'test' datasets.
        :param label_order: The order of labels to display in the stats.
        :return: A dictionary with stats as keys and formatted strings as values
        """
        train_df = fold_datasetdict["train"].to_pandas()
        dev_df = fold_datasetdict["dev"].to_pandas()
        test_df = fold_datasetdict["test"].to_pandas()

        stats: Dict[str, str] = dict()
        stats["total_files"] = {
            "train": len(train_df['ID'].unique()),
            "dev": len(dev_df['ID'].unique()),
            "test": len(test_df['ID'].unique())
        }

        stats["total_tokens"] = {
            "train": sum(train_df['token_count']),
            "dev": sum(dev_df['token_count']),
            "test": sum(test_df['token_count'])
        }
        stats["total_entities"] = {
            "train": sum(train_df['entity_count']),
            "dev": sum(dev_df['entity_count']),
            "test": sum(test_df['entity_count'])
        }

        stats["train_files"] = train_df['file_path'].unique().tolist()
        stats["dev_files"] = dev_df['file_path'].unique().tolist()
        stats["test_files"] = test_df['file_path'].unique().tolist()

        train_counts = self._aggregate_label_counts(train_df)
        dev_counts = self._aggregate_label_counts(dev_df)
        test_counts = self._aggregate_label_counts(test_df)

        for label in label_order:
            train_val = train_counts.get(label, 0)
            dev_val = dev_counts.get(label, 0)
            test_val = test_counts.get(label, 0)
            stats[label] = {
                "train": train_val, 
                "dev": dev_val, 
                "test": test_val
            }

        return stats
    
    @staticmethod
    def get_train_dev_test_folds(n_fold: int = 5, 
                                 train_percent: float = 0.6, 
                                 dev_percent: float = 0.2) -> List[Tuple]:
        fold_tuples = list()
        indices = list(range(n_fold))
        train_start = 0
        train_end = int(round(n_fold * train_percent))
        dev_start = train_end
        dev_end = int(round(n_fold * (train_percent + dev_percent)))
        test_start = dev_end
        test_end = n_fold
        for index in indices:
            rolled_indices = np.roll(indices, -index)
            train_indices = list(rolled_indices[train_start: train_end])
            dev_indices = list(rolled_indices[dev_start: dev_end])
            test_indices = list(rolled_indices[test_start: test_end])
            fold_tuples.append((
                index + 1,
                train_indices,
                dev_indices,
                test_indices
            ))
        return fold_tuples
    
    @staticmethod
    def _aggregate_label_counts(input_df: DataFrame, 
                                column_name: str = "label_wise_entity_count"):
        total = Counter()
        for item in input_df[column_name]:
            total.update(json.loads(item))
        return total
    
    def get_annotation_dataframe(self):
        return pd.read_parquet(
            self.data_dir / f"annotation_files_info_{self.version}.parquet", 
            engine="pyarrow"
        )