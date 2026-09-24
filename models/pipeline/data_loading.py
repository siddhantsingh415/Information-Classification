import json
import pandas as pd
from typing import Tuple

def load_averitec_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads the AVeriTeC JSON dataset and extracts the claim, speaker, and veracity label.
    Filters out ambiguous classes to maintain binary True/False alignment for the GAT pipeline.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    for item in data:
        label = item.get('label')

        # Map labels to binary classification
        # Skip "Not Enough Evidence" and "Conflicting Evidence/Cherry-picking"
        if label == "Supported":
            binary_label = 1
        elif label == "Refuted":
            binary_label = 0
        else:
            continue

        records.append({
            'text': item.get('claim', ''),
            'speaker': item.get('speaker', ''),
            'label': binary_label
        })

    return pd.DataFrame(records)

def get_train_test_split(df: pd.DataFrame, train_frac: float = 0.8, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the filtered dataframe into training and testing sets."""
    train_df = df.sample(frac=train_frac, random_state=random_state)
    test_df = df.drop(train_df.index)
    return train_df, test_df