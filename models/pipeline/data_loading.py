"""
ISOT Fake News dataset loading and preprocessing.
"""
import pandas as pd


def load_isot_dataset(fake_path: str = 'data/Fake.csv',
                       real_path: str = 'data/True.csv',
                       random_state: int = 42) -> pd.DataFrame:
    """
    Load and preprocess the ISOT Fake News dataset.

    - Concatenates the fake/real CSVs, shuffles, and adds a binary label
      (label_binary == 1 for fake, 0 for real).
    - Strips the Reuters dateline (e.g. "WASHINGTON (Reuters) - ") from real
      articles to prevent the model from trivially learning "has a dateline
      == real" instead of anything about the actual content.
    """
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)
    fake['label'] = 'fake'
    real['label'] = 'real'

    df = pd.concat([fake, real]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    df['label_binary'] = (df['label'] == 'fake').astype(int)

    df['text'] = df['text'].str.replace(
        r'^[A-Z\s,]+\([^)]+\)\s*-\s*', '', regex=True
    ).str.strip()

    return df
