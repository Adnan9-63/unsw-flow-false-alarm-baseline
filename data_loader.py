import os
import urllib.request
import pandas as pd

UNSW_MIRRORS = {
    "UNSW_NB15_training-set.csv": [
        "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv",
        "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv",
    ],
    "UNSW_NB15_testing-set.csv": [
        "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_testing-set.csv",
        "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv",
    ],
}


def ensure_dataset(data_dir: str = ".") -> tuple[str, str]:
    paths = {}
    for fname, urls in UNSW_MIRRORS.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            for url in urls:
                try:
                    urllib.request.urlretrieve(url, fpath)
                    break
                except Exception:
                    continue
        paths[fname] = fpath
    return paths["UNSW_NB15_training-set.csv"], paths["UNSW_NB15_testing-set.csv"]


def load_splits(data_dir: str = ".") -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path, test_path = ensure_dataset(data_dir)
    return pd.read_csv(train_path), pd.read_csv(test_path)
