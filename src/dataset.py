"""
dataset.py

Skanuje katalog z danymi audio (podzielony na podfoldery per gatunek),
buduje listę plików z etykietami i track_id, oraz dzieli dane na
train/val/test w sposób stratyfikowany po gatunku.

Zakładana struktura danych:

data/genres/
    blues/
        blues.00000.wav
        blues.00001.wav
        ...
    classical/
        ...
    ...

Gatunek jest wykrywany na podstawie nazwy podkatalogu - nie trzeba
niczego hardkodować, wystarczy dorzucić nowy folder z gatunkiem.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Rozszerzenia plików audio traktowane jako dane wejściowe.
AUDIO_EXTENSIONS = {".wav", ".mp3", ".au"}

# Domyślna lokalizacja danych (zgodna ze strukturą repo).
DEFAULT_DATA_DIR = Path("data/genres/genres_original")


def scan_dataset(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """
    Skanuje `data_dir` i zwraca DataFrame z kolumnami:
        - filepath: ścieżka do pliku audio
        - genre: etykieta gatunku (nazwa podkatalogu)
        - track_id: identyfikator oryginalnego utworu (na razie = nazwa
          pliku bez rozszerzenia). Przyda się, jeśli w przyszłości
          wprowadzimy segmentację na krótsze fragmenty - wszystkie
          fragmenty jednego utworu będą miały ten sam track_id, co
          pozwala uniknąć data leakage przy podziale na splity.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Katalog z danymi nie istnieje: {data_dir}. "
            "Pobierz dataset GTZAN i rozpakuj go zgodnie ze strukturą "
            "opisaną w README."
        )

    rows = []
    for genre_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        genre = genre_dir.name
        for filepath in sorted(genre_dir.iterdir()):
            if filepath.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            rows.append(
                {
                    "filepath": str(filepath),
                    "genre": genre,
                    "track_id": filepath.stem,
                }
            )

    if not rows:
        raise ValueError(
            f"Nie znaleziono żadnych plików audio w {data_dir}. "
            f"Sprawdź strukturę katalogów i rozszerzenia plików "
            f"({sorted(AUDIO_EXTENSIONS)})."
        )

    return pd.DataFrame(rows)


def split_dataset(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Dzieli dane na train/val/test, stratyfikując po gatunku.

    Split odbywa się na poziomie `track_id`, a nie pojedynczych wierszy.
    Przy obecnym podejściu (1 utwór = 1 wiersz, cały 30-sekundowy klip)
    nie ma to jeszcze znaczenia, ale jeśli w przyszłości dodamy
    segmentację utworów na krótsze fragmenty (wiele wierszy per
    track_id), ta funkcja i tak zagwarantuje, że wszystkie fragmenty
    jednego utworu trafią do tego samego splitu - bez tego łatwo
    o wyciek danych (ten sam utwór częściowo w train, częściowo w test).
    """
    # Unikalne utwory + ich gatunek (zakładamy 1 gatunek na track_id).
    tracks = df[["track_id", "genre"]].drop_duplicates(subset="track_id")

    train_tracks, temp_tracks = train_test_split(
        tracks,
        test_size=val_size + test_size,
        stratify=tracks["genre"],
        random_state=random_state,
    )

    # Z "temp" wydzielamy val i test w odpowiedniej proporcji.
    relative_test_size = test_size / (val_size + test_size)
    val_tracks, test_tracks = train_test_split(
        temp_tracks,
        test_size=relative_test_size,
        stratify=temp_tracks["genre"],
        random_state=random_state,
    )

    train_df = df[df["track_id"].isin(train_tracks["track_id"])].reset_index(drop=True)
    val_df = df[df["track_id"].isin(val_tracks["track_id"])].reset_index(drop=True)
    test_df = df[df["track_id"].isin(test_tracks["track_id"])].reset_index(drop=True)

    return train_df, val_df, test_df


def summarize(df: pd.DataFrame, name: str = "dataset") -> None:
    """Krótkie podsumowanie liczności per gatunek - do szybkiej weryfikacji."""
    print(f"\n{name}: {len(df)} plików")
    print(df["genre"].value_counts().sort_index())


if __name__ == "__main__":
    full_df = scan_dataset()
    summarize(full_df, "cały dataset")

    train_df, val_df, test_df = split_dataset(full_df)
    summarize(train_df, "train")
    summarize(val_df, "val")
    summarize(test_df, "test")
