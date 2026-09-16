"""
features.py

Ekstrakcja cech audio dla pojedynczego pliku oraz budowa tabeli cech
dla całego datasetu (na bazie DataFrame ze scan_dataset()).

Dla każdej cechy czasowej (MFCC, chroma, spectral centroid/bandwidth/
rolloff, zero crossing rate) liczymy dwie statystyki po klatkach:
mean i std. Tempo jest jedną liczbą (BPM) na cały utwór.

Wynik dla jednego pliku to płaski słownik {nazwa_cechy: wartość},
gotowy do wrzucenia do wiersza DataFrame / CSV.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

# Liczba współczynników MFCC - 20 to standardowy wybór dla muzyki
# (więcej niż typowe 13 używane w mowie, bo muzyka ma bogatszą barwę).
N_MFCC = 20

# Sample rate, do którego resamplujemy każdy plik - stały SR jest
# konieczny, żeby cechy różnych plików były porównywalne.
SAMPLE_RATE = 22050


def extract_features(filepath: Path | str, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> dict[str, float]:
    """
    Wczytuje pojedynczy plik audio i liczy zestaw cech: MFCC, chroma,
    spectral centroid, spectral bandwidth, spectral rolloff, zero
    crossing rate (każda jako mean + std po klatkach) oraz tempo (BPM).

    Zwraca płaski słownik {nazwa_cechy: wartość}.
    """
    # librosa.load domyślnie miksuje do mono i resampluje do `sr`.
    y, sr = librosa.load(filepath, sr=sr)

    features: dict[str, float] = {}

    # --- MFCC: barwa / timbre dźwięku ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        features[f"mfcc{i + 1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc{i + 1}_std"] = float(np.std(mfcc[i]))

    # --- Chroma: rozkład energii po 12 klasach wysokości dźwięku ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(chroma.shape[0]):
        features[f"chroma{i + 1}_mean"] = float(np.mean(chroma[i]))
        features[f"chroma{i + 1}_std"] = float(np.std(chroma[i]))

    # --- Spectral centroid: "jasność" dźwięku ---
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
    features["spectral_centroid_std"] = float(np.std(spectral_centroid))

    # --- Spectral bandwidth: szerokość widma wokół centroidu ---
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))

    # --- Spectral rolloff: częstotliwość skupiająca 85% energii ---
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
    features["spectral_rolloff_std"] = float(np.std(spectral_rolloff))

    # --- Zero crossing rate: jak często sygnał przecina zero ---
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # --- Tempo (BPM) ---
    # W nowszych wersjach librosa beat_track potrafi zwrócić tempo
    # jako tablicę (zamiast pojedynczej liczby) - obsługujemy oba przypadki.
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = float(np.atleast_1d(tempo)[0])

    return features


def build_feature_table(df: pd.DataFrame, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> pd.DataFrame:
    """
    Dla każdego wiersza w `df` (kolumny: filepath, genre, track_id)
    liczy cechy audio i zwraca nowy DataFrame:
    [track_id, genre, <wszystkie cechy>].

    Pliki, których nie da się wczytać (uszkodzone / zły format),
    są pomijane z ostrzeżeniem zamiast wywalać cały bieg.
    """
    rows = []
    n_total = len(df)

    for i, record in enumerate(df.itertuples(index=False), start=1):
        print(f"[{i}/{n_total}] {record.filepath}", end="\r")
        try:
            feats = extract_features(record.filepath, sr=sr, n_mfcc=n_mfcc)
        except Exception as exc:  # noqa: BLE001 - celowo szeroki catch, chcemy kontynuować bieg
            warnings.warn(f"Pominięto {record.filepath}: {exc}")
            continue

        feats["track_id"] = record.track_id
        feats["genre"] = record.genre
        rows.append(feats)

    print()  # nowa linia po pasku postępu z \r
    if not rows:
        raise RuntimeError("Nie udało się wyekstrahować cech z żadnego pliku.")

    feature_df = pd.DataFrame(rows)

    # Porządek kolumn: metadata na początku, potem cechy.
    meta_cols = ["track_id", "genre"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]
    return feature_df[meta_cols + feature_cols]


def save_feature_table(feature_df: pd.DataFrame, output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    print(f"Zapisano {len(feature_df)} wierszy do {output_path}")


if __name__ == "__main__":
    from src.dataset import DEFAULT_DATA_DIR, scan_dataset

    full_df = scan_dataset(DEFAULT_DATA_DIR)
    feature_table = build_feature_table(full_df)
    save_feature_table(feature_table, Path("data/features/features.csv"))
