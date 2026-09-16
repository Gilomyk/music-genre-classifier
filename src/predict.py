"""
predict.py

CLI do predykcji gatunku dla dowolnego pliku audio (WAV/MP3), na
podstawie wcześniej wytrenowanego modelu (models/model.joblib).

Użycie:
    python -m src.predict path/to/song.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.features import extract_features
from src.train import MODEL_PATH


def load_model(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Brak {path}. Najpierw uruchom: python -m src.train"
        )
    return joblib.load(path)


def predict_genre(filepath: Path | str, saved_model: dict) -> tuple[str, float | None]:
    """
    Zwraca (przewidziany_gatunek, confidence).

    confidence to prawdopodobieństwo klasy zwycięskiej z predict_proba:
    - dla RandomForest to udział drzew w lesie głosujących na tę klasę,
    - dla SVM to estymacja z Platt scaling (bo pipeline trenowany był
      z probability=True).
    To sensowna, ale przybliżona miara pewności, nie ścisłe prawdopodobieństwo
    statystyczne - jeśli kiedyś podmienimy model na taki bez predict_proba,
    zwracamy None zamiast zmyślać liczbę.
    """
    pipeline = saved_model["pipeline"]
    feature_columns = saved_model["feature_columns"]

    features = extract_features(filepath)
    # Ta sama kolejność kolumn co przy treningu.
    X = pd.DataFrame([features])[feature_columns]

    predicted_genre = pipeline.predict(X)[0]

    confidence = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)[0]
        class_index = list(pipeline.classes_).index(predicted_genre)
        confidence = float(proba[class_index])

    return predicted_genre, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Przewiduje gatunek muzyczny pliku audio.")
    parser.add_argument("audio_path", type=str, help="Ścieżka do pliku audio (WAV/MP3)")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {audio_path}")

    saved_model = load_model()
    genre, confidence = predict_genre(audio_path, saved_model)

    print(f"Predicted genre: {genre}")
    if confidence is not None:
        print(f"Confidence: {confidence:.2f}")
    else:
        print("Confidence: niedostępne dla tego modelu")


if __name__ == "__main__":
    main()