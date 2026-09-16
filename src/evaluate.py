"""
evaluate.py

Wczytuje wytrenowany model (models/model.joblib) oraz zbiór testowy
(ten sam test split, który był odłożony na bok podczas train.py -
nie brał udziału ani w treningu, ani w wyborze modelu) i liczy:
accuracy, macro F1, pełny classification_report oraz confusion matrix.
Dodatkowo wypisuje, które pary gatunków są najczęściej mylone.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.train import FEATURES_PATH, MODEL_PATH, attach_split, get_X_y, load_features

CONFUSION_MATRIX_PATH = Path("models/confusion_matrix.png")


def load_model(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Brak {path}. Najpierw uruchom: python -m src.train"
        )
    return joblib.load(path)


def get_test_split(features_path: Path = FEATURES_PATH) -> pd.DataFrame:
    feature_df = load_features(features_path)
    feature_df = attach_split(feature_df)
    return feature_df[feature_df["split"] == "test"].reset_index(drop=True)


def plot_confusion_matrix(cm, labels: list[str], output_path: Path = CONFUSION_MATRIX_PATH) -> None:
    """
    Rysuje confusion matrix znormalizowaną wierszami (czyli % predykcji
    per prawdziwa klasa) - łatwiej wtedy zauważyć, z czym dany gatunek
    jest mylony, niż patrząc na surowe liczby, które zależą od tego,
    ile przykładów danej klasy jest w teście.
    """
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predykcja")
    plt.ylabel("Prawdziwy gatunek")
    plt.title("Confusion matrix (znormalizowana wierszami)")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Zapisano confusion matrix do {output_path}")


def most_confused_pairs(cm, labels: list[str], top_n: int = 5) -> list[tuple[str, str, int]]:
    """
    Zwraca listę (prawdziwy_gatunek, przewidziany_gatunek, liczba_pomyłek)
    posortowaną malejąco, pomijając przekątną (poprawne predykcje).
    """
    pairs = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i == j:
                continue
            count = cm[i, j]
            if count > 0:
                pairs.append((true_label, pred_label, int(count)))

    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs[:top_n]


def main() -> None:
    saved = load_model()
    pipeline = saved["pipeline"]
    feature_columns = saved["feature_columns"]
    model_name = saved["model_name"]

    test_df = get_test_split()
    X_test, y_test = get_X_y(test_df)
    # Ta sama kolejność kolumn co przy treningu - na wypadek gdyby
    # coś się zmieniło w features.csv między biegami.
    X_test = X_test[feature_columns]

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Model: {model_name}")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Test macro F1: {macro_f1:.3f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plot_confusion_matrix(cm, labels)

    print("\nNajczęściej mylone pary (prawdziwy -> przewidziany, liczba):")
    for true_label, pred_label, count in most_confused_pairs(cm, labels):
        print(f"  {true_label:<10} -> {pred_label:<10} : {count}")


if __name__ == "__main__":
    main()