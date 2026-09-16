"""
train.py

Wczytuje wcześniej policzone cechy (data/features/features.csv),
odtwarza ten sam podział train/val/test co przy ekstrakcji cech
(przez dataset.scan_dataset + dataset.split_dataset, z tym samym
random_state - więc podział jest deterministyczny i spójny między
skryptami), trenuje i porównuje dwa proste modele (Random Forest,
SVM), wybiera lepszy na podstawie walidacji, a na końcu dotrenowuje
zwycięski model na train+val i zapisuje go przez joblib.

Model jest zapisywany jako sklearn Pipeline (StandardScaler + klasyfikator),
więc predict.py nie musi pamiętać osobno o skalowaniu cech - wystarczy
wczytać jeden obiekt i wywołać .predict() / .predict_proba().
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.dataset import DEFAULT_DATA_DIR, scan_dataset, split_dataset

FEATURES_PATH = Path("data/features/features.csv")
MODEL_PATH = Path("models/model.joblib")

# Kolumny, które są metadanymi, nie cechami - wszystko poza nimi
# w features.csv to wektor cech użyty do treningu.
META_COLUMNS = ["track_id", "genre"]

RANDOM_STATE = 42


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Brak {path}. Najpierw uruchom: python -m src.features"
        )
    return pd.read_csv(path)


def attach_split(feature_df: pd.DataFrame, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """
    Odtwarza ten sam podział train/val/test, co przy ekstrakcji cech,
    i dopisuje go jako kolumnę "split" do feature_df.

    Podział nie jest zapisany w features.csv celowo - żeby nie
    duplikować logiki splitu w dwóch miejscach. dataset.split_dataset
    z tym samym random_state zawsze da ten sam wynik, więc odtworzenie
    go tutaj jest bezpieczne i deterministyczne, pod warunkiem że
    zawartość data/genres/ nie zmieniła się między uruchomieniami.
    """
    file_df = scan_dataset(data_dir)
    train_df, val_df, test_df = split_dataset(file_df, random_state=RANDOM_STATE)

    split_map = {}
    split_map.update({tid: "train" for tid in train_df["track_id"]})
    split_map.update({tid: "val" for tid in val_df["track_id"]})
    split_map.update({tid: "test" for tid in test_df["track_id"]})

    feature_df = feature_df.copy()
    feature_df["split"] = feature_df["track_id"].map(split_map)

    missing = feature_df["split"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} wierszy w features.csv nie pasuje do żadnego track_id "
            "ze skanu data/genres/. Czy dataset się zmienił od czasu "
            "ostatniej ekstrakcji cech?"
        )
    return feature_df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c not in META_COLUMNS + ["split"]]
    return df[feature_cols], df["genre"]


def build_candidate_models() -> dict[str, Pipeline]:
    """
    Dwa proste, standardowe modele na cechach klasycznych.
    Oba w Pipeline ze StandardScaler - RF w teorii nie wymaga
    skalowania, ale nie szkodzi, a upraszcza kod (jeden wzorzec dla
    obu modeli, jeden zapisany artefakt).
    """
    return {
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                probability=True,  # potrzebne do sensownego predict_proba w predict.py
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def compare_on_validation(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, dict[str, float]]:
    results = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        results[name] = {
            "accuracy": accuracy_score(y_val, y_pred),
            "macro_f1": f1_score(y_val, y_pred, average="macro"),
        }
    return results


def print_comparison(results: dict[str, dict[str, float]]) -> None:
    print("\nPorównanie modeli na zbiorze walidacyjnym:")
    print(f"{'model':<15} {'accuracy':>10} {'macro F1':>10}")
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['accuracy']:>10.3f} {metrics['macro_f1']:>10.3f}")


def main() -> None:
    feature_df = load_features()
    feature_df = attach_split(feature_df)

    train_df = feature_df[feature_df["split"] == "train"]
    val_df = feature_df[feature_df["split"] == "val"]

    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)

    models = build_candidate_models()
    results = compare_on_validation(models, X_train, y_train, X_val, y_val)
    print_comparison(results)

    best_name = max(results, key=lambda name: results[name]["macro_f1"])
    print(f"\nWybrano: {best_name} (najwyższy macro F1 na walidacji)")

    # Dotrenowujemy zwycięski model na train+val, żeby wykorzystać
    # więcej danych do finalnego modelu - val nie jest już potrzebny
    # do selekcji, więc nie ma powodu, żeby leżał odłogiem. Test set
    # pozostaje nietknięty i posłuży do finalnej, uczciwej oceny
    # w evaluate.py.
    train_val_df = feature_df[feature_df["split"].isin(["train", "val"])]
    X_train_val, y_train_val = get_X_y(train_val_df)

    final_pipeline = build_candidate_models()[best_name]
    final_pipeline.fit(X_train_val, y_train_val)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_columns = list(X_train_val.columns)
    joblib.dump(
        {"pipeline": final_pipeline, "feature_columns": feature_columns, "model_name": best_name},
        MODEL_PATH,
    )
    print(f"Zapisano finalny model ({best_name}) do {MODEL_PATH}")


if __name__ == "__main__":
    main()