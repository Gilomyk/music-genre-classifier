"""
Testy dla dataset.py.

Zamiast prawdziwego GTZAN używamy tymczasowej, fałszywej struktury
katalogów (puste pliki .wav) - interesuje nas tylko logika skanowania
i podziału danych, nie samo audio.
"""

import pytest

from src.dataset import scan_dataset, split_dataset


@pytest.fixture
def fake_data_dir(tmp_path):
    genres = {"blues": 6, "rock": 6}
    for genre, n_files in genres.items():
        genre_dir = tmp_path / genre
        genre_dir.mkdir()
        for i in range(n_files):
            (genre_dir / f"{genre}.{i:05d}.wav").write_bytes(b"")
    return tmp_path


def test_scan_dataset_finds_all_files(fake_data_dir):
    df = scan_dataset(fake_data_dir)
    assert len(df) == 12
    assert set(df["genre"]) == {"blues", "rock"}


def test_scan_dataset_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_dataset(tmp_path / "does_not_exist")


def test_scan_dataset_empty_dir_raises(tmp_path):
    (tmp_path / "blues").mkdir()
    with pytest.raises(ValueError):
        scan_dataset(tmp_path)


def test_split_dataset_no_overlap(fake_data_dir):
    df = scan_dataset(fake_data_dir)
    train_df, val_df, test_df = split_dataset(
        df, val_size=0.2, test_size=0.2, random_state=0
    )

    train_ids = set(train_df["track_id"])
    val_ids = set(val_df["track_id"])
    test_ids = set(test_df["track_id"])

    # Żaden track_id nie powinien wystąpić w więcej niż jednym splicie.
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Wszystkie utwory powinny się gdzieś znaleźć.
    assert train_ids | val_ids | test_ids == set(df["track_id"])


def test_split_dataset_both_genres_in_train(fake_data_dir):
    df = scan_dataset(fake_data_dir)
    train_df, _, _ = split_dataset(df, val_size=0.2, test_size=0.2, random_state=0)
    assert set(train_df["genre"]) == {"blues", "rock"}
