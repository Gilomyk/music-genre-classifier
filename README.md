# Music Genre Classifier

Klasyfikacja gatunku muzycznego na podstawie cech audio (nie deep
learningu na surowym sygnale/spektrogramie). Projekt portfolio
pokazujący kompletny, prosty pipeline ML: od surowego audio, przez
ręczną ekstrakcję cech, po wytrenowany model i predykcję na własnym
pliku.

Nie jest to próba osiągnięcia najlepszego możliwego wyniku — priorytetem
było zbudowanie działającego, zrozumiałego pipeline'u.

## Dataset

[GTZAN Genre Collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
— 1000 utworów, 10 gatunków po 100 utworów, ~30 sekund każdy:

blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

Dataset nie jest częścią repozytorium (zbyt duży). Pobiera się go
osobno i rozpakowuje tak, by podfolder `genres_original/` znalazł się
w `data/genres/` (patrz sekcja "Uruchomienie").

GTZAN pochodzi z ok. 1999–2000 roku, co ma znaczenie dla interpretacji
wyników — patrz "Ograniczenia".

## Pipeline

audio (.wav)
↓
ekstrakcja cech (librosa)
↓
tabela cech (data/features/features.csv)
↓
train/val/test split (stratyfikowany po gatunku, po track_id)
↓
trening + porównanie modeli (Random Forest, SVM)
↓
wybór modelu wg macro F1 na walidacji
↓
finalny trening na train+val
↓
ewaluacja na test set
↓
predykcja dla dowolnego pliku audio


## Wykorzystane cechy

Dla każdego utworu liczone są cechy klasyczne (nie spektrogram jako
obraz), agregowane w czasie jako `mean` + `std`:

| Cecha | Co opisuje |
|---|---|
| MFCC (20 współczynników) | barwę / timbre dźwięku |
| Chroma (12 klas wysokości dźwięku) | rozkład energii harmonicznej |
| Spectral centroid | "jasność" brzmienia |
| Spectral bandwidth | szerokość widma wokół centroidu |
| Spectral rolloff | częstotliwość skupiająca 85% energii |
| Zero crossing rate | częstość przecinania zera przez sygnał |
| Tempo (BPM) | jedna wartość na utwór |

Łącznie ok. 73 cechy na utwór.

## Model

Porównano dwa proste modele klasyczne (oba w `Pipeline` ze
`StandardScaler`): Random Forest i SVM (RBF).

Wybór modelu następuje na zbiorze walidacyjnym wg macro F1 (a nie
accuracy — przy 10 klasach lepiej pokazuje, czy model nie ignoruje
trudniejszych gatunków kosztem łatwiejszych).

Wynik na walidacji:

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest | 0.727 | **0.727** |
| SVM (RBF) | 0.733 | 0.726 |

Różnica jest w granicach szumu — wybrano Random Forest.

## Wyniki (test set, nietknięty podczas treningu i wyboru modelu)

Test accuracy: 0.633
Test macro F1: 0.621


| Gatunek | Precision | Recall | F1 |
|---|---|---|---|
| classical | 0.82 | 0.93 | 0.88 |
| metal | 0.76 | 0.87 | 0.81 |
| pop | 0.72 | 0.87 | 0.79 |
| jazz | 0.75 | 0.80 | 0.77 |
| country | 0.62 | 0.67 | 0.65 |
| hiphop | 0.50 | 0.53 | 0.52 |
| reggae | 0.53 | 0.53 | 0.53 |
| blues | 0.67 | 0.53 | 0.59 |
| disco | 0.42 | 0.33 | 0.37 |
| rock | 0.36 | 0.27 | 0.31 |

Baseline losowy dla 10 klas to 10% — model wyraźnie się uczy, ale
wyniki są nierówne między gatunkami.

### Confusion matrix

![Confusion matrix](models/confusion_matrix.png)

Najczęściej mylone pary: `disco → pop`, `country → rock`,
`hiphop → disco`, `hiphop → reggae`, `jazz → classical`.

Wzorzec ma sens muzycznie: classical, jazz i metal mają wyraziste,
łatwo odróżnialne sygnatury spektralne i osiągają wysoki wynik. Rock
i disco mieszają się z wieloma innymi gatunkami — dla rocka to znany
problem GTZAN: etykieta obejmuje bardzo niejednorodny zbiór stylów
(od soft rocka po coś bliskiego metalowi), więc 100 przykładów nie
wystarcza, by uchwycić tę różnorodność jedną klasą.

## Uruchomienie

```bash
# 1. środowisko
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. dataset
# Pobierz GTZAN z Kaggle i rozpakuj tak, by struktura wyglądała:
#   data/genres/genres_original/<gatunek>/<plik>.wav
# (images_original/ i pliki features_*.csv z Kaggle nie są używane)

# 3. pipeline
python -m src.features    # ekstrakcja cech -> data/features/features.csv
python -m src.train        # trening + wybór modelu -> models/model.joblib
python -m src.evaluate     # ewaluacja na test set + confusion matrix

# 4. predykcja własnego pliku
python -m src.predict path/to/song.wav
```

## Przykład predykcji własnego audio

Model przetestowany na czterech utworach spoza GTZAN — dwóch
współczesnych produkcjach rockowych i dwóch klasykach (funk/disco,
pop) z lat 70./80.:

$ python -m src.predict September.mp3
Predicted genre: rock
Confidence: 0.31

$ python -m src.predict I_wanna_dance_with_somebody.mp3
Predicted genre: pop
Confidence: 0.45


Utwór popowy z 1987 roku trafił najbliżej realnego gatunku, z
najwyższym confidence spośród testowanych plików. Utwory ze
współczesnym masteringiem konsekwentnie dostawały niską pewność
predykcji (0.20–0.31) i trafiały w gatunki niepowiązane ze swoim
rzeczywistym stylem — patrz "Ograniczenia".

## Ograniczenia projektu

- **GTZAN jest mały i stary.** 100 przykładów na gatunek to mało, a
  dataset pochodzi z ok. 1999–2000 roku. Cechy klasyczne (MFCC,
  spectral centroid itd.) są częściowo wrażliwe na charakter
  masteringu/produkcji nagrania, nie tylko na gatunkową treść
  muzyczną — model gorzej generalizuje na nagrania ze współczesną
  produkcją, co widać w sekcji powyżej.
- **Etykieta "rock" w GTZAN jest bardzo niejednorodna** i systematycznie
  wypada najsłabiej (F1 0.31) — to znany problem tego datasetu, nie
  wada pipeline'u.
- **Confidence z `predict_proba` to przybliżona miara pewności**
  (dla RF: udział drzew głosujących na klasę; dla SVM: estymacja
  Platt scaling), nie ścisłe prawdopodobieństwo statystyczne.
- Cechy są agregowane statystykami (mean/std) po całym 30-sekundowym
  klipie — model nie widzi struktury czasowej utworu (np. zmiany
  dynamiki między zwrotką a refrenem).
- Projekt nie był optymalizowany pod maksymalny wynik (brak tuningu
  hiperparametrów, brak augmentacji, brak CNN na spektrogramie) —
  celowo, zgodnie z założeniem "działający, zrozumiały pipeline"
  zamiast wyniku SOTA.

## Licencja

Patrz `LICENSE`.