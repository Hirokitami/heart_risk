"""Загрузка модели, проверка входных данных и прогнозирование риска."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

FEATURES = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
            "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
            "Oldpeak", "ST_Slope"]
RANGES = {"Age": (18, 100), "RestingBP": (70, 250), "Cholesterol": (0, 700),
          "MaxHR": (60, 220), "Oldpeak": (-3.0, 7.0)}
CATS = {"Sex": ["M", "F"], "ChestPainType": ["ASY", "NAP", "ATA", "TA"],
        "RestingECG": ["Normal", "ST", "LVH"], "ExerciseAngina": ["N", "Y"],
        "ST_Slope": ["Up", "Flat", "Down"], "FastingBS": [0, 1]}
MODEL_NAMES = {"logreg": "Логистическая регрессия", "rf": "Случайный лес",
               "gb": "Градиентный бустинг"}


def load_model(name="best_model"):
    """Загружает сохранённый конвейер (предобработка + классификатор)."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Файл модели {path} не найден. Запустите src/train.py.")
    return joblib.load(path)


def validate(df):
    """Возвращает список ошибок входных данных (пустой список — данные корректны)."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return ["Отсутствуют столбцы: " + ", ".join(missing)]
    errors = []
    for col, (lo, hi) in RANGES.items():
        values = pd.to_numeric(df[col], errors="coerce")
        bad = ~values.between(lo, hi)
        if col == "Cholesterol":  # пустое значение холестерина = «неизвестно»
            bad &= df[col].notna()
        if bad.any():
            rows = ", ".join(str(i + 1) for i in df.index[bad][:5])
            errors.append(f"{col}: значения вне диапазона [{lo}; {hi}] (строки {rows})")
    for col, allowed in CATS.items():
        if (~df[col].isin(allowed)).any():
            errors.append(f"{col}: допустимы значения {allowed}")
    return errors


def prepare(df):
    """Отбирает признаки и заменяет «неизвестный» холестерин (0) на пропуск."""
    df = df[FEATURES].copy()
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
    return df


def risk_level(p):
    """Условный уровень риска (учебные границы, не клинические пороги)."""
    return "низкий" if p < 0.33 else "средний" if p < 0.66 else "высокий"


def predict(model, df):
    """Проверяет данные и возвращает таблицу с вероятностью и уровнем риска."""
    errors = validate(df)
    if errors:
        raise ValueError("\n".join(errors))
    result = df.copy()
    result["risk_probability"] = model.predict_proba(prepare(df))[:, 1]
    result["risk_level"] = result["risk_probability"].map(risk_level)
    return result
