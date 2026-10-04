"""Обучение, сравнение и сохранение моделей прогнозирования риска ССЗ."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
RANDOM_STATE = 42
NUM = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CAT = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
BIN = ["FastingBS"]


def metrics(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, proba),
            "confusion_matrix": confusion_matrix(y, pred).tolist()}


# 1. Загрузка и очистка данных
df = pd.read_csv(ROOT / "data" / "heart.csv").drop_duplicates()
df = df[df["RestingBP"] > 0]
df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
X, y = df.drop(columns="HeartDisease"), df["HeartDisease"]

# 2. Стратифицированное разделение 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
X_test.assign(HeartDisease=y_test).to_csv(ROOT / "data" / "test_split.csv", index=False)

# 3. Предобработка (обучается только на обучающей выборке внутри Pipeline)
prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ("bin", "passthrough", BIN),
])

# 4. Модели и сетки гиперпараметров
models = {
    "logreg": (LogisticRegression(max_iter=1000), {"clf__C": [0.1, 1, 10]}),
    "rf": (RandomForestClassifier(random_state=RANDOM_STATE),
           {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 5, 10]}),
    "gb": (GradientBoostingClassifier(random_state=RANDOM_STATE),
           {"clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3]}),
}
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
results, fitted, baseline = {}, {}, {}
for name, (clf, grid) in models.items():
    pipe = Pipeline([("prep", clone(prep)), ("clf", clf)])
    # модель с параметрами по умолчанию — для сравнения «до/после» настройки
    baseline[name] = metrics(clone(pipe).fit(X_train, y_train), X_test, y_test)
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    results[name] = {"best_params": search.best_params_,
                     "cv_roc_auc": search.best_score_, **metrics(best, X_test, y_test)}
    fitted[name] = best
    joblib.dump(best, MODELS / f"{name}.joblib")
    print(name, {k: round(v, 3) for k, v in results[name].items()
                 if isinstance(v, float)})

# 5. Выбор и сохранение лучшей модели: критерий — ROC-AUC на кросс-валидации
# (тестовая выборка в выборе не участвует), при равенстве — полнота (Recall)
best_name = max(results, key=lambda n: (round(results[n]["cv_roc_auc"], 3),
                                        results[n]["recall"]))
joblib.dump(fitted[best_name], MODELS / "best_model.joblib")
joblib.dump(fitted[best_name].named_steps["prep"], MODELS / "preprocessor.joblib")
with open(MODELS / "metrics.json", "w", encoding="utf-8") as f:
    json.dump({"best": best_name, "results": results, "baseline": baseline,
               "n_train": len(X_train), "n_test": len(X_test)},
              f, ensure_ascii=False, indent=2, default=float)
schema = {"numeric": {c: [float(X[c].min()), float(X[c].max())] for c in NUM},
          "categorical": {c: sorted(X[c].unique().tolist()) for c in CAT},
          "binary": {c: [0, 1] for c in BIN},
          "output": "вероятность наличия заболевания (0..1) и класс при пороге 0.5"}
with open(MODELS / "feature_schema.json", "w", encoding="utf-8") as f:
    json.dump(schema, f, ensure_ascii=False, indent=2)
print("Лучшая модель:", best_name)
