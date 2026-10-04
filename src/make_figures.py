"""Построение рисунков для отчёта и расчёт статистик EDA."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
NAMES = {"logreg": "Логистическая регрессия", "rf": "Случайный лес", "gb": "Градиентный бустинг"}

df = pd.read_csv(ROOT / "data" / "heart.csv")
num = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
stats = {}

# ---------- Рисунок 3: распределения и баланс классов
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.ravel(), num):
    sns.histplot(data=df, x=col, hue="HeartDisease", bins=30, ax=ax, palette=["#4C72B0", "#DD8452"],
                 alpha=0.6, edgecolor="white")
    ax.set_title(col)
    ax.set_ylabel("Количество")
    ax.get_legend().set_title("HeartDisease")
ax = axes.ravel()[5]
counts = df["HeartDisease"].value_counts().sort_index()
bars = ax.bar(["0 – нет", "1 – есть"], counts.values, color=["#4C72B0", "#DD8452"])
ax.bar_label(bars, labels=[f"{v} ({v / len(df):.0%})" for v in counts.values])
ax.set_ylim(0, counts.max() * 1.15)
ax.set_title("Баланс классов HeartDisease")
ax.set_ylabel("Количество пациентов")
fig.tight_layout()
fig.savefig(FIG / "fig3_distributions.png", dpi=200)
plt.close(fig)

# ---------- Рисунок 4: корреляционная матрица
enc = df.copy()
enc["Sex_M"] = (enc["Sex"] == "M").astype(int)
enc["ExerciseAngina_Y"] = (enc["ExerciseAngina"] == "Y").astype(int)
enc["ST_Slope_Flat"] = (enc["ST_Slope"] == "Flat").astype(int)
enc["ChestPain_ASY"] = (enc["ChestPainType"] == "ASY").astype(int)
cols = num + ["FastingBS", "Sex_M", "ExerciseAngina_Y", "ST_Slope_Flat", "ChestPain_ASY", "HeartDisease"]
corr = enc[cols].corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, annot_kws={"size": 10})
ax.set_title("Корреляционная матрица признаков")
fig.tight_layout()
fig.savefig(FIG / "fig4_correlation.png", dpi=200)
plt.close(fig)
stats["corr_target"] = corr["HeartDisease"].drop("HeartDisease").round(2).sort_values().to_dict()

# ---------- статистики EDA
stats["describe"] = df[num].describe().round(1).to_dict()
stats["cat_shares"] = {c: (df[c].value_counts(normalize=True) * 100).round(1).to_dict()
                       for c in ["Sex", "ChestPainType", "ST_Slope", "ExerciseAngina", "RestingECG"]}
stats["disease_rate_by"] = {c: (df.groupby(c)["HeartDisease"].mean() * 100).round(1).to_dict()
                            for c in ["ST_Slope", "ChestPainType", "ExerciseAngina", "Sex"]}
clean = df[df.RestingBP > 0].copy()
clean.loc[clean.Cholesterol == 0, "Cholesterol"] = np.nan
iqr_out = {}
for c in num:
    q1, q3 = clean[c].quantile([0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    iqr_out[c] = int(((clean[c] < lo) | (clean[c] > hi)).sum())
stats["iqr_outliers"] = iqr_out
stats["oldpeak_negative"] = int((df.Oldpeak < 0).sum())

# ---------- Рисунки 5–7: модели
test = pd.read_csv(ROOT / "data" / "test_split.csv")
X_test, y_test = test.drop(columns="HeartDisease"), test["HeartDisease"]
meta = json.load(open(ROOT / "models" / "metrics.json", encoding="utf-8"))
best = meta["best"]

fig, ax = plt.subplots(figsize=(7.5, 6.5))
for name, label in NAMES.items():
    model = joblib.load(ROOT / "models" / f"{name}.joblib")
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc_score(y_test, proba):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Случайное угадывание")
ax.set_xlabel("Доля ложноположительных (FPR)")
ax.set_ylabel("Доля верноположительных (TPR, Recall)")
ax.set_title("ROC-кривые моделей на тестовой выборке")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "fig5_roc.png", dpi=200)
plt.close(fig)

best_model = joblib.load(ROOT / "models" / "best_model.joblib")
fig, ax = plt.subplots(figsize=(6, 5.2))
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, display_labels=["Нет ССЗ (0)", "Есть ССЗ (1)"],
                                      cmap="Blues", colorbar=False, ax=ax)
ax.set_xlabel("Предсказанный класс")
ax.set_ylabel("Истинный класс")
ax.set_title(f"Матрица ошибок: {NAMES[best]}")
fig.tight_layout()
fig.savefig(FIG / "fig6_confusion.png", dpi=200)
plt.close(fig)

clf = best_model.named_steps["clf"]
names = best_model.named_steps["prep"].get_feature_names_out()
names = [n.split("__", 1)[1] for n in names]
imp = pd.Series(clf.feature_importances_, index=names).sort_values()
stats["importance_top"] = imp.sort_values(ascending=False).head(8).round(3).to_dict()
fig, ax = plt.subplots(figsize=(8, 7))
imp.plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_xlabel("Важность признака (среднее уменьшение неоднородности)")
ax.set_title(f"Важность признаков: {NAMES[best]}")
fig.tight_layout()
fig.savefig(FIG / "fig7_importance.png", dpi=200)
plt.close(fig)

json.dump(stats, open(ROOT / "reports" / "eda_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(stats, ensure_ascii=False, indent=1))
