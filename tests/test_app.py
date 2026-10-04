"""Автоматические тесты сценариев из таблицы 7 отчёта. Запуск: python tests/test_app.py"""
import io
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predict import FEATURES, load_model, predict  # noqa: E402

model = load_model()
test = pd.read_csv(ROOT / "data" / "test_split.csv")
good = test[FEATURES].head(5).copy()
patient = pd.DataFrame([[55, "M", "ASY", 140, 250, 0, "Normal", 120, "Y", 1.5, "Flat"]], columns=FEATURES)
results = []


def check(n, name, fn):
    try:
        ok, note = fn()
    except Exception as exc:  # noqa: BLE001
        ok, note = False, f"исключение: {exc}"
    results.append((n, name, "пройден" if ok else "НЕ пройден", note))


def t1():
    t0 = time.perf_counter()
    r = predict(model, patient)
    ms = (time.perf_counter() - t0) * 1000
    p = r["risk_probability"].iloc[0]
    return 0 <= p <= 1 and r["risk_level"].iloc[0] in ("низкий", "средний", "высокий"), f"p = {p:.3f}, {ms:.0f} мс"


def t2():
    row = patient.copy()
    row["Cholesterol"] = 0
    p = predict(model, row)["risk_probability"].iloc[0]
    return 0 <= p <= 1, f"p = {p:.3f}"


def t3():
    row = patient.copy()
    row["Age"] = 150
    try:
        predict(model, row)
    except ValueError as exc:
        return "Age" in str(exc), str(exc)
    return False, "ошибка не выдана"


def t4():
    r = predict(model, pd.read_csv(io.StringIO(good.to_csv(index=False))))
    ok = len(r) == 5 and r["risk_probability"].between(0, 1).all()
    # регрессионная проверка дефекта: пустой холестерин в CSV не считается ошибкой
    blank = good.copy()
    blank.loc[blank.index[0], "Cholesterol"] = None
    r2 = predict(model, pd.read_csv(io.StringIO(blank.to_csv(index=False))))
    return ok and len(r2) == 5, f"{len(r)} строк, пустой холестерин обработан"


def t5():
    try:
        predict(model, good.drop(columns=["MaxHR", "Sex"]))
    except ValueError as exc:
        return "MaxHR" in str(exc) and "Sex" in str(exc), str(exc)
    return False, "ошибка не выдана"


def t6():
    try:
        pd.read_csv(io.BytesIO(b"\x89PNG\r\n\x1a\n\x00\x00binary"))
        data = pd.read_csv(io.BytesIO(b"\x89PNG\r\n\x1a\n\x00\x00binary"))
        predict(model, data)
    except Exception as exc:  # noqa: BLE001
        return True, type(exc).__name__
    return False, "ошибка не выдана"


def t7():
    src = ROOT / "models" / "best_model.joblib"
    tmp = ROOT / "models" / "best_model.bak"
    shutil.move(src, tmp)
    try:
        load_model()
    except FileNotFoundError as exc:
        return "train.py" in str(exc), "FileNotFoundError"
    finally:
        shutil.move(tmp, src)
    return False, "ошибка не выдана"


def t8():
    import joblib
    direct = joblib.load(ROOT / "models" / "best_model.joblib").predict_proba(
        test[FEATURES].assign(Cholesterol=test["Cholesterol"]))[:, 1]
    app = predict(model, test[FEATURES].fillna({"Cholesterol": 0}))["risk_probability"].values
    diff = abs(direct - app).max()
    return diff < 1e-9, f"макс. расхождение {diff:.1e}"


for n, name, fn in [(1, "Корректные данные", t1), (2, "Холестерин 0", t2), (3, "Вне диапазона", t3),
                    (4, "Корректный CSV", t4), (5, "Нет столбцов", t5), (6, "Неверный формат", t6),
                    (7, "Нет файла модели", t7), (8, "Совпадение с ноутбуком", t8)]:
    check(n, name, fn)
for r in results:
    print(*r, sep=" | ")
print("Итого пройдено:", sum(r[2] == "пройден" for r in results), "из", len(results))
