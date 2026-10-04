"""Streamlit-приложение для прогнозирования риска сердечно-сосудистых заболеваний."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predict import CATS, FEATURES, MODEL_NAMES, load_model, predict, risk_level  # noqa: E402

st.set_page_config(page_title="Риск сердечно-сосудистых заболеваний", layout="wide")


@st.cache_resource
def get_model(name):
    return load_model(name)


@st.cache_data
def get_metrics():
    with open(ROOT / "models" / "metrics.json", encoding="utf-8") as f:
        return json.load(f)


st.title("Прогнозирование риска сердечно-сосудистых заболеваний")
st.warning("Учебный прототип. Результат не является диагнозом и не заменяет консультацию врача.")

try:
    meta = get_metrics()
except FileNotFoundError:
    st.error("Файлы модели не найдены. Сначала запустите обучение: python src/train.py")
    st.stop()

# ---------- боковая панель: выбор модели и её метрики
with st.sidebar:
    st.header("Модель")
    keys = list(MODEL_NAMES)
    choice = st.selectbox("Выберите модель", keys, index=keys.index(meta["best"]),
                          format_func=lambda k: MODEL_NAMES[k] + (" (лучшая)" if k == meta["best"] else ""))
    m = meta["results"][choice]
    st.subheader("Метрики на тестовой выборке")
    st.dataframe(pd.DataFrame({"Метрика": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                               "Значение": [round(m[k], 3) for k in
                                            ["accuracy", "precision", "recall", "f1", "roc_auc"]]}),
                 hide_index=True, use_container_width=True)
    st.caption(f"Обучающая выборка: {meta['n_train']}, тестовая: {meta['n_test']} пациентов")

try:
    model = get_model(choice)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

tab1, tab2, tab3 = st.tabs(["Один пациент", "Загрузка CSV", "Качество моделей"])

with tab1:
    c1, c2 = st.columns(2)
    age = c1.number_input("Возраст, лет", 18, 100, 55)
    sex = c1.selectbox("Пол (M – мужской, F – женский)", CATS["Sex"])
    cp = c1.selectbox("Тип боли в груди", CATS["ChestPainType"])
    bp = c1.number_input("Давление в покое, мм рт. ст.", 70, 250, 140)
    chol = c1.number_input("Холестерин, мг/дл (0 – неизвестно)", 0, 700, 250)
    fbs = c1.selectbox("Глюкоза натощак > 120 мг/дл (1 – да)", CATS["FastingBS"])
    ecg = c2.selectbox("ЭКГ в покое", CATS["RestingECG"])
    hr = c2.number_input("Максимальный пульс, уд/мин", 60, 220, 120)
    ang = c2.selectbox("Стенокардия при нагрузке (Y – да)", CATS["ExerciseAngina"], index=1)
    old = c2.number_input("Oldpeak (депрессия ST)", -3.0, 7.0, 1.5, step=0.1)
    slope = c2.selectbox("Наклон сегмента ST", CATS["ST_Slope"], index=1)
    if st.button("Рассчитать риск", type="primary"):
        row = pd.DataFrame([[age, sex, cp, bp, chol, fbs, ecg, hr, ang, old, slope]], columns=FEATURES)
        try:
            p = float(predict(model, row)["risk_probability"].iloc[0])
        except ValueError as exc:
            st.error(f"Ошибка входных данных: {exc}")
        else:
            r1, r2 = st.columns(2)
            r1.metric("Вероятность заболевания", f"{p:.1%}")
            r2.metric("Уровень риска", risk_level(p))
            st.progress(p)
            st.bar_chart(pd.DataFrame({"Вероятность": [1 - p, p]}, index=["Нет ССЗ", "Есть ССЗ"]))
            st.info("Границы уровней риска (0,33 и 0,66) условны и введены в учебных целях.")

with tab2:
    st.write("Загрузите CSV-файл со столбцами: " + ", ".join(FEATURES))
    file = st.file_uploader("CSV-файл с признаками пациентов", type="csv")
    if file is not None:
        try:
            data = pd.read_csv(file)
        except Exception as exc:  # некорректный формат файла
            st.error(f"Не удалось прочитать файл: {exc}")
            st.stop()
        try:
            res = predict(model, data)
        except ValueError as exc:
            for e in str(exc).split("\n"):
                st.error(e)
        else:
            st.success(f"Обработано пациентов: {len(res)}")
            res["risk_probability"] = res["risk_probability"].round(3)
            res = res[["risk_probability", "risk_level"] + FEATURES]
            st.dataframe(res, use_container_width=True)
            st.bar_chart(res["risk_level"].value_counts().reindex(["низкий", "средний", "высокий"]).fillna(0))
            st.download_button("Скачать результаты", res.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")

with tab3:
    rows = [{"Модель": MODEL_NAMES[k], "Accuracy": v["accuracy"], "Precision": v["precision"],
             "Recall": v["recall"], "F1": v["f1"], "ROC-AUC": v["roc_auc"], "CV ROC-AUC": v["cv_roc_auc"]}
            for k, v in meta["results"].items()]
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
    fig_dir = ROOT / "reports" / "figures"
    g1, g2 = st.columns(2)
    if (fig_dir / "fig5_roc.png").exists():
        g1.image(str(fig_dir / "fig5_roc.png"), caption="ROC-кривые")
    if (fig_dir / "fig7_importance.png").exists():
        g2.image(str(fig_dir / "fig7_importance.png"), caption="Важность признаков")
