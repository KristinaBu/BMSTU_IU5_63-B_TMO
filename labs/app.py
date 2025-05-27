import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, 
                             AdaBoostRegressor, GradientBoostingRegressor)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title(" Анализ данных: сравнение моделей ML")
st.write("Загрузите данные и выберите модели для обучения")

# 1. Загрузка данных
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("Данные успешно загружены!")
    
    # Показ первых строк
    if st.checkbox("Показать первые 5 строк"):
        st.write(data.head())

    # Выбор целевой переменной
    target = st.selectbox("Выберите целевую переменную (y)", data.columns)

    # Выбор признаков
    features = st.multiselect("Выберите признаки (X)", data.columns, default=list(data.columns[:-1]))

    # Разделение данных
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# После загрузки данных и выбора признаков:

# 1. Преобразование категориальных признаков

    # Выбираем только числовые колонки (или преобразуем категориальные)
    numeric_features = data[features].select_dtypes(include=['number']).columns
    if len(numeric_features) < len(features):
        st.warning("Обнаружены нечисловые признаки. Производим кодирование...")
        for col in set(features) - set(numeric_features):
            data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    # 2. Проверка на пропущенные значения
    if data[features + [target]].isnull().any().any():
        st.warning("Обнаружены пропущенные значения. Заполняем медианами...")
        data = data.fillna(data.median(numeric_only=True))

    # 3. Разделение данных (после обработки)
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# 2. Выбор моделей
st.sidebar.header("Выбор моделей")
models_options = {
    "Random Forest": RandomForestRegressor(),
    "Extra Trees": ExtraTreesRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}
selected_models = st.sidebar.multiselect("Выберите модели", list(models_options.keys()))

# 3. Настройка гиперпараметров
st.sidebar.header("Гиперпараметры")
params = {}
for model_name in selected_models:
    st.sidebar.subheader(model_name)
    if model_name == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 300, 100, key="rf_n_est")
    elif model_name == "AdaBoost":
        params["learning_rate"] = st.sidebar.slider("learning_rate", 0.01, 1.0, 0.1, key="ada_lr")

# 4. Обучение и оценка
if st.button("Обучить модели") and uploaded_file:
    results = []
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name in selected_models:
        model = models_options[model_name].set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Метрики
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({"Модель": model_name, "MSE": mse, "MAE": mae, "R²": r2})
        
        # График предсказаний
        ax.scatter(y_test, y_pred, alpha=0.5, label=model_name)
    
    # Вывод результатов
    st.subheader("Результаты обучения")
    results_df = pd.DataFrame(results)
    st.table(results_df.style.highlight_min(axis=0, color="#90EE90"))
    
    # График
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel("Реальные значения")
    ax.set_ylabel("Предсказанные значения")
    ax.legend()
    st.pyplot(fig)

    # Важность признаков (для tree-based моделей)
    if "Random Forest" in selected_models:
        st.subheader("Важность признаков (Random Forest)")
        importances = models_options["Random Forest"].feature_importances_
        importance_df = pd.DataFrame({"Признак": features, "Важность": importances})
        st.bar_chart(importance_df.set_index("Признак"))