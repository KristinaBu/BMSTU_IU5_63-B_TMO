import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix, RocCurveDisplay)
from sklearn.preprocessing import LabelEncoder

st.title('🫀 Прогнозирование сердечных заболеваний')
st.markdown("""
Это приложение демонстрирует работу модели Random Forest для классификации сердечных заболеваний.
Вы можете настроить гиперпараметры модели и увидеть, как они влияют на качество предсказаний. 
_Поставьте отлично, пожалуйста._
""")


# загрузка и предобработка данных
@st.cache_data
def load_data():
    data = pd.read_csv("heart.csv")

    # кодируем всякое
    le = LabelEncoder()
    data['Sex'] = le.fit_transform(data['Sex'])
    data['ExerciseAngina'] = le.fit_transform(data['ExerciseAngina'])

    # One-Hot Encoding
    data = pd.get_dummies(data, columns=['ChestPainType', 'RestingECG', 'ST_Slope'])

    return data


data = load_data()

# Разделение данных
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Сайдбар с настройками модели
st.sidebar.header('⚙️ Настройки шикарной модели')
n_estimators = st.sidebar.slider('Количество прекрасных деревьев', 10, 500, 100, 10)
max_depth = st.sidebar.slider('Максимальная глубина дерева', 1, 20, 5)
min_samples_split = st.sidebar.slider('Минимальное число образцов для разделения', 2, 10, 2)
min_samples_leaf = st.sidebar.slider('Минимальное число образцов в листе', 1, 10, 1)

# Обучение модели с выбранными параметрами
model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42
)
model.fit(X_train, y_train)

# Предсказания и метрики
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# Отображение метрик
st.header('📊 Крутейшие метрики модели')
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("Precision", f"{precision:.3f}")
col3.metric("Recall", f"{recall:.3f}")
col4.metric("F1-score", f"{f1:.3f}")
col5.metric("ROC-AUC", f"{roc_auc:.3f}")

# Визуализации
st.header('📈 Четкая визуализация результатов')

# ROC-кривая
st.subheader('ROC-кривая')
roc_fig = plt.figure(figsize=(8, 6))
RocCurveDisplay.from_estimator(model, X_test, y_test, ax=plt.gca())
plt.plot([0, 1], [0, 1], linestyle='--')
st.pyplot(roc_fig)

# Матрица ошибок
st.subheader('Матрица ошибок (на ошибках надо учиться)')
cm = confusion_matrix(y_test, y_pred)
cm_fig = plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Здоров', 'Болен'],
            yticklabels=['Здоров', 'Болен'])
plt.xlabel('Предсказание (из печенья)')
plt.ylabel('Фактическое значение')
st.pyplot(cm_fig)

# Важность признаков
st.subheader('Важность признаков')
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Признак': features, 'Важность': importances})
importance_df = importance_df.sort_values('Важность', ascending=False).head(10)

importance_fig = plt.figure(figsize=(10, 6))
sns.barplot(x='Важность', y='Признак', data=importance_df)
plt.title('Топ-10 важнейших признаков')
st.pyplot(importance_fig)

# Предсказание для новых данных
st.header('🔮 Сделать предсказание (гадаем на листьях деревьев бесплатно)')

st.markdown("""
Вы можете ввести параметры пациента и получить предсказание модели:
""")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Возраст (Age)', min_value=20, max_value=100, value=50)
    sex = st.selectbox('Пол (Sex)', ['M', 'F'])
    chest_pain = st.selectbox('Тип боли в груди (ChestPainType)', ['ATA', 'NAP', 'ASY', 'TA'])
    resting_bp = st.number_input('Давление в покое (RestingBP)', min_value=80, max_value=200, value=120)
    cholesterol = st.number_input('Холестерин (Cholesterol)', min_value=100, max_value=600, value=200)

with col2:
    fasting_bs = st.selectbox('Уровень сахара натощак > 120 мг/дл (FastingBS)', [0, 1])
    resting_ecg = st.selectbox('ЭКГ в покое (RestingECG)', ['Normal', 'ST', 'LVH'])
    max_hr = st.number_input('Макс. ЧСС (MaxHR)', min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox('Стенокардия при нагрузке (ExerciseAngina)', ['N', 'Y'])
    oldpeak = st.number_input('Депрессия ST (Oldpeak)', min_value=0.0, max_value=6.0, value=1.0)
    st_slope = st.selectbox('Наклон сегмента ST (ST_Slope)', ['Up', 'Flat', 'Down'])

if st.button('Предсказать'):
    sex_encoded = 1 if sex == 'M' else 0
    exercise_angina_encoded = 1 if exercise_angina == 'Y' else 0

    #  DataFrame с  данными
    input_data = {
        'Age': age,
        'Sex': sex_encoded,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina_encoded,
        'Oldpeak': oldpeak,
        'ChestPainType_ATA': 1 if chest_pain == 'ATA' else 0,
        'ChestPainType_NAP': 1 if chest_pain == 'NAP' else 0,
        'ChestPainType_ASY': 1 if chest_pain == 'ASY' else 0,
        'ChestPainType_TA': 1 if chest_pain == 'TA' else 0,
        'RestingECG_Normal': 1 if resting_ecg == 'Normal' else 0,
        'RestingECG_ST': 1 if resting_ecg == 'ST' else 0,
        'RestingECG_LVH': 1 if resting_ecg == 'LVH' else 0,
        'ST_Slope_Flat': 1 if st_slope == 'Flat' else 0,
        'ST_Slope_Up': 1 if st_slope == 'Up' else 0,
        'ST_Slope_Down': 1 if st_slope == 'Down' else 0
    }

    # DataFrame с правильными колонками
    input_df = pd.DataFrame([input_data])

    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # упоряд колонок как в обучающих данных
    input_df = input_df[X.columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f'🚨 Риск сердечного заболевания: {probability:.1%}. Отдыхайте больше!!!')
    else:
        st.success(f'✅ Низкий риск сердечного заболевания: {1 - probability:.1%}. Вот это вы молодец!')