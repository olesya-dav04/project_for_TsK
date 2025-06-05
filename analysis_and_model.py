import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

def main():
    st.title("Анализ данных и модель")

    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")

    # Используем session_state для хранения данных и моделей
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    if uploaded_file is not None:
        # Сброс состояния, если загружен новый файл
        if st.session_state.data_loaded:
            st.session_state.models = {}
            st.session_state.data_loaded = False

        data = pd.read_csv(uploaded_file)
        
        # ---- Предобработка данных ----
        # Удаляем ненужные столбцы (TWF, HDF, PWF, OSF, RNF)
        columns_to_drop = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        existing_columns = [col for col in columns_to_drop if col in data.columns]
        data = data.drop(columns=existing_columns)

        # Кодируем категориальный признак 'Type'
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

        # Масштабирование числовых признаков
        scaler = StandardScaler()
        numerical_features = [
            'Air temperature', 
            'Process temperature', 
            'Rotational speed', 
            'Torque', 
            'Tool wear'
        ]

        # Проверка наличия столбцов
        missing_cols = [col for col in numerical_features if col not in data.columns]
        if missing_cols:
            st.error(f"Столбцы отсутствуют: {missing_cols}")
        else:
            data[numerical_features] = scaler.fit_transform(data[numerical_features])

        # Разделение данных
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Обучение моделей
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "XGBoost": XGBClassifier()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            st.session_state.models[name] = model  # Сохраняем модель в session_state

        st.session_state.data_loaded = True
        st.success("Данные загружены и модель обучена!")

    # Если данные загружены, показываем результаты
    if st.session_state.data_loaded:
        st.header("Результаты обучения")
        for name, model in st.session_state.models.items():
            y_pred = model.predict(X_test)
            st.subheader(f"Модель: {name}")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            
            # Confusion Matrix
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

        # Интерфейс для предсказания
        st.header("Предсказание в реальном времени")
        with st.form("prediction_form"):
            air_temp = st.number_input("Температура воздуха", value=300.0)
            process_temp = st.number_input("Температура процесса", value=310.0)
            rotational_speed = st.number_input("Скорость вращения", value=1500)
            torque = st.number_input("Крутящий момент", value=40.0)
            tool_wear = st.number_input("Износ инструмента", value=100)
            product_type = st.selectbox("Тип продукта", ["L", "M", "H"])
            
            if st.form_submit_button("Предсказать"):
                # Преобразование категориального признака
                type_encoded = 0 if product_type == "L" else 1 if product_type == "M" else 2
                
                input_data = pd.DataFrame({
                    'Type': [type_encoded],
                    'Air temperature': [air_temp],
                    'Process temperature': [process_temp],
                    'Rotational speed': [rotational_speed],
                    'Torque': [torque],
                    'Tool wear': [tool_wear]
                })
                
                # Масштабирование
                input_data[numerical_features] = scaler.transform(input_data[numerical_features])
                
                # Предсказание
                model = st.session_state.models["Random Forest"]  # Используем лучшую модель
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1]
                
                st.success(f"Предсказание: {'- Отказ -' if prediction == 1 else '+ Нет отказа +'}")
                st.write(f"Вероятность отказа: {probability:.2%}")

    else:
        st.warning("Загрузите CSV-файл через форму выше, чтобы начать анализ.")

if __name__ == "__main__":
    main()