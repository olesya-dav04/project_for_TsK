import streamlit as st
import analysis_and_model
import presentation

# Настройка навигации
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите страницу:",
    ("Анализ и модель", "Презентация")
)

if page == "Анализ и модель":
    analysis_and_model.main()
elif page == "Презентация":
    presentation.main()