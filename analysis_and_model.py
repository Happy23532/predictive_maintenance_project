import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

def analysis_and_model_page():
    st.title("Анализ данных и модель")
 # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data = pd.read_csv("predictive_maintenance.csv")
        # Предобработка данных
        # Удаление ненужных столбцов
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        # Преобразование категориальной переменной Type в числовую
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

        # Проверка на пропущенные значения
        print(data.isnull().sum())
        # Разделение данных
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Создание и обучение модели Logistic Regression
        log_reg = LogisticRegression(class_weight='balanced')
        log_reg.fit(X_train, y_train)

        # Создание и обучение модели : Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Создание и обучение модели Gradient Boosting (XGBoost)
        xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb.fit(X_train, y_train)

        # Создание и обучение модели Support Vector Machine (SVM)
        svm = SVC(kernel='linear', random_state=42, probability=True)  # probability=True для ROC-AUC
        svm.fit(X_train, y_train)


        y_pred = log_reg.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        st.header("Результаты обучения модели")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
        st.subheader("Classification Report")
        st.text(classification_rep)
        # Интерфейс для предсказания
        st.header("Предсказание по новым данным")
        with st.form("prediction_form"):
            st.write("Введите значения признаков для предсказания:")
            productID = st.selectbox("productID", ["L", "M", "H"])
            air_temp = st.number_input("air temperature [K]")
            process_temp = st.number_input("process temperature [K]")
            rotational_speed = st.number_input("rotational speed [rpm]")
            torque = st.number_input("torque [Nm]")
            tool_wear = st.number_input("tool wear [min]")

