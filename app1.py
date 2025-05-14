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

# Функция для оценки модели
def evaluate_model(model, X_test, y_test):
     # Предсказания
     y_pred = model.predict(X_test)
     y_pred_proba = model.predict_proba(X_test)[:, 1] # Вероятности для  ROC-AUC

     # Метрики
     accuracy = accuracy_score(y_test, y_pred)
     conf_matrix = confusion_matrix(y_test, y_pred)
     class_report = classification_report(y_test, y_pred)
     roc_auc = roc_auc_score(y_test, y_pred_proba)

     # Вывод результатов
     print("Accuracy:", accuracy)
     print("Confusion Matrix:\n", conf_matrix)
     print("Classification Report:\n", class_report)
     print("ROC-AUC:", roc_auc)

     # Построение ROC-кривой
     fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
     plt.plot(fpr, tpr, label=f"{model.__class__.__name__} (AUC ={roc_auc:.2f})")



# Загрузка данных
data = pd.read_csv("predictive_maintenance.csv")

# Удаление ненужных столбцов
data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF','RNF'])

# Преобразование категориальной переменной Type в числовую
data['Type'] = LabelEncoder().fit_transform(data['Type'])

# Проверка на пропущенные значения
print(data.isnull().sum())

# Масштабирование числовых признаков
scaler = StandardScaler()
numerical_features = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
data[numerical_features] = scaler.fit_transform(data[numerical_features])
# Вывод первых строк данных после масштабирования
print(data.head())

# Признаки (X) и целевая переменная (y)
X = data.drop(columns=['Machine failure'])
y = data['Machine failure']
# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(data.head())

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
svm = SVC(kernel='linear', random_state=42, probability=True) # probability=True для ROC-AUC
svm.fit(X_train, y_train)



# Оценка Logistic Regression
print("Logistic Regression:")
evaluate_model(log_reg, X_test, y_test)
print("-------------------------------------")
# Оценка Random Forest
print("Random Forest:")
evaluate_model(rf, X_test, y_test)
print("-------------------------------------")
# Оценка XGBoost
print("XGBoost:")
evaluate_model(xgb, X_test, y_test)
print("-------------------------------------")
# Оценка SVM
print("SVM:")
evaluate_model(svm, X_test, y_test)
print("-------------------------------------")
# Визуализация ROC-кривых
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые')
plt.legend()
plt.show()