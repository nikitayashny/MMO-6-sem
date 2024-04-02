import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import RocCurveDisplay

# Загрузка данных из файла heart.csv
data = pd.read_csv('heart.csv')

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.drop('output', axis=1)
y = data['output']

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('-------------- Logistic Regression --------------')

# Обучение модели логистической регрессии
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Оценка точности модели на обучающем и тестовом наборах
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print("Правильность на обучающем наборе: {:.2f}".format(train_accuracy))
print("Правильность на тестовом наборе: {:.2f}".format(test_accuracy))

# Изменение параметра регуляризации C
model_C100 = LogisticRegression(C=100, max_iter=10000)
model_C001 = LogisticRegression(C=0.01, max_iter=1000)

model_C100.fit(X_train, y_train)
model_C001.fit(X_train, y_train)

train_accuracy_C100 = model_C100.score(X_train, y_train)
test_accuracy_C100 = model_C100.score(X_test, y_test)

train_accuracy_C001 = model_C001.score(X_train, y_train)
test_accuracy_C001 = model_C001.score(X_test, y_test)

print("Правильность (C=100) на обучающем наборе: {:.2f}".format(train_accuracy_C100))
print("Правильность (C=100) на тестовом наборе: {:.2f}".format(test_accuracy_C100))
print("Правильность (C=0.01) на обучающем наборе: {:.2f}".format(train_accuracy_C001))
print("Правильность (C=0.01) на тестовом наборе: {:.2f}".format(test_accuracy_C001))

# Добавление L2-регуляризации
model_l2 = LogisticRegression(penalty='l2', C=0.1, max_iter=1000)
model_l2.fit(X_train, y_train)

train_accuracy_l2 = model_l2.score(X_train, y_train)
test_accuracy_l2 = model_l2.score(X_test, y_test)

print("Правильность (L2) на обучающем наборе: {:.2f}".format(train_accuracy_l2))
print("Правильность (L2) на тестовом наборе: {:.2f}".format(test_accuracy_l2))

# Расчет метрик качества и матрицы ошибок
y_pred = model_l2.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Метрики качества:")
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("Матрица ошибок:")
print(confusion_mat)

print('-------------- Метод опорных векторов (SVC) --------------')

# Обучение модели метода опорных векторов
model_SVC = SVC()
model_SVC.fit(X_train, y_train)

# Оценка точности модели на обучающем и тестовом наборах
train_accuracy_SVC = model_SVC.score(X_train, y_train)
test_accuracy_SVC = model_SVC.score(X_test, y_test)

print("Правильность на обучающем наборе: {:.2f}".format(train_accuracy_SVC))
print("Правильность на тестовом наборе: {:.2f}".format(test_accuracy_SVC))

# Подбор параметров с помощью GridSearchCV
SVC_params = {"C": [0.1, 1, 10, 100, 1000], "gamma": [0.0001, 0.001, 0.01, 0,1]}
SVC_grid = GridSearchCV(model_SVC, SVC_params, cv=5, n_jobs=-1)
SVC_grid.fit(X_train, y_train)

best_score = SVC_grid.best_score_
best_params = SVC_grid.best_params_

print("Наилучшая точность: {:.2f}".format(best_score))
print("Наилучшие параметры:", best_params)

# Оценка точности модели с наилучшими параметрами на обучающем и тестовом наборах
best_model = SVC(**best_params)
best_model.fit(X_train, y_train)

train_accuracy_best = best_model.score(X_train, y_train)
test_accuracy_best = best_model.score(X_test, y_test)

print("Правильность на обучающем наборе (с наилучшими параметрами): {:.2f}".format(train_accuracy_best))
print("Правильность на тестовом наборе (с наилучшими параметрами): {:.2f}".format(test_accuracy_best))

# Расчет метрик качества и матрицы ошибок
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
confusion_mat_best = confusion_matrix(y_test, y_pred_best)

print("Метрики качества:")
print("Accuracy: {:.2f}".format(accuracy_best))
print("Precision: {:.2f}".format(precision_best))
print("Recall: {:.2f}".format(recall_best))
print("Матрица ошибок:")
print(confusion_mat_best)

print('-------------- Модель дерева решений --------------')

# Обучение модели дерева решений
model_DT = DecisionTreeClassifier()
model_DT.fit(X_train, y_train)

# Оценка точности модели на обучающем и тестовом наборах
train_accuracy_DT = model_DT.score(X_train, y_train)
test_accuracy_DT = model_DT.score(X_test, y_test)

print("Правильность на обучающем наборе (дерево решений): {:.2f}".format(train_accuracy_DT))
print("Правильность на тестовом наборе (дерево решений): {:.2f}".format(test_accuracy_DT))

print('-------------- Модель K-ближайших соседей --------------')

# Обучение модели K-ближайших соседей
model_KNN = KNeighborsClassifier()
model_KNN.fit(X_train, y_train)

# Оценка точности модели на обучающем и тестовом наборах
train_accuracy_KNN = model_KNN.score(X_train, y_train)
test_accuracy_KNN = model_KNN.score(X_test, y_test)

print("Правильность на обучающем наборе (K-ближайшие соседи): {:.2f}".format(train_accuracy_KNN))
print("Правильность на тестовом наборе (K-ближайшие соседи): {:.2f}".format(test_accuracy_KNN))

# ROC-кривые
fig, ax = plt.subplots()

# Логистическая регрессия
lr_disp = RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name='Логистическая регрессия')

# Метод опорных векторов
svc_disp = RocCurveDisplay.from_estimator(model_SVC, X_test, y_test, ax=ax, name='Метод опорных векторов')

# Дерево решений
dt_disp = RocCurveDisplay.from_estimator(model_DT, X_test, y_test, ax=ax, name='Дерево решений')

# K-ближайших соседей
knn_disp = RocCurveDisplay.from_estimator(model_KNN, X_test, y_test, ax=ax, name='K-ближайших соседей')

# Отображение ROC-кривых
plt.legend(loc='lower right')
plt.show()