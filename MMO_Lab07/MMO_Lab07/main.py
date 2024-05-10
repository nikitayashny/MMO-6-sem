import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('heart.csv')

# Разделение на признаки и целевую переменную
X = data.drop('output', axis=1)
y = data['output']

# Задание 1: Обучение модели случайного леса
# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели случайного леса
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = rf.predict(X_test)

# Расчет точности
accuracy = rf.score(X_test, y_test)
print("Точность модели случайного леса:", accuracy)

# Задание 2: Сокращение параметров датасета с использованием Feature Selection
# Создание объекта для сокращения параметров
selector = VarianceThreshold(threshold=0.1)

# Применение сокращения параметров к данным
X_reduced = selector.fit_transform(X)

# Получение списка выбранных признаков
selected_features = X.columns[selector.get_support()]

# Обновление датасета с выбранными признаками
X_reduced = pd.DataFrame(X_reduced, columns=selected_features)

# Разделение на обучающую и тестовую выборки
X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Обучение модели случайного леса на сокращенном датасете
rf_reduced = RandomForestRegressor(random_state=42)
rf_reduced.fit(X_train_reduced, y_train)

# Предсказание на тестовой выборке
y_pred_reduced = rf_reduced.predict(X_test_reduced)

# Расчет точности на сокращенном датасете
accuracy_reduced = rf_reduced.score(X_test_reduced, y_test)
print("Точность модели случайного леса на сокращенном датасете:", accuracy_reduced)

# Задание 3: Применение метода PCA
# Применение метода PCA для получения 2 главных компонент
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Визуализация данных по двум компонентам
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Задание 4: Обучение модели случайного леса на данных PCA
# Разделение на обучающую и тестовую выборки
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Обучение модели случайного леса на данных PCA
rf_pca = RandomForestRegressor(random_state=42)
rf_pca.fit(X_train_pca, y_train)

# Предсказание на тестовой выборке
y_pred_pca = rf_pca.predict(X_test_pca)

# Расчет точности на данных PCA
accuracy_pca = rf_pca.score(X_test_pca, y_test)
print("Точность модели случайного леса на данных PCA:", accuracy_pca)

# Задание 5: Определение количества главных компонент, сохраняющего 90% дисперсии
# Применение метода PCA для определения количества главных компонент, сохраняющего 90% дисперсии
pca = PCA(n_components=0.9)
X_pca = pca.fit_transform(X)

# Визуализация графика зависимости отклонения модели от количества главных компонент
plt.plot(range(1, pca.n_components_ + 1), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs Number of Principal Components')
plt.show()

# Определение количества главных компонент, сохраняющего 90% дисперсии
n_components = pca.n_components_
print("Количество главных компонент для сохранения 90% дисперсии:", n_components)

# Задание 6: Обучение модели с определенным количеством компонент
# Применение метода PCA с определенным количеством компонент
pca_final = PCA(n_components=n_components)
X_pca_final = pca_final.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train_pca_final, X_test_pca_final, y_train, y_test = train_test_split(X_pca_final, y, test_size=0.2, random_state=42)

# Обучение модели случайного леса на данных PCA с определенным количеством компонент
rf_pca_final = RandomForestRegressor(random_state=42)
rf_pca_final.fit(X_train_pca_final, y_train)

# Предсказание на тестовой выборке
y_pred_pca_final = rf_pca_final.predict(X_test_pca_final)

# Расчет точности на данных PCA с определенным количеством компонент
accuracy_pca_final = rf_pca_final.score(X_test_pca_final, y_test)
print("Точность модели случайного леса на данных PCA с", n_components, "компонентами:", accuracy_pca_final)


# по примеру
# pca = decomposition.PCA().fit(X)
#
# plt.figure(figsize=(10,7))
# plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
# plt.xlabel('Number of components')
# plt.ylabel('Total explained variance')
# plt.xlim(0, 63)
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.axvline(21, c='b')
# plt.axhline(0.9, c='r')
# plt.show();