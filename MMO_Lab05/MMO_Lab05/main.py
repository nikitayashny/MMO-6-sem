import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Загрузка датасета
df = pd.read_csv('insurance.csv')

# Преобразование категориальных переменных с помощью One-Hot Encoding
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])

# Задание 2: Визуализация матрицы корреляций
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Матрица корреляций')
plt.show()

# Задание 3: Построение матрицы диаграмм рассеяния
sns.pairplot(df, vars=['age', 'bmi', 'children', 'charges'])
plt.show()

# Задание 5: Рассчет и визуализация модели простой линейной регрессии
feature = 'age'

X = df[feature].values.reshape(-1, 1)
y = df['charges'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

y_train_pred = model_simple.predict(X_train)
y_test_pred = model_simple.predict(X_test)

plt.scatter(X_test, y_test, color='blue', label='Наблюдения')
plt.plot(X_test, y_test_pred, color='red', linewidth=2, label='Прогноз')
plt.xlabel('Возраст')
plt.ylabel('Затраты')
plt.title('Модель простой линейной регрессии')
plt.legend()
plt.show()

# Задание 6: Оценка качества модели простой линейной регрессии
r2_simple = r2_score(y_test, y_test_pred)
mse_simple = mean_squared_error(y_test, y_test_pred)
mae_simple = mean_absolute_error(y_test, y_test_pred)

print("Модель простой линейной регрессии:")
print("R-квадрат (для тестовой выборки):", r2_simple)
print("MSE (для тестовой выборки):", mse_simple)
print("MAE (для тестовой выборки):", mae_simple)
print()

# Задание 7: Добавление дополнительных параметров в модель
features = ['age', 'bmi', 'children']  # Выбранные параметры

X_multi = df[features].values

X_train_multi, X_test_multi, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train_multi, y_train)

y_train_pred_multi = model_multi.predict(X_train_multi)
y_test_pred_multi = model_multi.predict(X_test_multi)

# Задание 8: Оценка качества модели с несколькими параметрами
r2_multi = r2_score(y_test, y_test_pred_multi)
mse_multi = mean_squared_error(y_test, y_test_pred_multi)
mae_multi = mean_absolute_error(y_test, y_test_pred_multi)

print("Модель с несколькими параметрами:")
print("R-квадрат (для тестовой выборки):", r2_multi)
print("MSE (для тестовой выборки):", mse_multi)
print("MAE (для тестовой выборки):", mae_multi)
print()

# Задание 9: Вывод
# Сравнивая модель простой линейной регрессии с моделью, включающей несколько параметров,
# можно сделать вывод о том, что модель с несколькими параметрами лучше описывает зависимость
# целевой переменной от параметров. Это можно судить по более высокому значению R-квадрат,
# а также более низким значениям MSE и MAE для модели с несколькими параметрами.
# Включение дополнительных параметров может улучшить предсказательную способность модели и
# помочь лучше объяснить вариабельность целевой переменной.
