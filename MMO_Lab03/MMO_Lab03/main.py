# https://www.kaggle.com/code/nikitayashny/heart-attack-analysis-predict-806302/edit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import export_text

data = pd.read_csv("heart.csv")

# Выделение меток У и матрицы признаков Х
y = data["output"]
X = data.drop("output", axis=1)
print(X.head())
print(y.head())

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Inputs X (train/test):", X_train.shape, X_test.shape)
print("Outputs Y (train/test):", y_train.shape, y_test.shape)

# Создание и обучение моделей
dt_model = DecisionTreeClassifier(max_depth=5, random_state=0)
knn_model = KNeighborsClassifier()

dt_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

print("Правильность на обучающем наборе Decision Tree: {:.3f}".format(dt_model.score(X_train, y_train)))
print("Правильность на тестовом наборе Decision Tree: {:.3f}".format(dt_model.score(X_test, y_test)))

print("Правильность на обучающем наборе для K-Nearest Neighbors: {:.3f}".format(knn_model.score(X_train, y_train)))
print("Правильность на тестовом наборе для K-Nearest Neighbors: {:.3f}".format(knn_model.score(X_test, y_test)))

# Рассчет точности моделей
dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))
knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test))

print("Accuracy score Decision Tree: ", dt_accuracy)
print("Accuracy score K-Nearest Neighbors: ", knn_accuracy)

# Рассчет матрицы ошибок (confusion matrix)
dt_confusion_matrix = confusion_matrix(y_test, dt_model.predict(X_test))
knn_confusion_matrix = confusion_matrix(y_test, knn_model.predict(X_test))

# Выбор лучшей модели
if dt_accuracy > knn_accuracy:
    best_model = dt_model
    best_model_name = "Decision Tree"
else:
    best_model = knn_model
    best_model_name = "K-Nearest Neighbors"

print("Лучшая модель:", best_model_name)
print("Точность лучшей модели:", accuracy_score(y_test, best_model.predict(X_test)))
print("Матрица ошибок лучшей модели:")
print(confusion_matrix(y_test, best_model.predict(X_test)))

# Получение текстового представления модели дерева решений
tree_text = export_text(dt_model, feature_names=list(X.columns))

# Сохранение текстового представления в файл
with open("decision_tree.txt", "w") as file:
    file.write(tree_text)