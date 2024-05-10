import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # из-за того что в системе какая то ошибка не даёт ядра посчитать

# Шаг 1: Загрузка данных и формирование матрицы X
data = pd.read_csv('Country-data.csv')
X = data[['child_mort', 'income', 'gdpp']]

# Шаг 2: Проверка на пропуски и кодирование категориальных данных (если есть)
print(X.isnull().sum())

# Шаг 3: Нормализация значений в матрице X с использованием MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Шаг 4: Определение оптимального количества кластеров с помощью метода локтя
inertia = []
k_values = range(2, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Визуализация метода локтя
plt.plot(k_values, inertia, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Выбор оптимального количества кластеров
optimal_k = 4

# Кластеризация методом K-means
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Шаг 5: Визуализация результатов кластеризации
# Выбор двух параметров для визуализации
param1 = 'child_mort'
param2 = 'income'

plt.scatter(X[param1], X[param2], c=kmeans_labels)
plt.xlabel(param1)
plt.ylabel(param2)
plt.title('K-means Clustering')
plt.show()

# Шаг 6: Кластеризация методом иерархической кластеризации
hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# Шаг 7: Визуализация результатов иерархической кластеризации
linkage_matrix = linkage(X_scaled, method='ward')  # Вычисление матрицы связей
dendrogram(linkage_matrix)
plt.xlabel(param1)
plt.ylabel(param2)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

# Шаг 8: Оценка качества кластеризации
silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
silhouette_hierarchical = silhouette_score(X_scaled, hierarchical_labels)
print(f'Silhouette Score (K-means): {silhouette_kmeans}')
print(f'Silhouette Score (Hierarchical): {silhouette_hierarchical}')

# Шаг 9: Визуализация выбранного объекта в виде точки на графике кластеров
# Выбор любого конкретного объекта (страны)
chosen_country = 'Brazil'
chosen_country_data = X.loc[data['country'] == chosen_country]
chosen_country_cluster = kmeans.predict(scaler.transform(chosen_country_data))
plt.scatter(X[param1], X[param2], c=kmeans_labels)
plt.scatter(chosen_country_data[param1], chosen_country_data[param2], c='red', s=100, label=chosen_country)
plt.xlabel(param1)
plt.ylabel(param2)
plt.title('K-means Clustering with Chosen Country')
plt.legend()
plt.show()