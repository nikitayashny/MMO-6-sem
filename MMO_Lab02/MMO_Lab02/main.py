import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('smartwatches.csv')

# Task 1
missing_values = data.isnull().sum()
print(missing_values)

plt.figure(figsize=(10, 6))
missing_values.plot(kind="bar")
plt.xlabel("Параметр")
plt.ylabel("Количество пропусков")
plt.title("Анализ пропусков данных")
plt.show()

# Task 4
plt.figure(figsize=(10, 5))
data["Original Price"].hist(bins=10)
plt.xlabel('Original Price')
plt.ylabel("Частота")
plt.title("Распределение цены (до обработки пропусков)")
plt.show()

# Task 2
column_threshold = round(len(data) * 0.5)
data = data.dropna(thresh=column_threshold, axis=1)

row_threshold = round(len(data.columns) * 0.5)
data = data.dropna(thresh=row_threshold)

# Task 3
# Замена пропусков в числовых столбцах медианными значениями
numeric_columns = data.select_dtypes(include='number').columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

# Замена пропусков в категориальных столбцах наиболее часто встречающимися значениями
categorical_columns = data.select_dtypes(include='object').columns
data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

# Task 4
plt.figure(figsize=(10, 5))
data["Original Price"].hist(bins=10)
plt.xlabel('Original Price')
plt.ylabel("Частота")
plt.title("Распределение цены (до обработки пропусков)")
plt.show()

# Task 6
label_encoder = LabelEncoder()
data_encoded_label = data.copy()
for column in categorical_columns:
    data_encoded_label[column] = label_encoder.fit_transform(data_encoded_label[column])

# Task 5
threshold = 2.5

for column in data_encoded_label.columns:
    if data_encoded_label[column].dtype != object:
        outliers = data_encoded_label[data_encoded_label[column] > data_encoded_label[column].mean() + threshold
                                      * data_encoded_label[column].std()]
        data_encoded_label = data_encoded_label.drop(outliers.index)

# Task 7
data_encoded_label.to_csv('processed_smartwatches.csv', index=False)