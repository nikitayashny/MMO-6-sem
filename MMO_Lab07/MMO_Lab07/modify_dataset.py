import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("melb_data.csv")

missing_values = data.isnull().sum()
print(missing_values)

column_threshold = round(len(data) * 0.5)
data = data.dropna(thresh=column_threshold, axis=1)

row_threshold = round(len(data.columns) * 0.5)
data = data.dropna(thresh=row_threshold)

numeric_columns = data.select_dtypes(include='number').columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

categorical_columns = data.select_dtypes(include='object').columns
data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

label_encoder = LabelEncoder()
data_encoded_label = data.copy()
for column in categorical_columns:
    data_encoded_label[column] = label_encoder.fit_transform(data_encoded_label[column])

threshold = 2.5

for column in data_encoded_label.columns:
    if data_encoded_label[column].dtype != object:
        outliers = data_encoded_label[data_encoded_label[column] > data_encoded_label[column].mean() + threshold
                                      * data_encoded_label[column].std()]
        data_encoded_label = data_encoded_label.drop(outliers.index)

data_encoded_label.to_csv('processed_melb_data.csv', index=False)