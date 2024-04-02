import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("CarPricesPrediction.csv")
print(df)

column_name = "Price"
plt.hist(df[column_name])
plt.show()

median = df[column_name].median()
mean = df[column_name].mean()
print("Медиана:", median)
print("Среднее значение:", mean)

plt.boxplot(df[column_name])
plt.show()

description = df[column_name].describe()
print(description)

grouped_data = df.groupby('Year')['Price'].mean().reset_index()
print(grouped_data)