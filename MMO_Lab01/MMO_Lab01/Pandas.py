import numpy as np
import pandas as pd

numpy_array = np.array([10, 20, 30, 40, 50])
series = pd.Series(numpy_array)

addition = series + 5
subtraction = series - 10
multiplication = series * 2
division = series / 3

numpy_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df = pd.DataFrame(numpy_array)

df.columns = ['A', 'B', 'C']

df = df.drop(1)

df = df.drop('B', axis=1)

print("Размер DataFrame:", df.shape)

specified_value = 3
indices = np.where(df == specified_value)

print("\nDataFrame:")
print(df)
print("\nИндексы найденных значений:", indices)
