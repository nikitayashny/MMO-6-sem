import numpy as np

array = np.random.randint(0, 10,(4,5))

subarray1 = array[:2, :]
subarray2 = array[2:, :]

specified_values = 6
indices = np.where(subarray1 == specified_values)

count = len(indices[0])

print("\nКоличество найденных элементов:", count)