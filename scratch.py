import numpy as np
#103.27757521981862   161.72645050977144  позиция с помощью линейной регрессии
signals =[1.0, -0.6558002937530238, -0.7959170302523281, -0.8857052157668401,
          -0.9499427781936607, -0.9678739408251552, -0.9749334964680662, -0.9825073367056923, -1.0]

sensors_coord = [[89, 176], [150, 143], [165, 77], [230, 97], [292, 76], [353, 100], [385, 75], [425, 99], [513, 159]]


arr = [4, 8, 2, 5, 1, 6, 7, 3]
max_indices = sorted(range(len(arr)), key=lambda i: arr[i])[-3:]
max_values = sorted(arr)[-3:]

print(max_indices)
print(max_values)

p1 = np.array([1, 2])
p2 = np.array([4, 5])
p3 = np.array([7, 8])

# расстояния от объекта до каждого датчика
d1 = 2
d2 = 3
d3 = 4
from scipy.spatial import Delaunay
# оценка позиции объекта с помощью триангуляции
points = np.array([p1, p2, p3], dtype='float')
tri = Delaunay(points)
p = np.array([d1, d2, d3])
b = np.ones((len(points), 1))
A = np.hstack((points - points[0, :], p[:, np.newaxis]))
x = np.linalg.lstsq(A, b, rcond=None)[0].flatten()
object_pos = points[0, :] + x[:2] * d1

print(object_pos)
