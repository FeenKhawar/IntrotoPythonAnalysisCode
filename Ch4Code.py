import numpy as np
import matplotlib.pyplot as plt

# data = np.array([[1.5, -0.1, 3], [0, -3, 6.5]])
# print(data)
# print(data * 10)
# print((data + data) * 10)
# print(data.shape, data.dtype)

# print(np.zeros((2, 5, 5)))

# arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
# arr2 = arr.astype(np.int32)
# print(list(arr2))

# arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# lower_dim_slice = arr2d[1, :2]
# print(lower_dim_slice, lower_dim_slice.shape, lower_dim_slice.dtype)

# names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])
# data = np.array([[4, 7], [0, 2], [-5, 6], [0, 0], [1, 2], [-12, -4], [3, 4]])
# print(names == "Bob")
# print(data[names == "Bob"])
# print(data[names == "Bob", 0])
# cond = (names == "Bob")
# print(data[~cond])

# arr3 = np.arange(32).reshape((8,4))
# print(arr3)
# print(arr3[[1, 5, 7, 2], [0, 3, 1, 2]]) # This is how to get specific values from a matrix that follow no pattern, a bit tricky to understand compared to everything else

# rng = np.random.default_rng(seed=12345)
# print(f"rng: {rng}")
# data = rng.standard_normal((2, 3))
# print(f"data: {data}")

# arr4 = ([ 4.5146, -8.1079, -0.7909, 2.2474, -6.718 , -0.4084, 8.6237])
# out = np.zeros_like(arr4)
# print(out)

# points = np.arange(-5, 5, 0.01) # 100 equally spaced points
# xs, ys = np.meshgrid(points, points)
# print(ys)
# z = np.sqrt(xs ** 2 + ys ** 2)
# print(z)
# plt.imshow(z, cmap=plt.cm.gray, extent=[-5, 5, -5, 5])
# plt.colorbar()
# plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
# plt.savefig("plot.png")  # Save the plot to a file

# arr5 = np.arange(10)
# print(arr5)
# np.save("test_array", arr5)
# print(np.load("test_array.npy") * 2)
# print(arr5 * 2)
# How to save and load arrays via NumPy

# arr6 = np.arange(6)
# np.savez("test_array_archive.npz", a=arr6)

# arr7 = np.load("test_array_archive.npz")
# print(arr7["a"])
# Different way to save and load arrays via NumPy, can save multiple arrays in an uncompressed archive this way