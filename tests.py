from utils import *
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

path = "D:/wikiart/Abstract_Expressionism/mark-tobey_untitled-2.jpg"

image = read_image(path)

# image = compress_image(image, quality=1)
# image = resize_image(image, 200, 100)
image.show()

image_array = np.asarray(image)

pixels = image_array.reshape(-1, 3)

kmeans = KMeans(max_iter = 5, n_clusters = 12, init='k-means++', n_init='auto')
kmeans.fit(pixels)

main_colors = kmeans.cluster_centers_
print(main_colors)

histogram = color_histogram(image_array, main_colors)
# histogram = color_histogram(image_array, BASIC_COLORS)
print(histogram)

# ax = pd.DataFrame(histogram, index=['freq']).plot(kind="bar")
# ax.set_yticks(np.arange(0, 1.1, 0.1))
# plt.show()