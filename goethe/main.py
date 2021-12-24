import pathlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def open_image(filename: str):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB ordering
    return img


def dominant_colors(image, n_clusters: int = 3):
    # convert image to list of pixel color values
    pixels = image.reshape((image.shape[0] * image.shape[1], 3))

    # find clusters
    km = KMeans(n_clusters=n_clusters).fit(pixels)

    # create histogram of color/cluster frequency
    # km.labels_ == what cluster each pixel belongs to
    histogram, _ = np.histogram(km.labels_, bins=n_clusters)
    histogram = histogram.astype("float")
    histogram = histogram / histogram.sum()

    # map colors to percent coverage
    return [
        (list(color), percent)
        for color, percent in zip(km.cluster_centers_, histogram)
    ]


def plot_colors(colors, height=100, width=600):
    bar = np.zeros((height, width, 3), dtype="uint8")
    start = 0

    for color, percent in colors:
        end = start + int(percent * width)
        cv2.rectangle(
            bar,
            (start, 0),
            (end, height),
            color,
            -1
        )
        start = end

    return bar

# run example

BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
filename = str(BASE_DIR / "examples" / "moon.jpg")

img = open_image(filename)
colors = dominant_colors(img)

# plot bar of color distribution
bar = plot_colors(colors)
plt.axis("off")
plt.imshow(bar)
plt.show()
