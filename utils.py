from PIL import Image
import numpy as np

BASIC_COLORS = np.array([[255,0,0], [255,128,0], [255,255,0], [128,255,0], 
                         [0,255,0], [0,255,128], [0,255,255], [0,128,255], 
                         [0,0,255], [128,0,255], [255,0,255], [255,0,128]])

def read_image(path):
    image = Image.open(path)
    return image

def compress_image(image, quality):
    image.save("compressed.jpg", optimize=True, quality = quality)
    compressed = Image.open("compressed.jpg")
    return compressed

def resize_image(image, height, width):
    image = image.resize((width, height))
    image.save("resized.jpg")
    resized = Image.open("resized.jpg")
    return resized

def color_histogram(image_array, colors):
    histogram = {color:0 for color in map(tuple, colors.tolist())}

    pixels = image_array.reshape(-1, 3) # shape = (h * w, 3)
    for pixel in pixels:
        closest_color_idx = np.sum(np.abs(colors - pixel), axis=1).argmin()
        histogram[tuple(colors[closest_color_idx])] += 1

    # histogram = np.array([cnt / pixels.shape[0] for cnt in histogram.values()])

    histogram = {color : cnt / pixels.shape[0] for color, cnt in histogram.items()}
    return histogram