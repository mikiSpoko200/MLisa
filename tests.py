from match import *
from palette import generate_global_palette

# TODO: test on a folder/whole dataset
path = "D:/wikiart/Abstract_Expressionism/mark-tobey_untitled-2.jpg"

image = read_image(path)
image.show()

image_array = np.asarray(image)
patches = get_patches(image_array, 1.0, False)

images = [image_array]
palette = generate_global_palette(images, 1000)

# version 1
histogram = match1(image_array, palette, 0.1, False)
print(histogram)

# version 2
# distances, neighbors = match2(image_array, palette, 0.1, False, 3)
