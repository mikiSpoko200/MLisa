import matplotlib as plt


def plot_image(x, size):
    plt.figure(figsize=(1.5, 1.5))
    plt.imshow(x.reshape(size, size, 3))
    plt.show()
    plt.close()

