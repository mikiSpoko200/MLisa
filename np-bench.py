import numpy as np

import loader
import utils
import timeit


def bench():

    counter = 0
    for feature_batch_iterator in loader.BatchLoader(utils.ClassificationTarget.GENRE):
        for batch in feature_batch_iterator:
            for image in batch:
                _array = np.frombuffer(image, dtype='B')
        counter += 1
        if counter > 3:
            break


if __name__ == '__main__':
    _ = loader.BatchLoader(utils.ClassificationTarget.GENRE)
    timeit.timeit(lambda: bench(), number=5)

