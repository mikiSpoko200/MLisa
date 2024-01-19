# ml-project

# Ideas

Idea 1:
1. Collect patches from the whole set (or some random fragment) of images.
2. Perform a K-means on the set of patches, getting K template patches (cluster centers).
3. Make histograms of template patches (counting each patch as its closest template patch) in images.
4. Make an average histogram for each style/artist or perform logistic regression on histograms (?)
5. For each image to classify:
    - for each selected patch (e.g. through a sliding window, randomly):
        - count it as its closest template patch
    - make a histogram for this image out of counted template patches
    - classify the image as the label of average histogram closest to the just calculated histogram
    (or class of the calculated histogram, pointed by logistic regression)

We can use this idea first with 1x1 patches = pixels and their colors.
Colors can be grouped with K-means, so then we compare histograms of these 'main' colors.

Idea 2 (more weird, probably not worth it):
1. For each label (style/artist):
    - select a set of patches (e.g. from 10% percent of images)
    - perform a K-means on the set of patches, getting K template patches (cluster centers)
This gives us a global set of |labels| * K template patches, which are labeled
2. For each image to classify:
    - for each selected patch (e.g. through a sliding window, randomly):
        - calculate distance (pixelwise, e.g. Euclidean) to every template patch in the global set
        - get k (!= K) closest ones and let kNN decide the label of the patch
    - classify the image with the label, which occured the most across all considered patches

Is this even sensible? Too expensive or not?

Note: we don't necessarily have to classify every style/artist,
we can at first learn to differentiate e.g. cubism from romanticism.

# Functions we need

- collect_patches(image)
- make_a_histogram(image)
