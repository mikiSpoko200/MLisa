# ml-project

# Ideas

### Idea 1:
1. Collect patches from the whole set (or some random fragment) of images.
2. Perform a K-means on the set of patches, getting K template patches (cluster centers).
3. Make histograms of template patches (counting each patch as its closest template patch) in images.

> Now that I'm reading it, it's not clear to me what point 3 is about? Don't we just need per label average histograms generated in step 4?

4. Make an average histogram for each style/artist or perform logistic regression on histograms (?)

> I don't see how logistic regression fits here? I see this as follows
>   - For each label take all images.
>   - Sample images and for each sample perform knn with regard to the set of reference patches generated in step 2.
>   - Generate histogram containing 

5. For each image to classify:
    - for each selected patch (e.g. through a sliding window, randomly):
        - count it as its closest template patch
    - make a histogram for this image out of counted template patches
    - classify the image as the label of average histogram closest to the just calculated histogram
    (or class of the calculated histogram, pointed by logistic regression)

We can use this idea first with 1x1 patches = pixels and their colors.
Colors can be grouped with K-means, so then we compare histograms of these 'main' colors.

### Idea 2 (more weird, probably not worth it):
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

### Functions we need

- collect_patches(image)
- make_a_histogram(image)

### Idea 3

**Training**

1. Sample images group sample sets by label

Image sampling process can be parametrized over:
- patch size - how many pixels are used 1x1? 1x10? 10x10? etc.
- pixel bit-depth - how many bits per whole pixel, color channel, give more resolution to certain channels, etc.
- coverage in % - based on patch size and image size we can determine total number of distinct samples:
  - take all possible samples (100%)
  - take 50%
  - take 10%
- sampling strategy - how to choose sampling locations:
  - random
  - uniform with given stride
  - only middle of the picture
  - etc.

Multiple configurations can be used to influence output of the classifier.

2. Feed groups into k-means
    
For each group generate representatives using k-means and store the results.

3. Post processing

We can arrange representative sets into data structures that could accelerate later queries. For example, we could measure match quality by taking 10 best matches with dot-product. Then we could use a BST or Interval tree. Perhaps some space partitioning method could also help.

**Prediction**

1. During classification sample given image using the same strategy as during training.
2. Compare each sample against the representative set of each label using configurable quality metric (eg. MSE, dot product, minimal ). Pick result for which configurable goal function is the smallest.

**Justification**

Generation of representative sets on a per label basis can better capture features specific to each label. Additionally we gain much reacher information regarding match quality then in just plain knn, we know exactly how good the match was for each specific label.

Additionally this approach can be almost entirely parallelized and would lend itself perticularlly well for implementation on massively parallel processing units, thus reducing the cost.

**extensions to consider**

Decision trees could be used to generate weights that capture probability of certain patch appearing in certain part of picture (eg. Caravaggio has lost of black in the background, portraits have usually uniform background and have some *skin-colored* patches in the middle)