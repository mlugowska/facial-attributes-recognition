import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import keras

from keras.preprocessing.image import img_to_array, load_img

from classifiers.facial_attribute_prediction.data_preprocessing import CelebA

celeba = CelebA(drop_features=[
    'Attractive',
    'Pale_Skin',
    'Blurry',
])

"""# 2. Load Model
---
* Load the model to infer facial features
"""

model = keras.models.load_model(
    f"/Users/mlugowska/PhD/applied_statistics/classifiers/facial_attribute_prediction/weights-FC37-MobileNetV2-0.92.hdf5")

# model.summary()

img_size = 224  # @param ["224", "192"] {type:"raw"}

# default is 224
IMG_W = img_size
IMG_H = img_size
IMG_SHAPE = (IMG_H, IMG_W, 3)
TARGET_SIZE = (IMG_H, IMG_W)

"""# 3. Preparing the Data
---
* Selecting the facial attributes
* Loading the data (from various sources)
"""

# ------------------------------------------------------------------------------
# -- Maps: feature_name -> index, and index -> feature_name
# ------------------------------------------------------------------------------

dict_feature_name_to_index = {name: i for i, name in enumerate(celeba.features_name)}
dict_index_to_feature_name = {v: k for k, v in dict_feature_name_to_index.items()}


def features_to_indexes(features_name):
    '''From a feature_name returns an index.
       Example: 'Bangs' -> 4
    '''
    indexes = []

    for name in features_name:
        indexes.append(dict_feature_name_to_index[name])

    return indexes


def indexes_to_features(feature_indexes):
    '''From a feature_index returns its name.
       Example: 4 -> 'Bangs'
    '''
    features_name = []

    for index in feature_indexes:
        features_name.append(dict_index_to_feature_name[index])

    return features_name


def image_paths_from_folder(folder, amount=-1):
    '''From a given folder returns an array of images' path.
       Set amount > 0 to limit the number of path taken.

       Supports images with extendsion: '.jpeg', '.jpg' and '.png'
    '''
    assert (folder is not None)

    paths = []
    count = 0

    for file in os.listdir(folder):
        # select only images
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            paths.append(os.path.join(folder, file))
            count += 1

            if count == amount:
                break

    return paths


def load_reshape_img(fname, shape):
    img = load_img(fname, target_size=shape)
    x = img_to_array(img) / 255.0
    x = x.reshape(x.shape)
    return x


def infere_labels(model, paths):
    '''Use the given model to predict the images features.
       The images are loaded from the given paths.

       Returns an array of array of features.
    '''
    assert (model is not None)
    # first load images
    image_batch = []

    for path in paths:
        img = load_reshape_img(path, TARGET_SIZE)
        image_batch.append(img)

    # predict labels: batch_size will handle large amount of images
    preds = model.predict(np.array(image_batch), batch_size=64, verbose=1)

    # convert labels to 0, 1 integers.
    preds = np.round(preds).astype('int')
    return preds


def dataframe_from_folder_or_labels(features_name, folder, labels=None, model=None, amount=-1):
    '''Organize images and feature-labes in a DataFrame.
         - If labels are given it will take that instead, it uses the model to
           infere the images's labels.
         - Use amount to limit the number of images (and so predictions)

       Returns a pandas data-frame indexed by 'image_path'.
    '''
    paths = image_paths_from_folder(folder, amount)
    labels = labels or infere_labels(model, paths)
    indexes = features_to_indexes(features_name)

    df = pd.DataFrame()
    df['image_path'] = paths
    df.set_index('image_path', inplace=True)

    # select features for every image
    for i, name in enumerate(features_name):
        column = []

        for j, features in enumerate(labels):
            column.append(features[indexes[i]])
        df[name] = column

    #   # take features
    img_features = []
    for features in labels:
        features_ = []

        for i, name in enumerate(features_name):
            features_.append(features[indexes[i]])
        img_features.append(features_)

    # set a column with the selected features
    df['features'] = img_features

    return df


# ------------------------------------------------------------------------------
# -- Attribute selection:
# ------------------------------------------------------------------------------

# we can pick any feature, as an example we select these:
chosen_features = [
   'Wearing_Lipstick',
   'Smiling',
   'No_Beard',
   'Heavy_Makeup',
   'Bald',
   'Male',
   'Young',
   'Eyeglasses',
   'Blond_Hair',
   'Wearing_Hat'
]

chose_features_indexes = features_to_indexes(chosen_features)

print(f"selected featrues: {chosen_features}")
print(f"features indexes: {chose_features_indexes}")

# ------------------------------------------------------------------------------
# -- Loading data from CelebA
# ------------------------------------------------------------------------------

# Pick 400 samples sampled from 2000 instances.
# Let the model infer the selected features

celeba_df = dataframe_from_folder_or_labels(chosen_features, celeba.images_folder, model=model, amount=2000)
celeba_df = celeba_df.sample(400, random_state=51)

# celeba_df.head()

# ------------------------------------------------------------------------------
# -- Loading data from LFW
# ------------------------------------------------------------------------------

# we do the same as before but with a different data-source:

# lfw_df = dataframe_from_folder_or_labels(chosen_features, "lfw", model=model, amount=500)
# lfw_df = lfw_df.sample(100, random_state=51)

# lfw_df.head()

# ------------------------------------------------------------------------------
# -- Merge data into one data-frame
# ------------------------------------------------------------------------------

# before doing clustering on these data, we need to merge every df:
# data_df = celeba_df.append(lfw_df)
data_df = celeba_df
# show 10 random samples
data_df.sample(10)

"""# 4. Clustering
---
* KMeans
* Plot clusters
* Summarize clusters
* Weighted Clustering
"""

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.cluster import SpectralClustering, Birch
from sklearn.metrics import silhouette_score
from math import sqrt


def score(method, features):
    print(f'Score with {n_clusters} clusters is {-method.inertia_}')


def silhouette(features, labels):
    print(f'silhouette_score: {silhouette_score(features, labels)} for {n_clusters} clusters')


def resize(image, shape):
    return cv2.resize(image, shape)


def imread(path, shape=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if shape is not None:
        img = resize(img, shape)

    return img


def labels_to_indices(labels):
    '''Convert a list of cluster-labels into a dict of indices organized by
       cluster-id, including the noise claster (the one -1 labelled).

       Example:
         - labels = [0, 1, 1, 0, 2, 2]
         - returns -> { 0: [0, 3], 1: [1, 2], 2: [4, 5] }
    '''
    indices = dict()

    for cluster_id in set(labels):
        indices[cluster_id] = []

    for i, label in enumerate(labels):
        indices[label].append(i)

    return indices


# ------------------------------------------------------------------------------
# -- Cluster and Clustering utility class
# ------------------------------------------------------------------------------

class Cluster:
    '''A Single cluster with:
        - k: as cluster identifier,
        - features: of the items inside this cluster,
        - paths: of the images related to the items in the cluster.
    '''

    def __init__(self, k, features, paths):
        if len(features) != len(paths):
            raise ValueError("Size of [features] and [paths] parameters must be the same!")

        self.k = k
        self.features = features
        self.paths = paths
        self.size = len(paths)
        self.image = None
        self.eigenface = None

    def get_image(self, img_size=200, rows=None, cols=None):
        '''Returns an image that represents all cluster's items.
            - It caches the returned image for later reuse.
            - The returned image is in RGB format.
        '''
        if self.image is not None:
            return self.image

        if (rows is None) or (cols is None):
            rows = int(sqrt(self.size))
            cols = rows

        h, w = img_size, img_size
        image = np.zeros(shape=(h * rows, h * cols, 3), dtype=np.uint8)

        k = 0
        for i in range(rows):
            r = i * h
            for j in range(cols):
                c = j * w
                img = imread(self.paths[k], shape=(w, h))
                image[r:r + w, c:c + w] = img

                k = k + 1
                if k > self.size:
                    break

        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.image

    def plot(self, img_size=200, dpi=100, rows=None, cols=None):
        '''Visualize the images in this cluster.
            - img_size: specify the squared-size of a single image.
            - dpi: resolution of the plt.figure plot.
            - rows: amount of images in row.
            - cols: amount of images on columns.
        '''
        image = self.get_image(img_size, rows, cols)

        plt.figure(dpi=dpi)
        plt.imshow(image)
        plt.grid(False)

    def __len__(self):
        return self.size

    def features_frequency(self):
        '''Compute the frequency of every feature according to the items
           within the cluster.
        '''
        num = len(self.features[0])
        frequencies = np.zeros(num)

        for features in self.features:
            frequencies += features

        return np.round(frequencies / self.size, 2)

    def get_eigenface(self, w=200, h=200, components=None):
        '''Computes the average-face of this Cluster'''
        if self.eigenface is not None:
            return self.eigenface

        if components is None:
            components = self.size

        # load flat images into a matrix
        data = np.zeros((self.size, h * w * 3), dtype=np.float32)

        for i in range(self.size):
            img = imread(self.paths[i], shape=(h, w))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
            data[i, :] = img.flatten()

        # get eigen-vectors from the images
        mean, eigenvectors = cv2.PCACompute(data, mean=None, maxComponents=components)

        self.eigenface = mean.reshape((h, w, 3))
        return self.eigenface


class Clustering:
    '''Helpers for easy clustering'''

    def __init__(self, Method):
        self.method = Method
        self.result = None
        self.features = None
        self.labels = None
        self.num_clusters = None

    def fit(self, df, features_weights=None, verbose=False):
        '''Fits the given dataframe [df] using the given [Method].
            - The df dataframe must have an 'image_path' and 'features' columns.
            - The Method can be whatever instance of: KMeans, DBScan, etc.
            - feature_weights: can be used to give more/less importance to features.

           Returns an array of k Cluster instances.
        '''
        clusters = []
        features = df['features'].to_list()
        features = np.array(features)

        if features_weights is None:
            features_weights = np.ones(features.shape, dtype=np.float32)

        # fit features (viewed as n-dimensional points)
        result = self.method.fit(features * features_weights)

        # get clustering info
        labels = result.labels_
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_amount = list(labels).count(-1)  # count the occurrencies of '-1'

        # save as last fit
        self.result = result
        self.labels = labels
        self.features = features
        self.num_clusters = num_clusters

        if verbose is True:
            print(f'Estimated number of clusters: {num_clusters}')
            print(f'Estimated number of noise points: {noise_amount}')

        # for every cluster, get the indexes of the belonging samples
        indices = labels_to_indices(labels)

        # build Cluster istances
        for k in range(num_clusters):
            # for every cluster k find the corresponding image paths and features
            df_slice = df.iloc[indices[k]]
            paths = list(df_slice.index)
            features = list(df_slice['features'].values)

            clusters.append(Cluster(k, features, paths))

        return clusters

    def evaluate(self):
        '''Evaluate the lastly done clustering.
           It returns the silhouette_score og the clustering
        '''
        if self.result is None:
            raise ValueError("Fit data before evaluation!")

        return silhouette_score(self.features, self.labels)

    def plot(clusters, rows, cols, dpi=100):
        '''Shows the given clusters.
            - rows: number of rows in the subplot.
            - cols: nuber of columns in the subplot.
            - dpi: plt.figure resolution
        '''
        plt.figure(figsize=(20, 20), dpi=dpi)

        for i, cluster in enumerate(clusters):
            image = cluster.get_image()

            plt.subplot(rows, cols, i + 1)
            plt.title(f"Cluster {cluster.k}")
            plt.imshow(image)
            plt.grid(False)
            plt.tight_layout()

        _ = plt.show()

    def plot_with_frequencies(clusters, feature_names, dpi=100, figsize=(25, 25)):
        '''Shows the given clusters with the frequencies of the features within.
            - dpi: plot resolution (increase for a bigger plot)
        '''
        plt.figure(figsize=figsize, dpi=dpi)
        cols = 2
        rows = len(clusters)

        k = 1
        for i, cluster in enumerate(clusters):
            image = cluster.get_image()
            freqs = cluster.features_frequency()

            # cluster image
            plt.subplot(rows, cols, k)
            plt.title(f"Cluster {cluster.k}")
            plt.imshow(image)
            plt.grid(False)

            # frequency barplot
            plt.subplot(rows, cols, k + 1)
            plt.barh(feature_names, freqs, align='center')
            plt.xlabel('Features')
            plt.ylabel('Frequencies')
            plt.grid(False)
            plt.tight_layout()
            k += 2

        _ = plt.show()

    def plot_with_eigenfaces(clusters, dpi=100):
        '''Shows the given clusters along the cluster mean-face (eigenface).
             - dpi: plot resolution (increase for a bigger plot)
        '''
        plt.figure(figsize=(25, 25), dpi=dpi)
        cols = 2
        rows = len(clusters)

        k = 1
        for i, cluster in enumerate(clusters):
            image = cluster.get_image()
            eigenface = cluster.get_eigenface()

            # cluster image
            plt.subplot(rows, cols, k)
            plt.title(f"Cluster {cluster.k}")
            plt.imshow(image)
            plt.grid(False)

            # cluster eigenface
            plt.subplot(rows, cols, k + 1)
            plt.title(f"EigenFace {cluster.k}")
            plt.imshow(eigenface)
            plt.grid(False)
            k += 2

        _ = plt.show()

    def plot_all_stats(clusters, feature_names, dpi=100, figsize=(25, 25)):
        '''Shows the given clusters along with the:
            - features frequency of every cluster, and
            - eigenface of every cluster.
        '''
        cols = 3
        rows = len(clusters)

        plt.figure(figsize=figsize, dpi=dpi)
        gs1 = gridspec.GridSpec(rows, cols)
        gs1.update(wspace=0, hspace=0)

        k = 1
        for i, cluster in enumerate(clusters):
            image = cluster.get_image()
            eigenface = cluster.get_eigenface()
            freqs = cluster.features_frequency()

            # cluster image
            ax1 = plt.subplot(rows, cols, k)
            plt.title(f"Cluster {cluster.k}")
            plt.imshow(image)
            plt.grid(False)
            ax1.set_aspect('equal')

            # cluster eigenface
            ax2 = plt.subplot(rows, cols, k + 1)
            plt.title(f"EigenFace {cluster.k}")
            plt.imshow(eigenface)
            plt.grid(False)
            ax2.set_aspect('equal')

            # frequency barplot
            ax3 = plt.subplot(rows, cols, k + 2)
            plt.barh(feature_names, freqs, align='center')
            plt.xlabel('Features')
            plt.ylabel('Frequencies')
            plt.grid(False)
            plt.tight_layout()
            k += 3

        _ = plt.show()


# @title Clustering Parameters
num_clusters = 16  # @param {type:"integer"}

# ------------------------------------------------------------------------------
# -- KMeans
# ------------------------------------------------------------------------------

kmeans = KMeans(n_clusters=num_clusters)
kmeans = Clustering(kmeans)

clusters = kmeans.fit(data_df, verbose=True)

# prints the goodness of the clustering (values close to 1 are better)
print(f"Clustering score: {kmeans.evaluate()}")

# ------------------------------------------------------------------------------
# -- Plot Clusters
# ------------------------------------------------------------------------------

Clustering.plot(clusters, rows=4, cols=4, dpi=70)

# ------------------------------------------------------------------------------
# -- More insight on Clustering
# ------------------------------------------------------------------------------

# plots: cluster image, attribute occurrences-chart, cluster eigenface
Clustering.plot_all_stats([clusters[8]], chosen_features, dpi=60, figsize=(24, 8))
›
# 1. The eigenface summarize the most common attributes in the cluster, by
#    producing a synthesized face.

# 2. The attribute-occurrences chart shows the frequency of the single attributes.

# ------------------------------------------------------------------------------
# -- CelebA Attributes frequency
# ------------------------------------------------------------------------------

# pre-computed attributes frequency

# the same result can be obtained by this statement (but it take some time)
attributes_frequency = celeba.attributes.mean(axis=0).values


# compute weights for rare and common occurring attributes
weights_rare_features = 1 - attributes_frequency + min(attributes_frequency)
weights_common_features = attributes_frequency + (1 - max(attributes_frequency))

print(f"rare weights:\n {weights_rare_features}\n")
print(f"common weights:\n {weights_common_features}")

# ------------------------------------------------------------------------------
# -- Weighted Clustering: giving more importance to rare-occurring attributes
# ------------------------------------------------------------------------------

# get weights
rare_weights = []

for index in features_to_indexes(chosen_features):
    rare_weights.append(weights_rare_features[index])

# clustering:
rare_kmeans = Clustering(KMeans(n_clusters=num_clusters))
rare_clusters = rare_kmeans.fit(data_df, features_weights=rare_weights)

# plot first 4 clusters:
Clustering.plot_all_stats(rare_clusters[:4], chosen_features, dpi=60, figsize=(80, 20))

# ------------------------------------------------------------------------------
# -- Weighted Clustering: giving more importance to common-occurring attributes
# ------------------------------------------------------------------------------

# get weights
common_weights = []

for index in features_to_indexes(chosen_features):
    common_weights.append(weights_common_features[index])

# clustering:
common_kmeans = Clustering(KMeans(n_clusters=num_clusters))
common_clusters = common_kmeans.fit(data_df, features_weights=common_weights)

# plot first 4 clusters:
Clustering.plot_all_stats(common_clusters[:4], chosen_features, dpi=60, figsize=(80, 20))

"""# 5. Methods Comparison
---
* Fixed-clusters (DBSCAN) vs variable-clusters (KMeans, Agglomerative, Spectral, Birch)
"""

range_clusters = range(2, 50)

score = {
    'K-Means': [],
    'Agglomerative': [],
    'Birch': [],
    'Spectral': [],
    'DBSCAN': [],
}

# ------------------------------------------------------------------------------
# -- KMeans Performance
# ------------------------------------------------------------------------------

for k in range_clusters:
    # fit data for k clusters
    kmeans = Clustering(KMeans(n_clusters=k, n_init=20, max_iter=500))
    kmeans.fit(data_df)

    # evaluate clustering through silhouette score
    score['K-Means'].append(kmeans.evaluate())

# ------------------------------------------------------------------------------
# -- AgglomerativeClustering Performance
# ------------------------------------------------------------------------------

for k in range_clusters:
    # fit data for k clusters
    agglomerative = Clustering(AgglomerativeClustering(n_clusters=k,
                                                       affinity='manhattan',
                                                       linkage='complete'))
    agglomerative.fit(data_df)

    # evaluate clustering through silhouette score
    score['Agglomerative'].append(agglomerative.evaluate())

# ------------------------------------------------------------------------------
# -- SpectralClustering Performance
# ------------------------------------------------------------------------------

for k in range_clusters:
    # fit data for k clusters
    spectral = Clustering(SpectralClustering(n_clusters=k))
    spectral.fit(data_df)

    # evaluate clustering through silhouette score
    score['Spectral'].append(spectral.evaluate())

# ------------------------------------------------------------------------------
# -- Birch Performance
# ------------------------------------------------------------------------------

for k in range_clusters:
    # fit data for k clusters
    birch = Clustering(Birch(n_clusters=k, threshold=0.36))
    birch.fit(data_df)

    # evaluate clustering through silhouette score
    score['Birch'].append(birch.evaluate())

# ------------------------------------------------------------------------------
# -- DBSCAN Performance
# ------------------------------------------------------------------------------

# DBSCAN
dbscan = Clustering(DBSCAN(eps=.5, min_samples=3))
dbscan.fit(data_df)

score['DBSCAN'].append(dbscan.evaluate())
#
# # ------------------------------------------------------------------------------
# # -- OPTICS Performance
# # ------------------------------------------------------------------------------

# # OPTICS
# optics = Clustering(OPTICS(metric='hamming'))
# optics.fit(data_df)
#
# score['OPTICS'].append(optics.evaluate())

# ------------------------------------------------------------------------------
# -- Plot Comparison
# ------------------------------------------------------------------------------

x = list(range_clusters)

plt.figure(figsize=(12, 12))
plt.title('Silhouette-score Comparison')

# variable k methods
plt.plot(x, score['K-Means'])
plt.plot(x, score['Agglomerative'])
plt.plot(x, score['Spectral'])
plt.plot(x, score['Birch'])

# fixed k methods
plt.scatter([dbscan.num_clusters], score['DBSCAN'], marker='s', c='g')
# plt.scatter([optics.num_clusters], score['OPTICS'], marker='s', c='k')

plt.legend(score.keys())
plt.savefig(f'/Users/mlugowska/PhD/applied_statistics/figs/cluster_comparison.png')

_ = plt.show()
