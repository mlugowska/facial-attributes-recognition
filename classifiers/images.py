# import the necessary packages
import argparse
import random

import numpy as np
import pandas as pd
from PIL import Image
from imutils import paths
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

random.seed(200)


def extract_color_stats(image):
    # split the input image into its respective RGB color channels
    # and then create a feature vector with 6 values: the mean and
    # standard deviation for each of the 3 channels, respectively
    (R, G, B) = image.split()
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
                np.std(G), np.std(B)]

    # return our set of features
    return features


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="/Users/agatamuszynska/Downloads/celeba-dataset/Subset/",
                help="path to directory containing the '3scenes' dataset")
ap.add_argument("-m", "--model", type=str, default="mlp",
                help="type of python machine learning model to use")
args = vars(ap.parse_args())
# define the dictionary of models our script can use, where the key
# to the dictionary is the name of the model (supplied via command
# line argument) and the value is the model itself
models = {
    "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs"),
    "svm": SVC(kernel="linear"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "mlp": MLPClassifier()
}

# grab all image paths in the input dataset directory, initialize our
# list of extracted features and corresponding labels
print("[INFO] extracting image features...")
imagePaths = paths.list_images(args["dataset"])
data = []
labels = []

# loop over our input images
for imagePath in imagePaths:
    # load the input image from disk, compute color channel
    # statistics, and then update our data list
    image = Image.open(imagePath)
    features = extract_color_stats(image)
    data.append(features)

# extract the class label from the file path and update the
# labels list
# label = imagePath.split(os.path.sep)[-1]
# labels.append(label)


# encode the labels, converting them from strings to integers
labs = pd.read_csv("/Users/agatamuszynska/Downloads/celeba-dataset/list_attr_celeba.csv", index_col=0)
labs.replace(to_replace=-1, value=0, inplace=True)
labels = labs.Male[0:10000]
le = LabelEncoder()
labels = le.fit_transform(labels)
# perform a training and testing split, using 75% of the data for
# training and 25% for evaluation
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25)

# train the model
print("[INFO] using '{}' model".format(args["model"]))
model = models[args["model"]]
model.fit(trainX, trainY)
print("test accuracy {}".format(model.score(testX, testY)))
