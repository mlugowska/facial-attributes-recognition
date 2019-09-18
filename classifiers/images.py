import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from imutils import paths
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from classifiers.data_visualization import print_confusion_matrix

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


models = [
    ('Logistic Regression', LogisticRegression()),
    ('KNN', KNeighborsClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('SVM', SVC(kernel='linear')),
    ('Naive Bayes', GaussianNB()),
    ('MLP', MLPClassifier())
]

# grab all image paths in the input dataset directory, initialize our
# list of extracted features and corresponding labels
print('[INFO] extracting image features...')
image_paths = paths.list_images('/Users/mlugowska/PhD/applied_statistics/celeba-dataset/img_align_celeba')
data = []
labels = []

# loop over our input images
for image_path in image_paths:
    # load the input image from disk, compute color channel
    # statistics, and then update our data list
    image = Image.open(image_path)
    features = extract_color_stats(image)
    data.append(features)

# extract the class label from the file path and update the
# labels list
# label = imagePath.split(os.path.sep)[-1]
# labels.append(label)


# encode the labels, converting them from strings to integers
labs = pd.read_csv('/Users/mlugowska/PhD/applied_statistics/celeba-dataset/list_attr_celeba.csv', index_col=0)
labs.replace(to_replace=-1, value=0, inplace=True)
labels = labs.Male[0:10000]
le = LabelEncoder()
labels = le.fit_transform(labels)
# perform a training and testing split, using 75% of the data for
# training and 25% for evaluation
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)

# 3) Feature selection

## PCA

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

chosen_variance = [exp for exp in explained_variance if exp > 0.03]

pca = PCA(n_components=len(chosen_variance))
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

n_pcs = pca.components_.shape[0]
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
most_important_names = [X.columns[most_important[i]] for i in range(n_pcs)]
dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
df = pd.DataFrame(dic.items())

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_)
plt.xticks(features)
plt.yticks(features)
plt.xlabel('PCA features')
plt.ylabel('variance')

plt.semilogy(pca.explained_variance_ratio_, '--o')
plt.semilogy(pca.explained_variance_ratio_.cumsum(), '--o')
plt.show()

## RFE
lr = LinearRegression(normalize=True)
lr.fit(x_train, y_train)

# stop the search when only the last feature is left
rfe = RFE(lr, n_features_to_select=8, verbose=3)
rfe.fit(x_train, y_train)

selected_feature_names = x_train.columns[rfe.support_]
X_train = x_train[selected_feature_names]
X_test = x_test[selected_feature_names]

print(f'Number of features selected: {rfe.n_features_}')
print(f'Feature names: {selected_feature_names}')

## no selection
X_train = x_train
X_test = x_test

for name, model in models:
    start = time.time()
    clf = model.fit(X_train, y_train)
    end = time.time()
    print(f'Execution time for building the {name}: {float(end) - float(start)}')
    ac = accuracy_score(y_test, clf.predict(X_test))

    if hasattr(model, 'decision_function'):
        y_score = clf.decision_function(X_test)
    else:
        y_score = clf.predict_proba(X_test)

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    err = 1 - acc
    sens = (tp / (tp + fn))
    spec = (tn / (tn + fp))
    fig = print_confusion_matrix(confusion_matrix=cm, class_names=['Male', 'Female'], name=name)
    plt.savefig(f'/Users/mlugowska/PhD/applied_statistics/figs/images/no_feature_selection/{name}_confusion_matrix.png')

    msg = f'{name}: Accuracy - {ac}, Sensitivity - {sens}, Specifity - {spec}, Error - {err}'
    print(msg)

    if hasattr(model, 'decision_function'):
        fpr, tpr, thr = roc_curve(y_test, y_score)
    else:
        fpr, tpr, thr = roc_curve(y_test, y_score[:, 1])

    roc_auc = auc(fpr, tpr)
    figure = plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--', color='navy', lw=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(f'ROC Curve of {name}')
    plt.savefig(f'/Users/mlugowska/PhD/applied_statistics/figs/images/no_feature_selection/{name}.png')
