import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from classifiers.data_visualization import print_confusion_matrix

data = pd.read_csv('/Users/mlugowska/PhD/applied_statistics/celeba-dataset/list_attr_celeba.csv', index_col=0)

# NaN replacing
data.isnull().any()
data.info()
data.replace(0, np.nan, inplace=True)
data = data.replace(-1, 0)

# feature names as a list
col = data.columns  # .columns gives columns names in data
print(col)
col_labels = col.get_values()

# y includes our labels and x includes our features
y = data.Male  # 0 or 1

data_male = data['Male'].replace(0, 'Female')
data_male = data_male.replace(1, 'Male')
figure = plt.figure()
ax = sns.countplot(data_male, palette='Set3')

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                va='center', xytext=(0, 10), textcoords='offset points')

plt.savefig(f'/Users/mlugowska/PhD/applied_statistics/figs/gender_count.png')

Female, Male = y.value_counts()
print('Number of Male: ', Male)
print('Number of Female : ', Female)

X = data.drop(['Male'], axis=1)

models = [
    ('Logistic Regression', LogisticRegression()),
    ('KNN', KNeighborsClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('SVM', SVC(kernel='linear')),
    ('Naive Bayes', GaussianNB()),
    ('MLP', MLPClassifier())
]

# 1) Feature filtering (correlation)
f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.savefig(f'/Users/mlugowska/PhD/applied_statistics/figs/correlation.png')
plt.show()

# Correlation with output variable
cor_target = abs(data.corr()['Male'])
# Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.5]

# 2) Train test split
np.random.seed(1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
ax = plt.bar(features, pca.explained_variance_ratio_)
plt.xticks(features)
plt.yticks(features)
plt.ylim(0, 1)
plt.xlabel('PCA features')
plt.ylabel('variance')

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                va='center', xytext=(0, 10), textcoords='offset points')

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
    plt.savefig(f'/Users/mlugowska/PhD/applied_statistics/figs/attrs/RFE/{name}_confusion_matrix.png')

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
    plt.savefig(f'/Users/mlugowska/PhD/applied_statistics/figs/attrs/RFE/{name}_RFE.png')
