import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from classifiers.data_visualization import print_confusion_matrix

data = pd.read_csv("/Users/mlugowska/PhD/applied_statistics/celeba-dataset/list_attr_celeba.csv", index_col=0)

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
ax = sns.countplot(data_male, palette="Set3")

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                va='center', xytext=(0, 10), textcoords='offset points')

plt.savefig(f'/Users/mlugowska/PhD/applied_statistics/figs/gender_count.png')

Female, Male = y.value_counts()
print('Number of Male: ', Male)
print('Number of Female : ', Female)

X = data.drop(["Male"], axis=1)
########## CLASSIFIERS PREPARATION

models = [
    ('Logistic Regression', LogisticRegression()),
    ('KNN', KNeighborsClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('SVM', SVC(kernel='linear')),
    ('Naive Bayes', GaussianNB()),
    ("MLP", MLPClassifier())
]

# 1) Feature filtering (correlation)
f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.savefig(f'/Users/mlugowska/PhD/applied_statistics/figs/correlation.png')
plt.show()

# Correlation with output variable
cor_target = abs(data.corr()["Male"])
# Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.5]

# 2) Models quality without feature selection
np.random.seed(1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 3) Feature selection method 1


def print_best(scores):
    scores = sorted(scores, reverse=True)

    print("The 5 best features selected by this method are :")
    for i in range(5):
        print(scores[i][1])

## Univariate Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

test = SelectKBest(score_func=chi2, k=2)
test.fit(x_train, y_train)

scores_kbest = []
for i in range(len(col) - 1):
    score = test.scores_[i]
    scores_kbest.append((score, x_train.columns[i]))

print_best(scores_kbest)

## RFE
lr = LinearRegression(normalize=True)
rfe = RFE(lr, n_features_to_select=1)
rfe.fit(x_train, y_train)

scores_rfe = []
for i in range(len(col) - 1):
    scores_rfe.append((rfe.ranking_[i], x_train.columns[i]))

print_best(scores_rfe)

## Feature importance

rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
rf.fit(x_train, y_train)

scores_fi = []
for i in range(len(col) - 1):
    scores_fi.append((rf.feature_importances_[i], x_train.columns[i]))

print_best(scores_fi)

## Feature selection method 2

# Define dictionary to store our rankings
ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

test = SelectKBest(score_func=chi2, k=2)
test.fit(x_train, y_train)
ranks["kbest"] = ranking(list(map(float, test.scores_)), col, order=-1)


lr = LinearRegression(normalize=True)
lr.fit(x_train, y_train)
#stop the search when only the last feature is left
rfe = RFE(lr, n_features_to_select=1, verbose =3 )
rfe.fit(x_train, y_train)
ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), col, order=-1)

# Using Linear Regression
lr = LinearRegression(normalize=True)
lr.fit(x_train, y_train)
ranks["LinReg"] = ranking(np.abs(lr.coef_), col)

# Using Ridge
ridge = Ridge(alpha = 7)
ridge.fit(x_train, y_train)
ranks['Ridge'] = ranking(np.abs(ridge.coef_), col)

# Using Lasso
lasso = Lasso(alpha=.05)
lasso.fit(x_train, y_train)
ranks["Lasso"] = ranking(np.abs(lasso.coef_), col)

rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
rf.fit(x_train, y_train)
ranks["RF"] = ranking(rf.feature_importances_, col)

# Create empty dictionary to store the mean value calculated from all the scores
r = {}
for name in col[0:39]:
    r[name] = round(np.mean([ranks[method][name]for method in ranks.keys()]), 2)

methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")

print("\t%s" % "\t".join(methods))
for name in col[0:39]:
    print("%s\t%s" % (name, "\t".join(map(str,
                                          [ranks[method][name] for method in methods]))))

# Put the mean scores into a Pandas dataframe
meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])

# Sort the dataframe
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)
# Let's plot the ranking of the features
sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar",
               size=14, aspect=1.9, palette='coolwarm')
plt.savefig(f'/Users/mlugowska/PhD/applied_statistics/figs/mean_ranking.png')

higher_mean = []
for idx in range(0, 39):
    if meanplot['Mean Ranking'][idx] >= 0.3:
        higher_mean.append(meanplot['Feature'][idx])


### WITHOUT FEATURE SELECTION
for name, model in models:
    start = time.time()
    clf = model.fit(x_train, y_train)
    end = time.time()
    print(f"Execution time for building the {name}: {float(end) - float(start)}")
    ac = accuracy_score(y_test, clf.predict(x_test))

    if hasattr(model, 'decision_function'):
        y_score = clf.decision_function(x_test)
    else:
        y_score = clf.predict_proba(x_test)

    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    err = 1 - acc
    sens = (tp / (tp + fn))
    spec = (tn / (tn + fp))
    fig = print_confusion_matrix(confusion_matrix=cm, class_names=['Male', 'Female'], name=name)
    plt.savefig(f'/Users/mlugowska/PhD/applied_statistics/figs/{name}_confusion_matrix.png')

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
    plt.savefig(f'/Users/mlugowska/PhD/applied_statistics/figs/{name}.png')

#### WITH FEATURE SELECTION RFE

for name, model in models:

    # # RFE
    # rfe = RFECV(estimator=model, cv=4, scoring='accuracy')
    #
    # # AFTER FEATURE SELECTION
    start = time.time()
    # rfe = rfe.fit(x_train, y_train)
    end = time.time()
    print(f"Execution time for building the {name} is: {float(end) - float(start)}")

    cols = x_train.columns[]

    # # Select variables
    # cols = x_train.columns[rfe.support_]
    # print(f'{name}: Number of features selected: {rfe.n_features_}')
    # print(f'{name}: Feature names: {cols}')

    ac = accuracy_score(y_test, rfe.estimator_.predict(x_test[cols]))
    if hasattr(model, 'decision_function'):
        y_score = rfe.estimator_.decision_function(x_test[cols])
    else:
        y_score = rfe.estimator_.predict_proba(x_test[cols])

    y_pred = rfe.estimator_.predict(x_test[cols])
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    err = 1 - acc
    sens = (tp / (tp + fn))
    spec = (tn / (tn + fp))
    fig = print_confusion_matrix(confusion_matrix=cm, class_names=['Male', 'Female'], name=name)
    plt.savefig(f'/Users/mlugowska/PhD/applied_statistics/figs/{name}_confusion_matrix_rfe.png')

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
    plt.savefig(f'/Users/mlugowska/PhD/applied_statistics/figs/{name}_rfe.png')

# ### FEATURE SELECTION PCA

for name, model in models:
    sc = StandardScaler()
    X_train = sc.fit_transform(x_train)
    X_test = sc.transform(x_test)

    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    start = time.time()
    clf = model.fit(X_train, y_train)
    end = time.time()
    print(f"Execution time for building the {name}: {float(end) - float(start)}")
    ac = accuracy_score(y_test, clf.predict(X_test))

    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_)
    plt.xticks(features)
    plt.yticks(features)
    plt.xlabel("PCA features")
    plt.ylabel("variance")

    plt.semilogy(pca.explained_variance_ratio_, '--o')
    plt.semilogy(pca.explained_variance_ratio_.cumsum(), '--o')
    plt.show()

#     if hasattr(model, 'decision_function'):
#         y_score = clf.decision_function(X_test)
#     else:
#         y_score = clf.predict_proba(X_test)
#
#     y_pred = clf.predict(X_test)
#     cm = confusion_matrix(y_test, y_pred)
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#     acc = (tp + tn) / (tp + tn + fp + fn)
#     err = 1 - acc
#     sens = (tp / (tp + fn))
#     spec = (tn / (tn + fp))
#     fig = print_confusion_matrix(confusion_matrix=cm, class_names=['Male', 'Female'], name=name)
#     plt.savefig(f'/Users/mlugowska/PhD/applied_statistics/figs/{name}_confusion_matrix_pca.png')
#
#     msg = f'{name}: Accuracy - {ac}, Sensitivity - {sens}, Specifity - {spec}, Error - {err}'
#     print(msg)
#
#     if hasattr(model, 'decision_function'):
#         fpr, tpr, thr = roc_curve(y_test, y_score)
#     else:
#         fpr, tpr, thr = roc_curve(y_test, y_score[:, 1])
#
#     roc_auc = auc(fpr, tpr)
#     figure = plt.figure()
#     plt.title('Receiver Operating Characteristic')
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % roc_auc)
#     plt.legend(loc='lower right')
#     plt.plot([0, 1], [0, 1], 'r--', color='navy', lw=2)
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.title(f'ROC Curve of {name}')
#     plt.savefig(f'/Users/mlugowska/PhD/applied_statistics/figs/{name}_pca.png')
