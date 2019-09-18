from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression

from classifiers.atrybuty import x_train, y_train, col


def print_best(scores):
    scores = sorted(scores, reverse=True)

    print("The 5 best features selected by this method are :")
    for i in range(5):
        print(scores[i][1])


## Univariate Feature Selection
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
