# Define dictionary to store our rankings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler

from classifiers.atrybuty import X_train, y_train, col

ranks = {}


# Create function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))


kbest = SelectKBest(score_func=chi2, k=2)
kbest.fit(X_train, y_train)
ranks["kbest"] = ranking(list(map(float, kbest.scores_)), col, order=-1)

lr = LinearRegression(normalize=True)
lr.fit(X_train, y_train)
# stop the search when only the last feature is left
rfe = RFE(lr, n_features_to_select=1, verbose=3)
rfe.fit(X_train, y_train)
ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), col, order=-1)

# Using Linear Regression
lr = LinearRegression(normalize=True)
lr.fit(X_train, y_train)
ranks["LinReg"] = ranking(np.abs(lr.coef_), col)

# Using Ridge
ridge = Ridge(alpha=7)
ridge.fit(X_train, y_train)
ranks['Ridge'] = ranking(np.abs(ridge.coef_), col)

# Using Lasso
lasso = Lasso(alpha=.05)
lasso.fit(X_train, y_train)
ranks["Lasso"] = ranking(np.abs(lasso.coef_), col)

rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
rf.fit(X_train, y_train)
ranks["RF"] = ranking(rf.feature_importances_, col)

# Create empty dictionary to store the mean value calculated from all the scores
r = {}
for name in col[0:39]:
    r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)

methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")

print("\t%s" % "\t".join(methods))
for name in col[0:39]:
    print("%s\t%s" % (name, "\t".join(map(str,
                                          [ranks[method][name] for method in methods]))))

# Put the mean scores into a Pandas dataframe
meanplot = pd.DataFrame(list(r.items()), columns=['Feature', 'Mean Ranking'])

# Sort the dataframe
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)
# Plot the ranking of the features
sns.factorplot(x="Mean Ranking", y="Feature", data=meanplot, kind="bar",
               size=14, aspect=1.9, palette='coolwarm')
plt.savefig(f'/Users/mlugowska/PhD/applied_statistics/figs/mean_ranking.png')

higher_mean = []
for idx in range(0, 39):
    if meanplot['Mean Ranking'][idx] >= 0.3:
        higher_mean.append(meanplot['Feature'][idx])
