import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def violin_plot(X, y):
    # first ten features
    data_dia = y
    data = X
    data_n_2 = (data - data.mean()) / (data.std())  # standardization
    data = pd.concat([y, data_n_2.iloc[:, 0:10]], axis=1)
    data = pd.melt(data, id_vars="Male",
                   var_name="features",
                   value_name='value')
    plt.figure(figsize=(10, 10))
    sns.violinplot(x="features", y="value", hue="Male", data=data, split=True, inner="quart")
    plt.xticks(rotation=90)
    plt.show()

    # Second ten features
    data = pd.concat([y, data_n_2.iloc[:, 10:20]], axis=1)
    data = pd.melt(data, id_vars="Male",
                   var_name="features",
                   value_name='value')
    plt.figure(figsize=(10, 10))
    sns.violinplot(x="features", y="value", hue="Male", data=data, split=True, inner="quart")
    plt.xticks(rotation=90)
    plt.show()

    # Third ten features
    data = pd.concat([y, data_n_2.iloc[:, 20:31]], axis=1)
    data = pd.melt(data, id_vars="Male",
                   var_name="features",
                   value_name='value')
    plt.figure(figsize=(10, 10))
    sns.violinplot(x="features", y="value", hue="Male", data=data, split=True, inner="quart")
    plt.xticks(rotation=90)
    plt.show()

    # Fourth ten features
    data = pd.concat([y, data_n_2.iloc[:, 30:41]], axis=1)
    data = pd.melt(data, id_vars="Male",
                   var_name="features",
                   value_name='value')
    plt.figure(figsize=(10, 10))
    sns.violinplot(x="features", y="value", hue="Male", data=data, split=True, inner="quart")
    plt.xticks(rotation=90)
    plt.show()


def box_plot(data):
    plt.figure(figsize=(10, 10))
    sns.boxplot(x="features", y="value", hue="Male", data=data)
    plt.xticks(rotation=90)
    plt.show()

    # sns.set(style="white")
    # df = X.loc[:, :]
    # g = sns.PairGrid(df, diag_sharey=False)
    # g.map_lower(sns.kdeplot, cmap="Blues_d")
    # g.map_upper(plt.scatter)
    # g.map_diag(sns.kdeplot, lw=3)


def swarm_plot(X, y):
    sns.set(style="whitegrid", palette="muted")
    data_dia = y
    data = X
    data_n_2 = (data - data.mean()) / (data.std())  # standardization
    data = pd.concat([y, data_n_2.iloc[:, 0:10]], axis=1)
    data = pd.melt(data, id_vars="Male",
                   var_name="features",
                   value_name='value')
    plt.figure(figsize=(10, 10))
    sns.swarmplot(x="features", y="value", hue="Male", data=data)
    plt.xticks(rotation=90)
    plt.show()


def print_confusion_matrix(confusion_matrix, class_names, name, figsize=(10, 7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig