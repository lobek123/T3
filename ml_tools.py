import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def read_dataset(name, class_column, index_col=None):
    dataset = pd.read_csv(f'datasets/{name}.csv', index_col=index_col)
    classes = dataset[class_column].unique()
    y = dataset.pop(class_column).astype('category').cat.codes
    return dataset, y, classes


def calculate_metrics(target, prediction, average='macro'):
    accuracy = accuracy_score(target, prediction)
    precision = precision_score(target, prediction, average=average)
    recall = recall_score(target, prediction, average=average)
    f1 = f1_score(target, prediction, average=average)
    mislabeled = (target != prediction).sum()
    return accuracy, precision, recall, f1, mislabeled, len(target)


def print_results(metrics, classifier_id='classifier'):
    print(f'Results for {classifier_id}')
    print('----')
    print(f'  Accuracy:  {metrics[0]}')
    print(f'  Precision: {metrics[1]}')
    print(f'  Recall:    {metrics[2]}')
    print(f'  F1 score:  {metrics[3]}')
    print(f'  Mislabeled {metrics[4]} out of {metrics[5]}')
    print('\n')


def plot_cm(conf_mtx, classes=None):
    df = pd.DataFrame(conf_mtx, index=classes, columns=classes)
    g = sns.heatmap(df, annot=True, cmap='Blues')
    g.set_ylabel('Truth')
    g.set_xlabel('Prediction')
    return g


def cross_validate(classifier, kfold, X, y, params=None):
    if params is None:
        params = {}
    predicted = []
    target = []
    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clf = classifier(**params).fit(X_train, y_train)
        predicted = np.concatenate((predicted,
                                    clf.predict(X_test)))
        target = np.concatenate((target,
                                 y_test))
    metrics = calculate_metrics(target, predicted)
    cm = confusion_matrix(target, predicted)
    return metrics, cm


def get_grid(data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01),
                       np.arange(y_min, y_max, 0.01))


def plot_space(classifier, train_data, train_labels):
    classifier.fit(train_data, train_labels)
    xs, ys = get_grid(train_data)

    test_data = np.column_stack((xs.ravel(),
                                 ys.ravel()))

    predicted = classifier.predict(test_data).reshape(xs.shape)

    fig, ax = plt.subplots()
    ax.pcolormesh(xs, ys, predicted)
    ax.scatter(train_data[:, 0], train_data[:, 1], c=train_labels,
               edgecolors='k')
    plt.show()
