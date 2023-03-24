import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_classification(y_test, y_pred, labels=[]):
    print(classification_report(y_test, y_pred, labels = labels))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot()



def print_shapes(X_train, X_test, y_train, y_test):
    print(
        f"{X_train.shape = }\n{X_test.shape = }\n{y_train.shape = }\n{y_test.shape = }\n"
    )


def train_test_split(X, y, train_fraction=0.7, seed=42, replace=False):
    """Splits up X, y to training and testing data
    Parameters
    ----------
    X : array-like
        Length must be same as y
    y : array-like
        Length must be same as X
    train_fraction : float, optional
        Fraction for training data set
    seed : int, optional
        Random seed for reproducibility
    replace: bool, optional
        True for sampling with replacement

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """

    if len(X) != len(y):
        raise ValueError("Lengths of X and y not equal!")

    # split 70% training and 30% test
    train_fraction = int(len(X) * train_fraction)

    X_train = X.sample(n=train_fraction, random_state=seed, replace=replace)
    X_test = X.drop(X_train.index)
    y_train = y[X_train.index]
    y_test = y[X_test.index]

    return X_train, X_test, y_train, y_test


# code from
# https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html
def plot_svm_margins(clf, X, y):
    """
    =========================================
    SVM: Maximum margin separating hyperplane
    =========================================

    Plot the maximum margin separating hyperplane within a two-class
    separable dataset using a Support Vector Machine classifier with
    linear kernel.

    """
    # fit the model, don't regularize for illustration purposes
    clf.fit(X, y)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(
        XX,
        YY,
        Z,
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )
    # plot support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
