from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


class Perceptron:
    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1):
        # eta: 学修率 (0.0~1.0)
        self.eta = eta
        # n_iter: 訓練回数
        self.n_iter = n_iter
        # random_state: 重み初期化のための乱数シード
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, yi in zip(X, y):
                update = self.eta * (yi - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w_) + self.b_


def plot_decision_regions(
    X: np.ndarray, y: np.ndarray, classifier: Perceptron, resolution: float = 0.02
):
    # setup marker generator and color map
    markers = ("o", "x", "s", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.5, cmap=cmap)

    # plot class samples
    for i, c1 in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == c1, 0],
            y=X[y == c1, 1],
            alpha=0.8,
            c=colors[i],
            marker=markers[i],
            label=c1,
            edgecolor="black",
        )
