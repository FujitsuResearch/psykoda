"""Plot detection result"""

from logging import getLogger
from typing import Optional

import numpy as np
from matplotlib import axes, cm, figure, pyplot
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer

logger = getLogger(__name__)


def plot_detection(
    X,
    idx_anomaly,
    name_anomaly,
    X_labeled=None,
    name_labeled=None,
    path_saved: Optional[str] = None,
    no_plot: bool = False,
):
    """2-D scatter plot of feature representations

    .. todo::
        type annotation (with change of API)

    Parameters
    ----------
    X
        embeddings of samples
        :shape: (n_samples, n_features)
    idx_anomaly
        index of anomaly samples
    name_anomaly
        name of anomaly samples
    X_labeled
        embeddings of labeled samples
        :shape: (n_labeled, n_features)
    name_labeled
        name of labeled samples
    path_saved
        path to save figure to
    no_plot
        .. todo::
            replace with plot=True to make API simpler
    """
    #   X: {array-like, sparse matrix}
    #   idx_anomaly: list[int]
    #       e.g. [1,5] -> X[1] and X[5] are anomaly
    #   name_anomaly: list[str]
    #       e.g. [ "2021-04-01-14__10.1.1.1", "2021-04-01-14__10.1.1.2" ]
    #   X_labeled: {array-like, sparse matrix}
    #   name_anomaly: list[str]
    #   path_saved: str

    assert X.ndim == 2
    if X.shape[1] < 2:
        raise DimensionTooLow(f"2-D plot needs 2 or more features, got {X.shape[1]}")

    transformer = PCA()
    if X.shape[1] == 2:
        transformer = FunctionTransformer()

    X_reduced = transformer.fit_transform(X)

    fig = pyplot.figure(tight_layout=True)
    ax = fig.add_subplot()
    ax = _plot_normal(ax, X, X_reduced, idx_anomaly)
    ax = _plot_anomaly(ax, X_reduced, idx_anomaly, name_anomaly)
    ax = _plot_labeled(ax, X_labeled, name_labeled, transformer)
    ax = _plot_config(ax)
    _plot_output(fig, path_saved, no_plot)


def _plot_normal(ax: axes.Axes, X, X_reduced, idx_anomaly):
    idx_normal = np.ones(len(X), dtype=bool)
    idx_normal[idx_anomaly] = False

    ax.scatter(
        X_reduced[idx_normal, 0], X_reduced[idx_normal, 1], marker="o", alpha=0.7
    )
    return ax


def _plot_anomaly(ax: axes.Axes, X_reduced, idx_anomaly, name_anomaly):
    color_map = cm.get_cmap("hsv")
    for i, idx in enumerate(idx_anomaly):
        rgb = list(color_map(0.9 * i / len(idx_anomaly)))
        ax.scatter(
            X_reduced[idx, 0],
            X_reduced[idx, 1],
            label=name_anomaly[i],
            color=rgb,
            marker="x",
            s=80,
            alpha=0.7,
        )
    return ax


def _plot_labeled(ax: axes.Axes, X_labeled, name_labeled, transformer):
    color_map = cm.get_cmap("hsv")
    if X_labeled is not None:
        num_labeled_samples = len(X_labeled)
        if num_labeled_samples > 0:
            X_labeled_reduced = transformer.transform(X_labeled)
            for i in range(num_labeled_samples):
                rgb = list(color_map(0.9 * i / num_labeled_samples))
                label = name_labeled[i] if name_labeled is not None else None
                ax.scatter(
                    X_labeled_reduced[i, 0],
                    X_labeled_reduced[i, 1],
                    label=label,
                    color=rgb,
                    marker="^",
                    s=80,
                    alpha=0.7,
                )
    return ax


def _plot_config(ax: axes.Axes):
    ax.legend(
        bbox_to_anchor=(1.05, 1.0), loc="upper left", borderaxespad=0, fontsize=10
    )
    ax.set_xlabel("pc1")
    ax.set_ylabel("pc2")
    ax.set_title("plots of IP address samples")
    return ax


def _plot_output(fig: figure.Figure, path_saved, no_plot):
    if path_saved is not None:
        fig.savefig(path_saved)
    if not no_plot:
        pyplot.show()
    pyplot.close()


class DimensionTooLow(ValueError):
    """Not 2-D plottable"""
