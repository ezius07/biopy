import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import glob

from sklearn.manifold import TSNE

from ..datasets.datasets_nature import NucleiDatasetWrapper, OmicDatasetWrapper
from .utils import sample_encodings


def plot_encodings(model, dataset, tsne=False, dim=3, legend=False, legend_labels=None,
                   title=None, colors=None, save=False):
    if isinstance(model, list):
        assert len(model) == len(dataset)
        encodings = []
        for i, mod in enumerate(model):
            encodings.append(sample_encodings(mod, dataset[i], tsne=False))
        encodings_py = np.concatenate(encodings)
        if tsne:
            encodings_py = TSNE(n_components=dim).fit_transform(encodings_py)
    else:
        encodings_py = sample_encodings(model, dataset, tsne, dim)
    if colors is None:
        # Default colors to labels
        if isinstance(dataset, list):
            lab = []
            for i, data in enumerate(dataset):
                if isinstance(data, NucleiDatasetWrapper) or isinstance(data, OmicDatasetWrapper) or hasattr(
                        data, 'no_slicing'):
                    nlab = len(set(list(map(lambda x: x[1].numpy().item(), data))))
                    lab.append(np.array(list(map(lambda x: x[1] + (i * nlab), data))))
                else:
                    nlab = len(set(data[0][:][1].numpy()))
                    lab.append(np.array([int(item[1]) + (i * nlab) for item in data]))
            colors = np.concatenate(lab)
        else:
            if isinstance(dataset, NucleiDatasetWrapper) or isinstance(dataset, OmicDatasetWrapper) or hasattr(dataset,
                                                                                                               'no_slicing'):
                colors = np.array(list(map(lambda x: x[1], dataset)))
            else:
                colors = np.array([int(item[1]) for item in dataset])
        # colors = dataset[:][1].numpy()

    if dim == 3:
        plot_3d(encodings_py[:, 0], encodings_py[:, 1], encodings_py[:, 2], colors=colors,
                legend=legend, legend_labels=legend_labels, title=title, save=save)
    elif dim == 2:
        plot_2d(encodings_py[:, 0], encodings_py[:, 1], colors=colors,
                legend=legend, legend_labels=legend_labels, title=title, save=save)


def plot_3d(x, y, z, colors=False, save=False, legend=False, legend_labels=None, title=None, static_size=False):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    if colors is not False:
        sc = ax.scatter(x, y, z, c=colors, cmap='Paired')
    else:
        ax.scatter(x, y, z)

    #     ax.set_xlabel('X Label')
    #     ax.set_ylabel('Y Label')
    #     ax.set_zlabel('Z Label')
    if legend:
        ax.legend(sc.legend_elements()[0], legend_labels) if legend_labels != None else plt.legend(
            *sc.legend_elements())
    if static_size:
        ax.set_xlim((-70, 70))
        ax.set_ylim((-70, 70))
        ax.set_zlim((-70, 70))

    if save:
        os.makedirs('imgs', exist_ok=True)
        if os.path.isfile("imgs/plot_3d.png"):
            n_file = 1 + max(list(
                map(lambda x: int(x),
                    map(lambda x: x if x != '3d' else 0,
                        map(lambda x: x.split(".png")[0].split("_")[-1],
                            glob.glob("imgs/plot_2d*.png"))))))
            file_suffix = '_' + str(n_file)
        else:
            file_suffix = ''
        plt.savefig(os.path.join('imgs', 'plot_3d' + file_suffix + '.png'))

    if title is not None:
        plt.title(title)
    plt.show()


def plot_2d(x, y, colors=False, save=False, legend=False, legend_labels=None, title=None):
    fig, ax = plt.subplots(figsize=(11, 7))
    if colors is not False:
        sc = ax.scatter(x, y, c=colors, cmap='Paired')
    else:
        ax.scatter(x, y)
    if legend:
        ax.legend(sc.legend_elements()[0], legend_labels) if legend_labels != None else plt.legend(
            *sc.legend_elements())
    if save:
        os.makedirs('imgs', exist_ok=True)
        if os.path.isfile("imgs/plot_2d.png"):
            n_file = 1 + max(list(
                map(lambda x: int(x),
                    map(lambda x: x if x != '2d' else 0,
                        map(lambda x: x.split(".png")[0].split("_")[-1],
                            glob.glob("imgs/plot_2d*.png"))))))
            file_suffix = '_' + str(n_file)
        else:
            file_suffix = ''
        plt.savefig(os.path.join('imgs', 'plot_2d' + file_suffix + '.png'))

    if title is not None:
        plt.title(title)
    plt.show()


def plot_loss_from_epoch(train_loss, test_loss=None, parameters=None, from_epoch=0, to_epoch=30, plot_regularizer=True,
                         save=False):
    fig, ax = plt.subplots(figsize=(11, 7))
    # plt.ylim((1, 4.5))
    ax.plot(range(from_epoch, to_epoch), train_loss[0][from_epoch:to_epoch], '-bD', label="train loss, MSE")
    if plot_regularizer:
        ax.plot(range(from_epoch, to_epoch), train_loss[1][from_epoch:to_epoch], '-rD', label="train loss, regularizer")
    if test_loss is not None:
        ax.plot(range(from_epoch, to_epoch), test_loss[from_epoch:to_epoch], '-gD', label="test  loss, MSE only")
    if parameters is not None:
        plt.title("floss, adam, lr = {}/{} steps, GAMMA={}, W decay={}", str(parameters['LR']),
                  str(parameters['STEP_SIZE']),
                  str(parameters['GAMMA']),
                  str(parameters['WEIGHT_DECAY']))
    ax.legend()
    ax.grid(axis='y')
    if save:
        plt.savefig('loss.png')
    plt.show()
