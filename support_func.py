import pandas as pd
import numpy as np
import matplotlib
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams


rcParams['figure.figsize'] = 15, 6


def load_data(file_location):
    data = pd.read_csv(file_location, sep=",", header=0)

    return data


def save_hist(data, bins, alpha, image_dir, color='green'):
    fig = plt.figure()
    ax = plt.subplot()
    plt.hist(data, bins=bins, color=color, alpha=alpha)
    ax.set_xlabel(data.name)
    plt.title(data.name)
    plt.savefig(image_dir + data.name + '.png', bbox_inches='tight')
    plt.close(fig)


def save_scatter(x, y, image_dir):
    """
    saves scatter plot for given pandas series as x and y
    :param x: pd series
    :param y: pd series
    :param image_dir: file save location
    :return: none
    """
    df = pd.DataFrame()
    df[str(x.name)] = x.astype(np.int64)
    df[str(y.name)] = y
    fig = plt.figure()
    ax = plt.subplot()
    df.plot(x=str(x.name), y=str(y.name), kind='scatter', ax=ax)
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)

    ax.set_xticklabels([datetime.fromtimestamp(ts / 1e9).strftime('%Y-%m-%d') for ts in ax.get_xticks()])
    plt.title(y.name + " vs " + x.name)
    plt.savefig(image_dir + y.name + "_vs_" + x.name + '.png', bbox_inches='tight')
    plt.close(fig)

def save_boxPlots_for_cat_var(data, cat_var_list, y, image_dir):

    for elem in cat_var_list:

        if len(data[elem].unique()) <= 30:
            dfg = data.groupby(elem)

            counts = [len(v) for k, v in dfg]
            total = float(sum(counts))
            cases = len(counts)

            widths = [c / total for c in counts]

            fig = plt.figure()
            # ax = plt.subplot()
            cax = data[[elem, y]].boxplot(by= elem)
            cax.set_xticklabels(['%s\n$n$=%d' % (k, len(v)) for k, v in dfg])
            plt.savefig(image_dir + y + "_vs_" + elem + ".png", bbox_inches='tight')
            plt.close(fig)

