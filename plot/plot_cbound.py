from argparse import ArgumentParser
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

###############################################################################


def rename_index(index):
    """
    Rename an index in the csv file for the plot

    Parameters
    ----------
    index: str
        The name of the index in the csv file
    """
    index_dict = {
        "mnist_1_7": "mnist:1vs7",
        "mnist_4_9": "mnist:4vs9",
        "mnist_5_6": "mnist:5vs6",
        "letter_a_b": "letter:AvsB",
        "letter_d_o": "letter:DvsO",
        "letter_o_q": "letter:OvsQ",
        "fashion_to_pu": "fashion:TOvsPU",
        "fashion_sa_bo": "fashion:SAvsBO",
        "fashion_co_sh": "fashion:COvsSH",
    }
    # We rename the index when it is available in the dictionary
    index = index.split("--")[0]
    if(index in index_dict):
        index = index_dict[index]
    return index


###############################################################################

def main():

    # We parse the arguments
    arg_parser = ArgumentParser(
        description="Generate the figures with the empirical C-Bounds "
        + "(Figures 2 & 3 in the paper)")
    arg_parser.add_argument(
        "path", metavar="path", type=str,
        help="Path of the csv file containing the results")
    arg_list = arg_parser.parse_args()
    path = arg_list.path

    # We read the csv
    data = pd.read_csv(path, index_col=0)
    data = data.sort_index(ascending=False)

    # We get the results for each algorithm
    data_cbound_joint = data.filter(like="--c-bound-joint", axis=0)
    data_bound_joint = data.filter(like="--bound-joint", axis=0)
    data_bound_risk = data.filter(like="--bound-risk", axis=0)
    data_cb_boost = data.filter(like="--cb-boost", axis=0)
    data_mincq = data.filter(like="--mincq", axis=0)

    # We rename the indices
    data_cbound_joint.index = [
        rename_index(n) for n in list(data_cbound_joint.index)]
    data_bound_joint.index = [
        rename_index(n) for n in list(data_bound_joint.index)]
    data_bound_risk.index = [
        rename_index(n) for n in list(data_bound_risk.index)]
    data_cb_boost.index = [
        rename_index(n) for n in list(data_cb_boost.index)]
    data_mincq.index = [
        rename_index(n) for n in list(data_mincq.index)]

    # We get the empirical joint error and the empirical disagreement
    # ------- #
    eS_cbound_joint = data_cbound_joint["eS"]
    dS_cbound_joint = data_cbound_joint["dS"]
    # ------- #
    eS_bound_joint = data_bound_joint["eS"]
    dS_bound_joint = data_bound_joint["dS"]
    # ------- #
    eS_bound_risk = data_bound_risk["eS"]
    dS_bound_risk = data_bound_risk["dS"]
    # ------- #
    eS_cb_boost = data_cb_boost["eS"]
    dS_cb_boost = data_cb_boost["dS"]
    # ------- #
    eS_mincq = data_mincq["eS"]
    dS_mincq = data_mincq["dS"]

    # We create the C-Bound values
    e, d = np.meshgrid(
        np.linspace(0.0, 0.4999, 1000), np.linspace(0.0, 0.4999, 1000))
    e_ = np.linspace(0.0, 0.4999, 1000)
    cb = (1.0-((1.0-(2.0*e+d))**2.0)/(1.0-2.0*d))
    cond_1 = (2*e+d >= 1)
    cond_2 = (d >= 2*(np.sqrt(e)-e))
    cb = np.ma.array(cb, mask=cond_1+cond_2)

    # We plot the result for the datasets in two graphs
    # -> We organize in "dataset_list" where the datasets will be.
    dataset_list = [
        ["mnist:1vs7", "mnist:4vs9", "mnist:5vs6",
         "fashion:TOvsPU", "fashion:SAvsBO", "fashion:COvsSH", "adult"],
        ["letter:AvsB", "letter:DvsO", "letter:OvsQ",
         "credit", "glass", "heart", "tictactoe", "usvotes", "wdbc"],
    ]

    # Each graphs will be organized in nb_line_max x 2 subgraphs
    # -> We compute below nb_line_max
    nb_line_max = math.ceil(len(dataset_list[0])/2)
    for i in range(len(dataset_list)):
        nb_line = math.ceil(len(dataset_list[i])/2)
        if(nb_line_max < nb_line):
            nb_line_max = nb_line

    # For each graph,
    for k in range(len(dataset_list)):

        dataset_list_ = dataset_list[k]
        nb_line = math.ceil(len(dataset_list_)/2)

        # We create the figure
        ax = []
        fig = plt.figure(figsize=(9.8, 13.3), dpi=100, constrained_layout=True)
        gs = fig.add_gridspec(nb_line_max, 4)

        # Then, for each line
        for i in range(nb_line):
            ax.append([])

            nb_column = 2
            if(i+1 == nb_line and len(dataset_list_) % 2 == 1):
                nb_column = 1

            # for each column (there is either 1 or 2 columns)
            for j in range(nb_column):

                # We create the subgraph
                if(nb_column == 2):
                    ax[i].append(fig.add_subplot(gs[i, j*2:(j+1)*2]))
                else:
                    ax[i].append(fig.add_subplot(gs[i, 1:3]))
                ax[i][j].set_title(dataset_list_[i*2+j])

                # We plot the C-Bound values
                cs = ax[i][j].contourf(e, d, cb, 20)
                ax[i][j].plot(
                    e_, 2.0*(np.sqrt(np.minimum(0.25, e_))-e_),
                    "black", linewidth=1)

                # We plot the results of the different algorithms
                ax[i][j].scatter(
                    eS_cbound_joint[dataset_list_[i*2+j]],
                    dS_cbound_joint[dataset_list_[i*2+j]],
                    marker="d", color="black", s=100, alpha=1, zorder=100)
                ax[i][j].scatter(
                    eS_bound_joint[dataset_list_[i*2+j]],
                    dS_bound_joint[dataset_list_[i*2+j]],
                    marker="^", color="black", s=100, alpha=1, zorder=100)
                ax[i][j].scatter(
                    eS_bound_risk[dataset_list_[i*2+j]],
                    dS_bound_risk[dataset_list_[i*2+j]],
                    marker="*", color="black", s=100, alpha=1, zorder=100)
                ax[i][j].scatter(
                    eS_cb_boost[dataset_list_[i*2+j]],
                    dS_cb_boost[dataset_list_[i*2+j]],
                    marker="o", color="black", s=100, alpha=1, zorder=100)
                ax[i][j].scatter(
                    eS_mincq[dataset_list_[i*2+j]],
                    dS_mincq[dataset_list_[i*2+j]],
                    marker="x", color="black", s=100, alpha=1, zorder=100)

                ax[i][j].contour(cs, colors="w", linewidths=0.1)
                ax[i][j].set_xlabel(" ")
                ax[i][j].set_ylabel(" ")

        # We save the graph
        fig.savefig("cbound_"+str(k)+".png", format="png", bbox_inches="tight")

    # We save the legend
    fig, ax = plt.subplots(figsize=(7.84, 10.64), dpi=100)
    fig.colorbar(cs, ax=ax, orientation="horizontal")
    ax.remove()
    fig.savefig("cbound_legend.png", format="png", bbox_inches="tight")


if __name__ == "__main__":
    main()
