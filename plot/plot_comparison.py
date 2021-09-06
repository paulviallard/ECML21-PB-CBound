from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns

###############################################################################


def main():

    # We initialize the seeds
    np.random.seed(42)
    random.seed(42)

    # We parse the arguments
    arg_parser = ArgumentParser(
        description="Generate the figure with the pairwise comparisons "
        + "(Figure 1 in the paper)")
    arg_parser.add_argument(
        "path", metavar="path", type=str,
        help="Path of the csv file containing the results")
    arg_list = arg_parser.parse_args()
    path = arg_list.path

    # We read the csv
    csv_pd = pd.read_csv(path, index_col=0)
    csv_pd = csv_pd.sort_index(ascending=False)

    # We get the results for each algorithm
    data_cbound_joint = csv_pd.filter(
        like="--c-bound-joint", axis=0)[["zero_one_T", "c_bound_joint"]]
    data_bound_joint = csv_pd.filter(
        like="--bound-joint", axis=0)[["zero_one_T", "joint_bound"]]
    data_bound_risk = csv_pd.filter(
        like="--bound-risk", axis=0)[["zero_one_T", "risk_bound"]]
    data_cb_boost = csv_pd.filter(
        like="--cb-boost", axis=0)[["zero_one_T", "c_bound_joint"]]
    data_mincq = csv_pd.filter(
        like="--mincq", axis=0)[["zero_one_T", "c_bound_joint"]]

    # We get the bound values and the test loss
    data_cbound_joint.columns = ["zero_one_T", "Bnd"]
    data_list = [data_bound_risk, data_bound_joint, data_cb_boost, data_mincq]
    for i in range(len(data_list)):
        data_list[i].columns = ["zero_one_T", "Bnd"]

    alg_list = ["2r", "Masegosa", "Cb-Boost", "MinCq"]
    markers = ["p", "X", "o", "s"]
    colors = sns.color_palette("viridis")
    markers_colors = [(m, c) for m in markers for c in colors]

    fig = plt.figure(figsize=(20.0, 10.0), dpi=100, constrained_layout=True)
    gs = fig.add_gridspec(2, 4)

    # For the bound values and the test loss,
    for i in range(2):

        # For each pair (Alg.3, Alg)
        # where Alg in {"2r", "Masegosa", "Cb-Boost", "MinCq"}
        for j in range(4):

            # We create the subgraph
            ax = fig.add_subplot(gs[i, j])
            ax.set_title(" ")
            ax.set_xlabel(" ")
            ax.set_ylabel(" ")

            # We get the bound values OR the test loss
            comp = data_list[0].columns[i]
            x = data_cbound_joint[comp]
            y = data_list[j][comp]
            x = np.minimum(np.array(x), 1.0)
            y = np.minimum(np.array(y), 1.0)

            # We plot the results
            for k in range(len(x)):
                m, c = markers_colors[k]
                ax.scatter(x[k], y[k], marker=m, color=c, s=50)

            # We plot the line y=x
            eps = 0.05
            x_ = [np.maximum(np.min(x)-eps, 0.0), np.max(x)+eps]
            ax.plot(x_, x_, c="black", linestyle="dashed", zorder=-100)

    # We save the graph
    fig.savefig("comparison.pdf", format="pdf")


if __name__ == "__main__":
    main()
