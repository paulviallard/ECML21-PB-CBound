from argparse import ArgumentParser
import h5py
import numpy as np
import random
import warnings

###############################################################################


def main():

    # We initialize the seeds
    np.random.seed(42)
    random.seed(42)
    warnings.filterwarnings("ignore")

    # We parse the arguments
    arg_parser = ArgumentParser(
        description="Reduce the size of the dataset by"
        + "taking the \"short\" first examples for each class")
    arg_parser.add_argument(
        "old_path", metavar="old_path", type=str,
        help="Path of the h5 dataset file")
    arg_parser.add_argument(
        "new_path", metavar="new_path", type=str,
        help="Path of the new h5 dataset file")
    arg_parser.add_argument(
        "--short", metavar="short", default=200, type=int,
        help="Number of examples per class")

    arg_list = arg_parser.parse_args()
    old_path = arg_list.old_path
    new_path = arg_list.new_path
    short = arg_list.short

    # We open the dataset (h5 file)
    dataset_file = h5py.File(old_path, "r")

    example_train = np.array(dataset_file["x_train"])
    label_train = np.array(dataset_file["y_train"])
    example_test = np.array(dataset_file["x_test"])
    label_test = np.array(dataset_file["y_test"])

    label_list = np.unique(label_train).tolist()

    # We take the "short" first examples
    new_example_train = None
    new_label_train = None
    for y in label_list:
        if(new_example_train is None):
            new_example_train = (example_train[label_train == y])[:short]
            new_label_train = (label_train[label_train == y])[:short]
        else:
            new_label_train = np.concatenate(
                (new_label_train, (label_train[label_train == y])[:short]))
            new_example_train = np.concatenate(
                (new_example_train, (example_train[label_train == y])[:short]))

    # We shuffle the training set
    permutation = np.arange(new_example_train.shape[0])
    np.random.shuffle(permutation)
    new_example_train = new_example_train[permutation, :]
    new_label_train = new_label_train[permutation]

    # We save the new data
    if(old_path == new_path):
        dataset_file.close()
    dataset_file = h5py.File(new_path, "w")

    dataset_file["x_train"] = new_example_train
    dataset_file["y_train"] = new_label_train
    dataset_file["x_test"] = example_test
    dataset_file["y_test"] = label_test


if __name__ == "__main__":
    main()
