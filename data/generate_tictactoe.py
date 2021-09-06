from argparse import ArgumentParser
import h5py
import numpy as np
import os
import pandas as pd
import random
import requests
import warnings

###############################################################################


def get_label(data):
    """
    Get the labels of the loaded dataset


    Parameters
    ----------
    data : DataFrame
        Pandas dataframe containing the loaded dataset

    Returns
    -------
    ndarray
        Vector of size N (N is the number of inputs) with the labels
    """
    label = data[9]
    label.loc[label == "positive"] = 0
    label.loc[label == "negative"] = 1
    label = label.astype(int)
    label = label.to_numpy()
    return label


def get_input(data):
    """
    Get the inputs of the loaded dataset

    Parameters
    ----------
    data : DataFrame
        Pandas dataframe containing the loaded dataset

    Returns
    -------
    ndarray
        Matrix of size NxM with the inputs
        (N is the number of examples & M is the number of features)
    """
    data = data.drop([9], axis=1)
    for i in range(9):
        data[i].loc[data[i] == "x"] = 0.0
        data[i].loc[data[i] == "b"] = 0.5
        data[i].loc[data[i] == "o"] = 1.0
    input = data.to_numpy().astype(np.float32)

    # We rescale the data with a min-max scaling
    input_min_max = (input - input.min(axis=0))
    input_min_max /= (input.max(axis=0) - input.min(axis=0))
    return input_min_max


def get_train_test(input, label, ratio_test):
    """
    Get the train/test dataset

    Parameters
    ----------
    input: ndarray
        Matrix of size NxM with the inputs
        (N is the number of examples & M is the number of features)
    label: ndarray
        Vector of size N (N is the number of inputs) with the labels
    ratio_test: float
        Percentage of the original set as the test set

    Returns
    -------
    ndarray, ndarray, ndarray, ndarray
        Input matrices and label vectors for the train set and the test set
    """
    size_test = int(ratio_test*len(input))

    input_test = input[:size_test, :]
    input_train = input[size_test:, :]
    label_test = label[:size_test]
    label_train = label[size_test:]

    return input_train, input_test, label_train, label_test


def shuffle(input, label):
    """
    Shuffle the dataset

    Parameters
    ----------
    input: ndarray
        Matrix of size NxM with the inputs
        (N is the number of examples & M is the number of features)
    label: ndarray
        Vector of size N (N is the number of inputs) with the labels

    Returns
    -------
    ndarray, ndarray
        The shuffled matrix and vector
    """
    permutation = np.arange(input.shape[0])
    np.random.shuffle(permutation)
    input = input[permutation, :]
    label = label[permutation]
    return input, label


def save(path, input_train, input_test, label_train, label_test):
    """
    Save the train/test set in the h5 file

    Parameters
    ----------
    path : str
        Path of the h5 file to save
    input_train: ndarray
        Inputs of the train set
    input_test: ndarray
        Inputs of the test set
    label_train: ndarray
        Labels of the train set
    label_test: ndarray
        Labels of the test set
    """
    dataset_file = h5py.File(path, "w")

    dataset_file["x_train"] = input_train
    dataset_file["y_train"] = label_train
    dataset_file["x_test"] = input_test
    dataset_file["y_test"] = label_test


###############################################################################

def main():

    # We initialize the seeds
    np.random.seed(42)
    random.seed(42)
    warnings.filterwarnings("ignore")

    # We parse the arguments
    arg_parser = ArgumentParser(
        description="Generate tictactoe dataset")
    arg_parser.add_argument(
        "path", metavar="path", type=str,
        help="Path of the h5 dataset file")
    arg_parser.add_argument(
        "--test", metavar="test", default=0.5, type=float,
        help="Proportion of the test set")

    arg_list = arg_parser.parse_args()
    path = arg_list.path
    ratio_test = arg_list.test

    # We download the dataset
    if(not(os.path.exists("data-tictactoe/"))
       or not(os.path.exists("data-tictactoe/tictactoe.data"))):

        if(not(os.path.exists("data-tictactoe"))):
            os.mkdir("data-tictactoe")

        r = requests.get(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            + "tic-tac-toe/tic-tac-toe.data", allow_redirects=True)
        f = open("data-tictactoe/tictactoe.data", "wb")
        f.write(r.content)
        f.close()

    # We open, process and save the dataset

    data = pd.read_csv(
        "data-tictactoe/tictactoe.data", sep=",",
        na_values="?", header=None)

    label = get_label(data)
    input = get_input(data)

    input, label = shuffle(input, label)
    (input_train, input_test, label_train, label_test) = get_train_test(
        input, label, ratio_test)
    save(path, input_train, input_test, label_train, label_test)


if __name__ == "__main__":
    main()
