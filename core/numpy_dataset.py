import copy
import numpy as np
from re import sub
import torch
from torch.utils.data import Dataset

###############################################################################


class NumpyDataset(Dataset):

    def __init__(self, dataset_dict):
        """
        Initialize the dataset

        Parameters
        ----------
        dataset_dict: dict
            The dictionary of datasets
        """
        self._dataset_key = list(dataset_dict.keys())

        self._mode_list = []

        # We remove the mode in self._dataset_key
        new_dataset_key = {}
        for key in self._dataset_key:
            new_key = sub("_[^_]+$", "", key)
            if(new_key != key):
                new_dataset_key[new_key] = None
        self._dataset_key = list(new_dataset_key.keys())

        self._dataset_dict = copy.deepcopy(dataset_dict)
        self._mode = "train"

    def set_mode(self, mode):
        """
        Change the "mode"

        Parameters
        ----------
        mode: str
            The "mode" of the dataset (train or test)
        """
        # We set the mode of the dataset
        self._mode = mode

    def get_mode(self):
        """
        Get the "mode"
        """
        # We get the mode of the dataset
        return self._mode

    def get_mode_dataset(self, key):
        """
        Get the input/label (the "key") of the according "mode"

        Parameters
        ----------
        key: str
            The name that represents either the input or the label
        """
        if(self._mode == "train" or self._mode == "test"):
            mode_dict_key = key+"_"+self._mode
        else:
            mode_dict_key = key+"_train_"+self._mode

        if(mode_dict_key in self._dataset_dict):
            return self._dataset_dict[mode_dict_key]
        return self._dataset_dict[mode_dict_key]

    def __len__(self):
        """
        Get the size of a dataset (of a given "mode")
        """
        return len(self.get_mode_dataset("x"))

    def label_size(self):
        """
        Get the number of labels
        """
        if("y"+self._mode in self._dataset_dict):
            return len(np.unique(self.get_mode_dataset("y")))
        return 1

    def input_size(self):
        """
        Get the number of features (i.e, the input size)
        """
        return list(self.get_mode_dataset("x").shape[1:])

    def __getitem__(self, i):
        """
        Get the i-th example

        Parameters
        ----------
        i: int
            The index of the example

        Returns
        -------
        dict
            A dictionary with the input and the labels
        """
        item_dict = {
            "mode": self._mode,
            "size": self.__len__(),
            "label_size": self.label_size()
        }

        # For each key (i.e., "dataset"), we insert it in a dictionary
        for key in self._dataset_key:
            item_dict[key] = torch.tensor(self.get_mode_dataset(key)[i])

        return item_dict
