import numpy as np
import pandas as pd


def save_csv(name_csv, dict_save, name_exp, erase=False):
    """
    We save in a csv file a python dictionary

    Parameters
    ----------
    name_csv: float
        The path of the csv file
    dict_save: dict
        The dictionary to save in the csv (as a row in the csv)
    name_exp: str
        The name of the experiment (i.e., the name of the row)
    erase: bool, optional
        True if we allow to erase an existing row and False otherwise
    """
    # We open or create the dataset
    try:
        csv_pd = pd.read_csv(name_csv, index_col=0)
    except FileNotFoundError:
        csv_pd = pd.DataFrame()

    # If the row (i.e, the name of the experiment) exists
    if(name_exp in csv_pd.index):
        # for all columns in the dictionary
        for column, value in dict_save.items():

            # We create the column in the csv file if it does not exist
            if(column not in csv_pd.columns):
                csv_pd[column] = np.nan

            # We replace the value if erase is True
            if(np.isnan(csv_pd.loc[name_exp][column])
               or (not(np.isnan(csv_pd.loc[name_exp][column])) and erase)):
                csv_pd.at[name_exp, column] = value

        # We save the csv file
        csv_pd.to_csv(name_csv)
        return None

    # Otherwise, we create the row and insert the values

    old_set = set(csv_pd.columns)
    new_set = set(dict_save.keys())

    new_column = old_set.difference(new_set)
    new_column = new_column.union(new_set.difference(old_set))
    new_column = sorted(list(new_column))

    # We create the new columns
    for column in new_column:
        csv_pd[column] = np.nan

    # We add the values
    csv_pd.loc[name_exp] = dict_save
    csv_pd.to_csv(name_csv)
