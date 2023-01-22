#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 01-05-2022
# @author: wiebket, AnnaLesch

import pandas as pd
import yaml
import os
import sys
import importlib.resources


def load_data(data_in):
    """Read a csv, txt file or a DataFrame into a DataFrame. If given a file it uses the Python parsing engine to automatically detect the separator.

    :param data_in: Either path to csv or txt file or a Pandas DataFrame
    :type data_in: str or DataFrame

    :returns: data
    :rtype: DataFrame

    """
    if data_in == "vox1_meta.csv":
        with importlib.resources.path("bt4vt.data", data_in) as path:
            data = pd.read_csv(str(path), sep=None, engine="python")
            return data

    if isinstance(data_in, str):
        data = pd.read_csv(data_in, sep=None, engine="python")
    elif isinstance(data_in, pd.DataFrame):
        data = data_in
    else:
        data = None
        # TODO: error handling

    return data


def load_config(file_name):
    """Read a yaml config file into a dictionary.

    :param file_name: path to the yaml config file
    :type file_name: str

    :returns: config
    :rtype: dict

    """
    if not file_name.lower().endswith("yaml"):
        raise Exception("Config File has to be a YAML file with .yaml extension")

    with open(os.path.expanduser(file_name), 'r') as file:

        config = yaml.safe_load(file)

        # check if list of lists
        if not all(isinstance(el, list) for el in config["dcf_costs"]):
            raise ValueError("DCF Costs in config file must be a list of lists")

        # check if tuple conversion works
        try:
            # conversion to tuple as safe_load does not load tuples
            config["dcf_costs"] = [tuple(v) for v in config["dcf_costs"]]
        except TypeError:
            print("Error: DCF costs in config file must be a list of lists containing float values")
            sys.exit(1)

    return config


def write_data(data, file_name):

    data.to_csv(file_name, index=False, na_rep="NaN")

    return
