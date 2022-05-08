#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 01-05-2022
# @author: wiebket, AnnaLesch

import pandas as pd
import yaml


def load_data(file, delim_whitespace=False):

    if isinstance(file, str):
        data = pd.read_csv(file, delim_whitespace=delim_whitespace)
    elif isinstance(file, pd.DataFrame):
        data = pd.read_table(file)
    else:
        data = None
        #TODO: error handling

    return data


def load_config(file_name):

    with open(file_name, 'r') as file:
        config = yaml.safe_load(file)

        # conversion to tuple as safe_load does not load tuples
        config["dcf_costs"] = [tuple(v) for v in config["dcf_costs"]]

    return config



