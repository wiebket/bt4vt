#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 01-05-2022
# @author: wiebket, AnnaLesch

import pandas as pd
import yaml


def load_data(file_name):

    data = pd.read_csv(file_name)

    return data


def load_config(file_name):

    with open(file_name, 'r') as file:
        config = yaml.safe_load(file)

    return config



