#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 01-05-2022
# @author: wiebket, AnnaLesch

def split_scores_by_speaker_groups(scores, speaker_metadata, speaker_groups):

    scores_by_speaker_groups = dict()

    #TODO: implement method
    #0. Choose group (e.g. Gender)
    # NB: for groups joined on columns, join columns with '_'
    # OLD method
    #    filters = ' & '.join([f"{k}=='{v}'" for k, v in filter_dict.items()])
    #    sg = df.query(filters)
    #    sg[['sc','lab']+list(filter_dict.keys())]

    #1. Get categories in group
    # OLD method
    # for f_key in filter_keys:
    #    f_vals = list(df[f_key].unique())
    #    f_vals.sort()
    #    filter_dict[f_key] = f_vals

    #2. Filter speaker_metadata by category (e.g. male)
    #3. Get ids of all speakers in category in speaker_metadata
    #4. Filter & get score & label columns for ids from 3. that are in ref column in scores
    #5. Save as dictionary: scores_by_speaker_groups['group': {'category': (scores.label, scores.score)}]

    return scores_by_speaker_groups


def sg_fpfnth_metrics(df, filter_keys: list, dcf_p_target=0.05, dcf_c_fn=1, dcf_c_fp=1):
    """
    This function returns false negative rates, false positive rates and threshold values
    for subgroups in dataframe df. Subgroups are defined by the column names specified in
    the filter_keys argument.

    :param df: dataframe with speaker verification predictions that has columns, df['sc'] scores, df['lab'] binary labels (0=False, 1=True)
    :type df: dataframe
    :param filter_keys: list of column names in df that define subgroups, e.g. ['nationality','sex'], Creates subgroups from all unique value pairs in the designated columns. Currently supports lists with 1, 2 or 3 items.
    :type filter_keys: list
    :param dcf_p_target: detection cost function target (default = 0.05)
    :type dcf_p_target: float
    :param dcf_c_fn: detection cost function false negative weight (default = 1)
    :type dcf_c_fn: float
    :param dcf_c_fp: detection cost function  false positive weight (default = 1)
    :type dcf_c_fp: float

    :returns: fpfnth, metrics dataframe with the following columns 'fnrs','fprs','thresholds','subgroup' and minimum detection cost and equal error rate metrics for each subgroup
    :rtype: dataframe, dict

    """

    # Create a dictionary to construct subgroups from filter_keys {filter_key: [unique filter_key values], }
    filter_dict = {}

    for f_key in filter_keys:
        f_vals = list(df[f_key].unique())
        f_vals.sort()
        filter_dict[f_key] = f_vals

    # Create a list of dictionaries of potential subgroups from all combinations of values in filter_dict
    # [{filter_keys[0]: filter_keys[0] first unique value, filter_keys[1]: filter_keys[1] first unique value},
    #  {filter_keys[0]: filter_keys[0] second unique value, filter_keys[1]: filter_keys[1] first unique value}, ...]
    filter_items = []

    for val0 in filter_dict[filter_keys[0]]:
        try:
            for val1 in filter_dict[filter_keys[1]]:
                try:
                    for val2 in filter_dict[filter_keys[2]]:
                        f_item = {filter_keys[0]: val0, filter_keys[1]: val1, filter_keys[2]: val2}
                        filter_items.append(f_item)
                except IndexError:
                    f_item = {filter_keys[0]: val0, filter_keys[1]: val1}
                    filter_items.append(f_item)
        except IndexError:
            f_item = {filter_keys[0]: val0}
            filter_items.append(f_item)

    fpfnth_list = []
    metrics = {}

    for fi in filter_items:
        try:
            sg = _subgroup(df, fi)
            sg_fpfnth, sg_metrics = fpfnth_metrics(sg, dcf_p_target, dcf_c_fn, dcf_c_fp)
            for key, val in fi.items():
                sg_fpfnth[key] = val
            sg_name = '_'.join([v.replace(" ", "").lower() for v in fi.values()])
            sg_fpfnth['subgroup'] = sg_name
            fpfnth_list.append(sg_fpfnth)
            metrics[sg_name] = sg_metrics
        except:
            pass

    fpfnth = pd.concat(fpfnth_list, ignore_index=True)

    return (fpfnth, metrics)
