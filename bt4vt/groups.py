#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 01-05-2022
# @author: wiebket, AnnaLesch

import itertools
import numpy as np


def split_scores_by_speaker_groups(scores, speaker_metadata, speaker_groups):
    """ Construction of a dictionary that holds a list of tuples (label, score) for the speaker groups as defined in the config file and their corresponding categories.

    :param scores: DataFrame that contains reference and test utterances and corresponding labels and scores
    :type scores: DataFrame
    :param speaker_metadata: DataFrame that contains speaker metadata with speaker ids and speaker groups attributes as specified in config file
    :type speaker_metadata: DataFrame
    :param speaker_groups: List of speaker groups as specified in config file
    :type speaker_groups: list
    :param log_file: filename of dataset evaluation log file, if not None dataset evaluation information is written to the log file
    :type log_file: str or None

    :returns: scores_by_speaker_groups
    :rtype: dict

    """

    scores_by_speaker_groups = dict()

    # create id column for scores
    scores['ref_id'] = scores['ref'].apply(lambda x: x.split('/')[0])

    for group in speaker_groups:
        categories_per_group = dict()
        group_copy = group.copy()

        while len(group_copy) > 0:
            group_name = group_copy[0]
            categories = list(speaker_metadata[group_name].unique())
            categories_per_group.update({group_name: categories})
            group_copy.pop(0)

        scores_by_speaker_groups["_".join(categories_per_group.keys())] = dict()
        # for a list of categories for groups create category combination
        # e.g. Gender: [m, f], Nationality: [India] becomes [(m, India), (f, India)]
        categories_combinations = list(itertools.product(*categories_per_group.values()))

        for combination in categories_combinations:
            subgroup_dataframe = speaker_metadata
            for index, subcategory in enumerate(combination):
                subgroup_dataframe = subgroup_dataframe.loc[subgroup_dataframe[list(categories_per_group.keys())[index]] == subcategory]

            # subgroup combination not available in speaker_metadata
            if subgroup_dataframe.empty:
                scores_by_speaker_groups["_".join(categories_per_group.keys())].update({"_".join(combination): [(np.nan, np.nan)]})
                continue

            id_list = subgroup_dataframe["id"]
            scores_filtered = scores[scores['ref_id'].astype(int).isin(id_list)]

            # speaker id in metadata but no scores provided
            if scores_filtered.empty:
                scores_by_speaker_groups["_".join(categories_per_group.keys())].update({"_".join(combination): [(np.nan, np.nan)]})
                continue

            label_score_list = scores_filtered[["label", "score"]].to_records(index=False)
            scores_by_speaker_groups["_".join(categories_per_group.keys())].update({"_".join(combination): label_score_list})
    return scores_by_speaker_groups