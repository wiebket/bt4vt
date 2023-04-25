#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 01-05-2022
# @author: wiebket, AnnaLesch

import itertools
import numpy as np


def split_scores_by_speaker_groups(scores, speaker_metadata, group_names, id_delimiter):
    """ Construction of a dictionary that holds a list of tuples (label, score) for the speaker groups as defined in the config file and their corresponding subgroups.

    :param scores: DataFrame that contains reference and test utterances and corresponding labels and scores
    :type scores: DataFrame
    :param speaker_metadata: DataFrame that contains speaker metadata with speaker ids and speaker groups attributes as specified in config file
    :type speaker_metadata: DataFrame
    :param group_names: List of speaker groups as specified in config file
    :type group_names: list
    :param id_delimiter: If not specified in config file, default is "/"
    :type id_delimiter: string

    :returns: scores_by_speaker_groups
    :rtype: dict

    """

    scores_by_speaker_groups = dict()

    # create id column for scores, first split by dot to get rid of .wav, then by id_delimiter
    scores['ref_id'] = scores['ref'].apply(lambda x: x.split(".")[0]).apply(lambda x: x.split(id_delimiter)[0])

    for group in group_names:
        subgroup_per_group = dict()
        group_copy = group.copy()

        while len(group_copy) > 0:
            group_name = group_copy[0]
            subgroups = list(speaker_metadata[group_name].unique())
            subgroup_per_group.update({group_name: subgroups})
            group_copy.pop(0)

        scores_by_speaker_groups["_".join(subgroup_per_group.keys())] = dict()
        # for a list of subgroups for groups create category combination
        # e.g. Gender: [m, f], Nationality: [India] becomes [(m, India), (f, India)]
        subgroups_combinations = list(itertools.product(*subgroup_per_group.values()))

        for combination in subgroups_combinations:
            subgroup_dataframe = speaker_metadata
            for index, subcategory in enumerate(combination):
                subgroup_dataframe = subgroup_dataframe.loc[subgroup_dataframe[list(subgroup_per_group.keys())[index]] == subcategory]

            # subgroup combination not available in speaker_metadata
            if subgroup_dataframe.empty:
                scores_by_speaker_groups["_".join(subgroup_per_group.keys())].update({"_".join(combination): [(np.nan, np.nan)]})
                # TODO logging here
                continue

            id_list = subgroup_dataframe["id"]
            scores_filtered = scores[scores['ref_id'].isin(id_list)]

            # speaker id in metadata but no scores provided
            if scores_filtered.empty:
                scores_by_speaker_groups["_".join(subgroup_per_group.keys())].update({"_".join(combination): [(np.nan, np.nan)]})
                # TODO Logging here
                continue

            label_score_list = scores_filtered[["label", "score"]].to_records(index=False)
            scores_by_speaker_groups["_".join(subgroup_per_group.keys())].update({"_".join(combination): label_score_list})

    return scores_by_speaker_groups
