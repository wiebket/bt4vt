#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 01-05-2022
# @author: wiebket, AnnaLesch

import itertools


def split_scores_by_speaker_groups(scores, speaker_metadata, speaker_groups):
    """ Construction of a dictionary that holds a list of tuples (label, score) for the speaker groups as defined in the config file and their corresponding categories.

    :param scores: DataFrame that contains reference and test utterances and corresponding labels and scores
    :type scores: DataFrame
    :param speaker_metadata: DataFrame that contains speaker metadata with speaker ids and speaker groups attributes as specified in config file
    :type speaker_metadata: DataFrame
    :param speaker_groups: List of speaker groups as specified in config file
    :type speaker_groups: list

    :returns: scores_by_speaker_groups
    :rtype: dict

    """

    scores_by_speaker_groups = dict()

    # create id column for scores
    scores['ref_id'] = scores['ref'].apply(lambda x: x.split('/')[0])

    for group in speaker_groups:
        categories_per_group = {}
        group_copy = group.copy()

        while len(group_copy) > 0:
            group_name = group_copy[0]
            categories = list(speaker_metadata[group_name].unique())
            categories_per_group.update({group_name: categories})
            group_copy.pop(0)

        scores_by_speaker_groups["_".join(categories_per_group.keys())] = dict()
        categories_combinations = list(itertools.product(*categories_per_group.values()))

        for combination in categories_combinations:
            subgroup_dataframe = speaker_metadata
            for index, subcategory in enumerate(combination):
                subgroup_dataframe = subgroup_dataframe.loc[subgroup_dataframe[list(categories_per_group.keys())[index]] == subcategory]

            if subgroup_dataframe.empty:
                print("Warning: No values for subgroup" + str(combination))
                continue

            id_list = subgroup_dataframe["id"]
            scores_filtered = scores[scores['ref_id'].isin(id_list)]

            # check if scores empty (example Sudan)
            if scores_filtered.empty:
                print("Warning: No scores available for subgroup" + str(combination))
                continue

            label_score_list = scores_filtered[["label", "score"]].to_records(index=False)
            scores_by_speaker_groups["_".join(categories_per_group.keys())].update({"_".join(combination): label_score_list})

    return scores_by_speaker_groups