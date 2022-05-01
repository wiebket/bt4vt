#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 01-05-2022
# @author: wiebket, AnnaLesch

def summarise_dataset(demo_df):

    # TODO: clean up file and make independent from voxceleb
    """
    This function creates a multi-index ['ref_nationality','ref_gender'] dataframe with the following columns

    - count of unique speakers (unique_speakers)

    - mean utterances per speaker (utterances_speaker)

    - percentage of verification pairs in the same subgroup (same_subgroup)

    :param demo_df: pandas dataframe object in the format of output of voxceleb_scores_with_demographics
    :type demo_df: dataframe

    :returns: [description]
    """

    df = demo_df.groupby(['subgroup', 'ref_nationality', 'ref_gender']).agg(
        {'ref_vggface1': 'nunique', 'lab': 'count', 'same_sg': 'sum'})
    df['same_sg'] = round(df['same_sg'] / df['lab'], 3) * 100
    df['lab'] = round(df['lab'] / df['ref_vggface1'], 1)
    df.rename({'ref_vggface1': 'unique_speakers',
               'lab': 'utterances_speaker',
               'same_sg': 'same_subgroup'}, inplace=True, axis='columns')
    df.reset_index(inplace=True)

    return df

def compare_experiments(experiment_dict: dict, comparison: str):
    """[summary]

    :param experiment_dict: key [fpfnth dataframe, metrics dict]
    :type experiment_dict: dict
    :param comparison: [description]
    :type comparison: str

    :returns: [description]
    :rtype: [type]

    """

    compare_df = []
    compare_metrics = {}

    for k, v in experiment_dict.items():
        df = v[0]
        df[comparison] = k
        compare_metrics[k] = v[1]
        compare_df.append(df)

    compare_fpfnth = pd.concat(compare_df)

    return compare_fpfnth, compare_metrics


def score_overlap(df, metrics):
    """Algorithm to calculate FP-FN overlap area for each subgroup

    For each subgroup

    Lookup the equal error rate threshold

    Find min score where lab = 1

    Find max score where lab = 0

    If lab = 1

        Count scores where min_score <= score <= eer

    If lab = 0

        Count scores where eer <= score <= max_score

    Sum the overlap counts

    Calculate probability ratio = overlap instances / total instances

    :param df: [description]
    :type df: pandas.DataFrame
    :param metrics: [description]
    :type metrics: [type]

    :returns: [description]
    :rtype: [type]

    """

    score_overlap = {}

    for subgroup in list(df['subgroup'].unique()):
        eer_threshold = metrics[subgroup]['eer_threshold']
        min_sc = df[(df['subgroup'] == subgroup) & (df['lab'] == 1)]['sc'].min()
        max_sc = df[(df['subgroup'] == subgroup) & (df['lab'] == 0)]['sc'].max()
        overlap_fn = \
        df[(df['subgroup'] == subgroup) & (df['lab'] == 1) & (df['sc'] >= min_sc) & (df['sc'] <= eer_threshold)][
            'sc'].count()
        overlap_fp = \
        df[(df['subgroup'] == subgroup) & (df['lab'] == 0) & (df['sc'] >= eer_threshold) & (df['sc'] <= max_sc)][
            'sc'].count()
        overlap_total = overlap_fn + overlap_fp
        total_instances = df[df['subgroup'] == subgroup]['sc'].count()
        overlap_probability = overlap_total / total_instances

        score_overlap[subgroup] = overlap_probability

    return score_overlap