#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 01-05-2022
# @author: wiebket, AnnaLesch

import numpy as np
import scipy as sp


#########################################
# In this section we compute performance evaluation metrics
# 1. Equal Error Rate
# 2. Minimum of the Detection Cost Function (mincdet)
#########################################

def compute_eer(fprs, fnrs, thresholds):
    """3. equal error rate and corresponding threshold score value"""

    # Calculate the equal error rate and its threshold
    min_ix = np.nanargmin(np.absolute((fnrs - fprs)))
    eer = max(fprs[min_ix], fnrs[min_ix]) * 100
    eer_threshold = thresholds[min_ix]

    return (eer, eer_threshold)


def compute_min_cdet(fprs, fnrs, thresholds, dcf_p_target, dcf_c_fp, dcf_c_fn):
    """Compute the minimum of the detection cost function as defined in the NIST Speaker Recognition Evaluation Plan 2019.

    :param fnrs: [description]
    :type fnrs: [type]
    :param fprs: [description]
    :type fprs: [type]
    :param thresholds: [description]
    :type thresholds: [type]
    :param dcf_p_target: [description]
    :type dcf_p_target: [type]
    :param dcf_c_fn: [description]
    :type dcf_c_fn: [type]
    :param dcf_c_fp: [description]
    :type dcf_c_fp: [type]

    :returns: min_cdet, min_cdet_threshold

    """

    cdet = np.array([fnr * dcf_c_fn * dcf_p_target for fnr in fnrs]) + np.array(
        [fpr * dcf_c_fp * (1 - dcf_p_target) for fpr in fprs])
    min_ix = np.nanargmin(cdet)
    min_cdet = cdet[min_ix]  # NB: divide by 0.05 to get results that correspond with VoxCeleb benchmark
    min_cdet_threshold = thresholds[min_ix]

    return (min_cdet, min_cdet_threshold)


def get_fpfn_at_threshold(fprs, fnrs, thresholds, threshold_value: float, ppf_norm=False):
    """ Calculate the false positive rate (FPR) and false negative rate (FNR) at the minimum threshold value.

    :param df: dataframe, must contain false negative rates ['fnrs'], false positive rates ['fprs'] and threshold values ['thresholds']
    :type df: pandas.DataFrame
    :param threshold: score at threshold 'min_cdet_threshold' or 'eer_threshold'
    :type threshold: float
    :param ppf_norm: normalise the FNR and FPR values to the percent point function. Defaults to False.
    :type ppf_norm: bool, optional

    :returns: [FPR, FNR] at minimum threshold value
    :rtype: list

    """
    # Note: fprs, fnrs, thresholds are dataframes... structure has changed since rewriting. Need to check if this still works.
    # Two options for changing:
    # 1: only pass one column of dataframe, e.g. 'overall'
    # 2: make sure method works for all columns, i.e. all subgroups in dataframe as once by ensuring matrix computation

    # Find the index in df that is closest to the SUBGROUP minimum threshold value
    threshold_diff = np.array([abs(i - threshold_value) for i in thresholds])

    if ppf_norm == True:
        threshold_fpr = sp.stats.norm.ppf(fprs)[np.ndarray.argmin(threshold_diff)]
        threshold_fnr = sp.stats.norm.ppf(fnrs)[np.ndarray.argmin(threshold_diff)]
    else:
        threshold_fpr = fprs[np.ndarray.argmin(threshold_diff)]
        threshold_fnr = fnrs[np.ndarray.argmin(threshold_diff)]

    return threshold_fpr, threshold_fnr


def compute_cdet_at_threshold(fprs, fnrs, thresholds, threshold_value: float, dcf_p_target, dcf_c_fn, dcf_c_fp):

    fp_at_threshold, fn_at_threshold = get_fpfn_at_threshold(fprs, fnrs, thresholds, threshold_value)
    cdet_at_threshold = fp_at_threshold * dcf_c_fp * (1 - dcf_p_target) + fn_at_threshold * dcf_c_fn * dcf_p_target

    return cdet_at_threshold


#########################################
# In this section we compute bias metrics
# 1. Ratio of group mincdet / overall mincdet
# 2. Ratio of group fp, fn rates / overall fp, fn rates
#########################################

def compute_metrics_ratios(metric_scores):

    metrics_ratios = metric_scores.iloc[:, 1:].div(metric_scores['overall'], axis=0)

    return metrics_ratios


def compute_fpfn_ratio(fpfnth, metrics, metrics_baseline, filter_keys: list, threshold_type):

    # nice function for future to understand the real life impact of threshold settings

    return