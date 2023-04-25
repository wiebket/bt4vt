#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 01-05-2022
# @author: wiebket, AnnaLesch

import numpy as np
import scipy as sp
import pandas as pd

#########################################
# In this section we compute performance evaluation metrics
# 1. Equal Error Rate
# 2. Minimum of the Detection Cost Function (mincdet)
#########################################


def compute_eer(fprs, fnrs, thresholds):
    """Computation of the Equal Error Rate and its threshold

    :param fprs: Array of False Positive Rates
    :type fprs: ndarray
    :param fnrs: Array of False Negative Rates
    :type fnrs: ndarray
    :param thresholds: Array of Threshold values corresponding to fprs and fnrs
    :type thresholds: ndarray

    :returns: eer, eer_threshold
    :rtype: float, float
    """

    # Compute the equal error rate and its threshold
    min_ix = np.nanargmin(np.absolute((fnrs - fprs)))
    eer = max(fprs[min_ix], fnrs[min_ix]) * 100
    eer_threshold = thresholds[min_ix]

    return eer, eer_threshold


def compute_min_cdet(fprs, fnrs, thresholds, dcf_p_target, dcf_c_fp, dcf_c_fn):
    """Computation of the minimum of the detection cost function and its threshold.
    Computation is performed as defined in the `NIST Speaker Recognition Evaluation Plan 2019 <https://www.nist.gov/itl/iad/mig/nist-2019-speaker-recognition-evaluation>`_

    .. math:: C_{Det}(\\theta) = C_{FN} \\times P_{Target} \\times P_{FN}(\\theta) + C_{FP} \\times (1 - P_{Target}) \\times P_{FP}(\\theta)

    :param fprs: Array of False Positive Rates
    :type fprs: ndarray
    :param fnrs: Array of False Negative Rates
    :type fnrs: ndarray
    :param thresholds: Array of Threshold values corresponding to fprs and fnrs
    :type thresholds: ndarray
    :param dcf_p_target: [description]
    :type dcf_p_target: float
    :param dcf_c_fp: [description]
    :type dcf_c_fp: float
    :param dcf_c_fn: [description]
    :type dcf_c_fn: float

    :returns: min_cdet, min_cdet_threshold
    :rtype: float, float
    """

    cdet = np.array([fnr * dcf_c_fn * dcf_p_target for fnr in fnrs]) + np.array(
        [fpr * dcf_c_fp * (1 - dcf_p_target) for fpr in fprs])
    min_ix = np.nanargmin(cdet)
    min_cdet = cdet[min_ix]
    min_cdet_threshold = thresholds[min_ix]

    return min_cdet, min_cdet_threshold


def get_fnthreshold_at_fp(fprs, fnrs, thresholds, fpr_value, ppf_norm=False):
    """Get the False Negative Rate and threshold at a given False Positive Rate value.

    :param fprs: Array of False Positive Rates
    :type fprs: ndarray
    :param fnrs: Array of False Negative Rates
    :type fnrs: ndarray
    :param thresholds: Array of Threshold values corresponding to fprs and fnrs
    :type thresholds: ndarray
    :param fpr_value: False positive rate value to get fnr and threshold for
    :type fpr_value: float
    :param ppf_norm: normalise the fpr and fnr values to the percent point function. Default is set to False.
    :type ppf_norm: bool

    :returns: fnr_at_fpr, threshold_at_fpr
    :rtype: float, float

    """
    # Find the index in df that is closest to the required fpr value
    fpr_diff = np.array([abs(i - fpr_value) for i in fprs])

    if ppf_norm:
        threshold_at_fpr = sp.stats.norm.ppf(thresholds)[np.ndarray.argmin(fpr_diff)]
        fnr_at_fpr = sp.stats.norm.ppf(fnrs)[np.ndarray.argmin(fpr_diff)]
    else:
        threshold_at_fpr = thresholds[np.ndarray.argmin(fpr_diff)]
        fnr_at_fpr = fnrs[np.ndarray.argmin(fpr_diff)]

    return fnr_at_fpr, threshold_at_fpr


def get_fpfn_at_threshold(fprs, fnrs, thresholds, threshold_value, ppf_norm=False):
    """Get the False Positive Rate and False Negative Rate at a given threshold value.

    :param fprs: Array of False Positive Rates
    :type fprs: ndarray
    :param fnrs: Array of False Negative Rates
    :type fnrs: ndarray
    :param thresholds: Array of Threshold values corresponding to fprs and fnrs
    :type thresholds: ndarray
    :param threshold_value: Threshold value to get fpr and fnr for i.e. min_cdet_threshold
    :type threshold_value: float
    :param ppf_norm: normalise the fpr and fnr values to the percent point function. Default is set to False.
    :type ppf_norm: bool

    :returns: fpr_at_threshold, fnr_at_threshold
    :rtype: float, float

    """
    # Find the index in df that is closest to the SUBGROUP minimum threshold value
    threshold_diff = np.array([abs(i - threshold_value) for i in thresholds])

    if ppf_norm:
        fpr_at_threshold = sp.stats.norm.ppf(fprs)[np.ndarray.argmin(threshold_diff)]
        fnr_at_threshold = sp.stats.norm.ppf(fnrs)[np.ndarray.argmin(threshold_diff)]
    else:
        fpr_at_threshold = fprs[np.ndarray.argmin(threshold_diff)]
        fnr_at_threshold = fnrs[np.ndarray.argmin(threshold_diff)]

    return fpr_at_threshold, fnr_at_threshold


def compute_cdet_at_threshold(fprs, fnrs, thresholds, threshold_value, dcf_p_target, dcf_c_fp, dcf_c_fn):
    """Computation of detection cost function at a given threshold. Computation is performed as defined in the `NIST Speaker Recognition Evaluation Plan 2019 <https://www.nist.gov/itl/iad/mig/nist-2019-speaker-recognition-evaluation>`_

    .. math:: C_{Det}(\\theta) = C_{FN} \\times P_{Target} \\times P_{FN}(\\theta) + C_{FP} \\times (1 - P_{Target}) \\times P_{FP}(\\theta)

    :param fprs: Array of False Positive Rates
    :type fprs: ndarray
    :param fnrs: Array of False Negative Rates
    :type fnrs: ndarray
    :param thresholds: Array of Threshold values corresponding to fprs and fnrs
    :type thresholds: ndarray
    :param threshold_value: Threshold value to compute detection cost function for
    :type threshold_value: float
    :param dcf_p_target: [description]
    :type dcf_p_target: float
    :param dcf_c_fp: [description]
    :type dcf_c_fp: float
    :param dcf_c_fn: [description]
    :type dcf_c_fn: float

    :returns: cdet_at_threshold
    :rtype: float

    """

    fpr_at_threshold, fnr_at_threshold = get_fpfn_at_threshold(fprs, fnrs, thresholds, threshold_value)
    cdet_at_threshold = fpr_at_threshold * dcf_c_fp * (1 - dcf_p_target) + fnr_at_threshold * dcf_c_fn * dcf_p_target

    return cdet_at_threshold


#########################################
# In this section we compute bias measures based on ratios and differences of performance metrics
#########################################

class BiasMeasures:

    def __init__(self, metrics_df:pd.DataFrame, group_names:list):

        self.metrics_df = metrics_df
        self.group_names = group_names

        return

    def subgroup_to_average_ratio(self):

        df = self.metrics_df.set_index(['group_name','group_category'])
        ratios = df.loc[self.group_names].div(df.loc[['average']].values[0], axis=1)
        ratios.reset_index(inplace=True)

        return ratios

    def log_subgroup_to_average_ratio(self):

        ratios = self.subgroup_to_average_ratio()
        ratios.set_index(['group_name','group_category'], inplace=True)

        # NB: ln(0) = -infinity -> by replacing infinity with 0, we treat the ratio in the same way as a ratio of 1 (i.e. subgroup performance equals average performance)
        # As all our metrics are error rates, a score of 0 indicates top performance. So this approach will not account for some cases of favoritism
        # i.e. when a subgroup has a 0 score, it has to be better performance than average (unless average is also 0). However, we treat it as equal performance.
        log_ratios = np.log(ratios, where=ratios != 0)
        log_ratios.reset_index(inplace=True)

        return log_ratios

    def subgroup_to_average_difference(self):

        df = self.metrics_df.set_index(['group_name','group_category'])
        differences = df.loc[self.group_names].sub(df.loc[['average']].values[0], axis=1)
        differences.reset_index(inplace=True)

        return differences

    def absolute_subgroup_to_average_difference(self):

        differences = self.subgroup_to_average_difference()
        differences.set_index(['group_name','group_category'], inplace=True)
        abs_differences = np.abs(differences)
        abs_differences.reset_index(inplace=True)

        return abs_differences

    def subgroup_distance_to_group_min(self):

        df = self.metrics_df.set_index(['group_name','group_category'])
        dist_to_min = df.loc[self.group_names] - df.loc[self.group_names].groupby('group_name').transform('min')
        dist_to_min.reset_index(inplace=True)

        return dist_to_min


# Meta-measures

def fairness_discrepancy_rate(fpr_dist_to_min:pd.Series, fnr_dist_to_min:pd.Series, alpha:float):
    """
    This function implements the fairness discrepancy rate (FDR) measure introduced in the paper
    'Fairness in Biometrics: A Figure of Merit to Assess Biometric Verification Systems' https://doi.org/10.1109/TBIOM.2021.3102862

    Use BiasMeasures.subgroup_distance_to_group_min() to get fpr_dist_to_min and fnr_dist_to_min. This guarantess that distances values
    are always positive, and no absolute value needs to be taken. Must select on threshold and one group_name for which to calculate the fdr.
    """

    max_A = max(fpr_dist_to_min)
    max_B = max(fnr_dist_to_min)

    fdr = 1 - (alpha*max_A + (1 - alpha)*max_B)

    return fdr

def reliability_bias(metric_log_ratios:pd.Series, norm=True, weights:pd.Series=None):
    """
    This function implements the reliablity bias measure as introduced in the paper
    'Tiny, always-on and fragile: Bias propagation through design choices in on-device machine learning workflows' https://arxiv.org/abs/2201.07677

    Use BiasMeasures.log_subgroup_to_average_ratio() to get metric_log_ratios. Must select on metric (i.e. column) and one group_name
    for which to calculate reliability bias.
    """

    if weights is not None:
        metric_log_ratios = metric_log_ratios*weights

    reliability_bias = sum(abs(metric_log_ratios))

    if norm is True:
        reliability_bias = reliability_bias/len(metric_log_ratios)

    return reliability_bias