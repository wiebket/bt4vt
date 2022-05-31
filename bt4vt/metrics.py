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
# In this section we compute bias metrics
# 1. Ratio of group mincdet / overall mincdet
# 2. Ratio of group fp, fn rates / overall fp, fn rates
#########################################

def compute_metrics_ratios(metrics):
    """Computation of metric ratios defined as the subgroup metric scores divided by the overall metric score.

    :param metrics: DataFrame that contains metric scores for the overall evaluation and subgroup evaluations. The first row corresponds to the eer, all other rows correspond to the min_cdet scores with weights specified in the config file
    :type metrics: DataFrame

    :returns: metric_ratios
    :rtype: DataFrame

    """
    metrics_ratios = metrics.iloc[:, 1:].div(metrics['overall'], axis=0)

    return metrics_ratios


def compute_fpfn_ratio(fpfnth, metrics, metrics_baseline, filter_keys: list, threshold_type):

    # nice function for future to understand the real life impact of threshold settings

    return