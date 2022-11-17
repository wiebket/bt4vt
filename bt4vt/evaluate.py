#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 18-05-2021
# @author: wiebket

import sklearn.metrics as sklearn_metrics
from .metrics import compute_eer, compute_min_cdet, compute_cdet_at_threshold


def compute_fpfnth(scores, labels):
    """ Calculation of False Positive Rates and False Negative Rates and corresponding thresholds

    :param scores: Series of scores
    :type scores: pandas.Series
    :param labels: Series of labels; labels have to be either {-1,1} or {0,1}
    :type labels: pandas.Series

    :returns: fprs, fnrs, thresholds
    :rtype: ndarray, ndarray, ndarray

    """

    fprs, fnrs, thresholds = sklearn_metrics.det_curve(labels, scores, pos_label=1)

    return fprs, fnrs, thresholds


def evaluate_scores(scores, labels, dcf_costs, threshold_values=None):
    """ Evaluation of scores for the overall dataset and for specified speaker groups. In the average case no threshold_values are provided.
        Threshold values are used to compute the detection cost function for specified speaker groups.
        The function returns False Positive Rates, False Negative Rates and corresponding thresholds as well as the corresponding metric scores. In the average case, metric thresholds are returned in addition.

        :param scores: Series of scores
        :type scores: pandas.Series
        :param labels: Series of labels
        :type labels: pandas.Series
        :param dcf_costs: list of tuples specifying the weights for the detection cost function (dcf_p_target, dcf_c_fp, dcf_c_fn)
        :type dcf_costs: list
        :param threshold_values: Series of threshold values computed for the overall dataset and used to determine the metric scores for the specified speaker groups
        :type threshold_values: pandas.Series

        :returns: fprs, fnrs, thresholds, metric_scores, (metric_thresholds)
        :rtype: ndarray, ndarray, ndarray, list, (list)

    """

    fprs, fnrs, thresholds = compute_fpfnth(scores, labels)

    metric_scores = []
    metric_thresholds = []

    eer, eer_threshold = compute_eer(fprs, fnrs, thresholds)
    metric_scores.append(eer)
    metric_thresholds.append(eer_threshold)
    # TODO: error handling check that dcf_cost is not empty
    # this is the average case
    if threshold_values is None:
        for cost in dcf_costs:
            min_cdet, min_cdet_threshold = compute_min_cdet(fprs, fnrs, thresholds, cost[0], cost[1], cost[2])
            metric_scores.append(min_cdet)
            metric_thresholds.append(min_cdet_threshold)

        return fprs, fnrs, thresholds, metric_scores, metric_thresholds
    # this is the group case
    else:
        # TODO error handling check that threshold_values is length(dcf_costs) + 2 as first one refers to subgroup and second to eer
        for index, cost in enumerate(dcf_costs):
            cdet_at_threshold = compute_cdet_at_threshold(fprs, fnrs, thresholds, threshold_values[index + 2], cost[0],
                                                          cost[1], cost[2])
            metric_scores.append(cdet_at_threshold)

        return fprs, fnrs, thresholds, metric_scores
