#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 18-05-2021
# @author: wiebket

import sklearn.metrics as sklearn_metrics
from .metrics import compute_eer, compute_min_cdet, compute_cdet_at_threshold


def compute_fpfnth(scores, labels):
    """ Calculate:
    1. false negative (false reject) and false positive (false accept) rates
    """
    # Calculate false negative (false reject) and false positive (false accept) rates
    fprs, fnrs, thresholds = sklearn_metrics.det_curve(labels, scores, pos_label=1)

    return fprs, fnrs, thresholds


def evaluate_scores(score, label, dcf_costs, threshold_values=None):
    fprs, fnrs, thresholds = compute_fpfnth(score, label)

    metric_scores = []
    metric_thresholds = []

    eer, eer_threshold = compute_eer(fprs, fnrs, thresholds)
    metric_scores.append(eer)
    metric_thresholds.append(eer_threshold)
    # TODO: error handling check that dcf_cost is not empty
    # this is the overall case
    if threshold_values is None:
        for cost in dcf_costs:
            min_cdet, min_cdet_threshold = compute_min_cdet(fprs, fnrs, thresholds, cost[0], cost[1], cost[2])
            metric_scores.append(min_cdet)
            metric_thresholds.append(min_cdet_threshold)

        return fprs, fnrs, thresholds, metric_scores, metric_thresholds
    # this is the group case
    else:
        # TODO error handling check that threshold_values is length(dcf_costs) + 1
        for index, cost in enumerate(dcf_costs):
            cdet_at_threshold = compute_cdet_at_threshold(fprs, fnrs, thresholds, threshold_values[index + 1], cost[0],
                                                          cost[1], cost[2])
            metric_scores.append(cdet_at_threshold)

        return fprs, fnrs, thresholds, metric_scores
