#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 18-05-2021
# @author: wiebket

import sklearn.metrics as sklearn_metrics


def compute_fpfnth(scores, labels):
    """ Calculate:
    1. false negative (false reject) and false positive (false accept) rates
    """

    # Calculate false negative (false reject) and false positive (false accept) rates
    fprs, fnrs, thresholds = sklearn_metrics.det_curve(labels, scores, pos_label=1)

    return(fprs, fnrs, thresholds)


def evaluate_scores(score, label, dcf_costs, threshold_values=None):

    fprs, fnrs, thresholds = compute_fpfnth(score, label)

    # TODO calculate metrics
    # 1. calculate eer, eer_threshold = compute_eer()
    # if threshold_values=None:
        # 2. loop over dcf_costs:
            # 3. min_cdet, min_cdet_threshold = compute_min_cdet
        # metric_scores append eer and cdet, metric_thresholds append eer_threshold and cdet_thresholds
        #return fprs, fnrs, thresholds, metric_scores, metric_thresholds
    # else:
        # calculate eer, eer_threshold = compute_eer()
        # loop over dcf_costs:
            # select threshold_value from threshold_values
            # cdet_at_threshold = compute_cdet_at_threshold(fprs, fnrs,thresholds, threshold_value, dcf_costs)
            # metric_scores append eer and cdet,
        #return fprs, fnrs, thresholds, metrics_scores

    return