#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 18-05-2021
# @author: wiebket

import sklearn.metrics as sklearn_metrics
from .metrics import compute_eer, compute_min_cdet, get_fpfn_at_threshold, get_fnthreshold_at_fp, compute_cdet_at_threshold
import pdb


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


def evaluate_scores(scores, labels, fpr_values, dcf_costs, threshold_values):
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

    keys = ["EER"]
    eer, eer_threshold = compute_eer(fprs, fnrs, thresholds)
    metric_scores.append(eer)
    metric_thresholds.append(eer_threshold)
    
    # `Average` evaluation case --> returns threshold for metric
    if threshold_values is None:
        if fpr_values is not None:
            for fpr_val in fpr_values:
                fnr_at_fpr, threshold_at_fpr = get_fnthreshold_at_fp(fprs, fnrs, thresholds, fpr_val, ppf_norm=False)
                metric_scores.extend([fpr_val, fnr_at_fpr])
                metric_thresholds.extend([threshold_at_fpr, threshold_at_fpr])
                keys.extend(["FPR@fpr" + str(fpr_val), "FNR@fpr" + str(fpr_val)])
        
        if dcf_costs is not None:
            for cost in dcf_costs:
                min_cdet, min_cdet_threshold = compute_min_cdet(fprs, fnrs, thresholds, cost[0], cost[1], cost[2])
                fpr_at_threshold, fnr_at_threshold = get_fpfn_at_threshold(fprs, fnrs, thresholds, min_cdet_threshold, ppf_norm=False)
                metric_scores.extend([min_cdet, fpr_at_threshold, fnr_at_threshold])
                metric_thresholds.extend([min_cdet_threshold, min_cdet_threshold, min_cdet_threshold])
                keys.extend(["minCDet" + str(cost), "FPR@minCDet" + str(cost), "FNR@minCDet" + str(cost)])
                
        metric_scores_dict = dict(zip(keys, metric_scores))
        metric_thresholds_dict = dict(zip(keys, metric_thresholds))

        return fprs, fnrs, thresholds, metric_scores_dict, metric_thresholds_dict
    
    # Group evaluation case -> does not return threshold for metric
    else:
        for k, v in threshold_values.items():
            if "FPR@fpr" in k:
                fpr_at_threshold, fnr_at_threshold = get_fpfn_at_threshold(fprs, fnrs, thresholds, v, ppf_norm=False)
                metric_scores.extend([fpr_at_threshold, fnr_at_threshold])
                
            elif "FPR@minCDet" in k:
                cost = [float(i) for i in k.split('(')[-1].split(')')[0].split(',')]
                cdet_at_threshold = compute_cdet_at_threshold(fprs, fnrs, thresholds, v, cost[0], cost[1], cost[2])
                fpr_at_threshold, fnr_at_threshold = get_fpfn_at_threshold(fprs, fnrs, thresholds, v, ppf_norm=False)
                metric_scores.extend([cdet_at_threshold, fpr_at_threshold, fnr_at_threshold])

        metric_scores_dict = dict(zip(threshold_values.keys(), metric_scores))

        return fprs, fnrs, thresholds, metric_scores_dict
