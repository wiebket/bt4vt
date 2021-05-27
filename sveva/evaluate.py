#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 18-05-2021
@author: wiebket
"""

import pandas as pd
import numpy as np
import scipy as sp
import sklearn.metrics as sklearn_metrics


def _compute_min_cdet(fnrs, fprs, thresholds, dcf_p_target, dcf_c_fn, dcf_c_fp):
    """
    Compute the minimum of the detection cost function as defined in the 
    NIST Speaker Recognition Evaluation Plan 2019.
    """

    cdet = np.array([fnr*dcf_c_fn*dcf_p_target for fnr in fnrs]) + np.array([fpr*dcf_c_fp*(1-dcf_p_target) for fpr in fprs])
    min_ix = np.nanargmin(cdet)
    min_cdet = cdet[min_ix]
    min_cdet_threshold = thresholds[min_ix]

    return(min_cdet, min_cdet_threshold)



def _evaluate_sv(sc, lab, dcf_p_target, dcf_c_fn, dcf_c_fp):
    """
    Calculate:
    1. false negative (false reject) and false positive (false accept) rates
    2. minimum of the detection cost function and corresponding threshold score value
    3. equal error rate and corresponding threshold score value
    """

    # Calculate false negative (false reject) and false positive (false accept) rates
    fprs, fnrs, thresholds = sklearn_metrics.det_curve(lab, sc, pos_label=1)

    # Calculate the minimum of the detection cost function and corresponding threshold
    min_cdet, min_cdet_threshold = _compute_min_cdet(fnrs, 
                                                  fprs, 
                                                  thresholds, 
                                                  dcf_p_target, 
                                                  dcf_c_fn, 
                                                  dcf_c_fp)

    # Calculate the equal error rate and its threshold
    min_ix = np.nanargmin(np.absolute((fnrs - fprs)))
    eer  = max(fprs[min_ix], fnrs[min_ix])*100
    eer_threshold = thresholds[min_ix] 

    return(fnrs, fprs, thresholds, min_cdet, min_cdet_threshold, eer, eer_threshold)



def _subgroup(df, group_filter: dict):
    """
    Filter scores by a demographic subgroup.
    """

    filters = ' & '.join([f"{k}=='{v}'" for k, v in group_filter.items()])
    sg = df.query(filters)

    return(sg[['sc','lab']+list(group_filter.keys())])



def _eval_subgroup(df, group_filter: dict, dcf_p_target=0.05, dcf_c_fn=1, dcf_c_fp=1):
    """
    Calculate 
    """

    sg = _subgroup(df, group_filter)

    if len(sg)>0:
        sg_eval = _evaluate_sv(sg['sc'], sg['lab'], dcf_p_target, dcf_c_fn, dcf_c_fp)
        sg_fnfpth = pd.DataFrame(data={'fnrs':sg_eval[0],
                                       'fprs':sg_eval[1],
                                       'thresholds':sg_eval[2]})

        sg_fnfpth['subgroup'] = '_'.join(i 
                                      for i in [''.join(v.split()).lower() 
                                      for v in group_filter.values()])

        for key, value in group_filter.items():
            sg_fnfpth[key] = value

            sg_metrics = {}
            sg_metrics['min_cdet'] = sg_eval[3]
            sg_metrics['min_cdet_threshold'] = sg_eval[4]
            sg_metrics['eer'] = sg_eval[5]
            sg_metrics['eer_threshold'] = sg_eval[6]

        return(sg_fnfpth, sg_metrics)

    else:
        return(None)  #TO DO: silent pass --> consider response
    
    
    
def fnfpth(df, **kwargs): #TO DO: generalise to group_filter: dict
    """
    This function returns false negative rates, false positive rates and the 
    corresponding threshold values for scores in dataset df. 
    
    ARGUMENTS
    ---------
    df [dataframe]:
        df['sc']: scores
        df['lab']: binary labels (0=False, 1=True)
    
    valid **kwargs
    --------------
    ref_nationality [list]: one or more unique reference nationalities in df
    ref_gender [list]: one or more unique reference genders in df
    dcf_p_target [float]: detection cost function target (default = 0.05)
    dcf_c_fn [float]: detection cost function false negative weight (default = 1)
    dcf_c_fp [float]: detection cost function  false positive weight (default = 1)
    
    OUTPUT
    ------
    all_fnfpth [dataframe]:
    all_metrics [dictionary]:
    """

    # Get ref_nationality, ref_gender from keyword arguments. Use all unique values if not specified.
    ref_nationality = kwargs.get('ref_nationality', list(df['ref_nationality'].unique()))
    ref_gender = kwargs.get('ref_gender', list(df['ref_gender'].unique()))
    #assert that ref_nationality, ref_gender are lists
    ref_nationality.sort()
    ref_gender.sort()

    # Get fnrs, fprs, thresholds for all values
    eval_all = _evaluate_sv(df['sc'], 
                         df['lab'], 
                         dcf_p_target=kwargs.get('dcf_p_target',0.05),
                         dcf_c_fn=kwargs.get('dcf_c_fn',1),
                         dcf_c_fp=kwargs.get('dcf_c_fp',1)
                         )

    fnfpth_list = [pd.DataFrame(data={'fnrs':eval_all[0], 'fprs':eval_all[1], 'thresholds':eval_all[2], 'subgroup':'all'})]
    metrics = {'all':{'min_cdet':eval_all[3],'min_cdet_threshold':eval_all[4],'eer':eval_all[5],'eer_threshold':eval_all[6]}}

    for nat in ref_nationality:
        for gen in ref_gender:
            try:
                sg_fnfpth, sg_metrics = _eval_subgroup(df, {'ref_nationality':nat,'ref_gender':gen})
                fnfpth_list.append(sg_fnfpth)
                sg_name = sg_fnfpth['subgroup'].unique()[0]
                metrics[sg_name] = sg_metrics
            except TypeError as e:
                print('Failed to filter by: ', nat, ' ', gen, ': ', e)
                pass

    fnfpth = pd.concat(fnfpth_list)

    return fnfpth, metrics