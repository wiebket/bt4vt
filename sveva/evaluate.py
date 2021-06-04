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



def _subgroup(df, filter_dict:dict):
    """
    Filter dataframe by a demographic subgroup.
    
    ARGUMENTS
    ---------
    df [dataframe]: 
    group_filter [dict]:
    
    OUTPUT
    ------
    """

    filters = ' & '.join([f"{k}=='{v}'" for k, v in filter_dict.items()])
    sg = df.query(filters)

    return(sg[['sc','lab']+list(filter_dict.keys())])



def fnfpth_metrics(df, dcf_p_target=0.05, dcf_c_fn=1, dcf_c_fp=1):
    """
    Calculate 
    
    ARGUMENTS
    ---------
    df [dataframe]: results dataframe with columns ['sc'] (scores), ['lab'] (labels)
    dcf_p_target [float]:
    dcf_c_fn [float]:
    dcf_c_fp [float]:
    
    OUTPUT
    ------
    
    """

    if len(df)>0:
        df_eval = _evaluate_sv(df['sc'], df['lab'], dcf_p_target, dcf_c_fn, dcf_c_fp)
        df_fnfpth = pd.DataFrame(data={'fnrs':df_eval[0],
                                       'fprs':df_eval[1],
                                       'thresholds':df_eval[2]})
        
        df_metrics = {'min_cdet':df_eval[3],'min_cdet_threshold':df_eval[4],'eer':df_eval[5],'eer_threshold':df_eval[6]}       

        return(df_fnfpth, df_metrics)

    else:
        return(None)  #TO DO: silent pass --> consider response
    
    
    
def sg_fnfpth_metrics(df, filter_keys:list, **kwargs): #TO DO: generalise to group_filter: dict
    """
    This function returns false negative rates, false positive rates and the 
    corresponding threshold values for scores in dataset df. 
    
    ARGUMENTS
    ---------
    df [dataframe]:
        df['sc']: scores
        df['lab']: binary labels (0=False, 1=True)
    filter_keys [list]:
    
    valid **kwargs
    --------------
    dcf_p_target [float]: detection cost function target (default = 0.05)
    dcf_c_fn [float]: detection cost function false negative weight (default = 1)
    dcf_c_fp [float]: detection cost function  false positive weight (default = 1)
    
    OUTPUT
    ------
    fnfpth [dataframe]:
    metrics [dictionary]:
    """

    filter_dict = {}
    
    for f_key in filter_keys:
        f_vals = list(df[f_key].unique())
        f_vals.sort()
        filter_dict[f_key] = f_vals

    filter_items = []

    for val0 in filter_dict[filter_keys[0]]:
        try:
            for val1 in filter_dict[filter_keys[1]]:
                try:
                    for val2 in filter_dict[filter_keys[2]]:
                        f_item = {filter_keys[0]:val0, filter_keys[1]:val1, filter_keys[2]:val2}
                        filter_items.append(f_item)
                except IndexError:
                    f_item = {filter_keys[0]:val0, filter_keys[1]:val1}
                    filter_items.append(f_item)
        except IndexError:
            f_item = {filter_keys[0]:val0}
            filter_items.append(f_item)

    fnfpth_list = []
    metrics = {}

    for fi in filter_items:
        try:
            sg = _subgroup(df, fi)
            sg_fnfpth, sg_metrics = fnfpth_metrics(sg)
            for key, val in fi.items():
                sg_fnfpth[key] = val
            fnfpth_list.append(sg_fnfpth)
            sg_name = '_'.join([v.replace(" ", "").lower() for v in fi.values()])
            metrics[sg_name] = sg_metrics
        except:
            print('Failed to filter by: ', fi.values())
            pass

    fnfpth = pd.concat(fnfpth_list)

    return(fnfpth, metrics)



def compare_experiments(experiment_dict:dict, comparison:str):
    """
    
    ARGUMENTS
    ---------
    
    """
    
    compare_df = []
    compare_metrics = {}
    
    for k, v in experiment_dict.items():
        df = v[0]
        df[comparison] = k
        compare_metrics[k] = v[1]
        compare_df.append(df)
    
    compare_fnfpth = pd.concat(compare_df)
    
    return compare_fnfpth, compare_metrics