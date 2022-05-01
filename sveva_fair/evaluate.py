#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 18-05-2021
# @author: wiebket

import pandas as pd
import numpy as np
import scipy as sp
import sklearn.metrics as sklearn_metrics


def _compute_min_cdet(fnrs, fprs, thresholds, dcf_p_target, dcf_c_fn, dcf_c_fp):
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

    cdet = np.array([fnr*dcf_c_fn*dcf_p_target for fnr in fnrs]) + np.array([fpr*dcf_c_fp*(1-dcf_p_target) for fpr in fprs])
    min_ix = np.nanargmin(cdet)
    min_cdet = cdet[min_ix] # NB: divide by 0.05 to get results that correspond with VoxCeleb benchmark
    min_cdet_threshold = thresholds[min_ix]

    return(min_cdet, min_cdet_threshold)



def _evaluate_sv(sc, lab, dcf_p_target, dcf_c_fn, dcf_c_fp):
    """ Calculate:
    1. false negative (false reject) and false positive (false accept) rates
    2. minimum of the detection cost function and corresponding threshold score value
    3. equal error rate and corresponding threshold score value

    :param sc: [description]
    :type sc: [type]
    :param lab: [description]
    :type lab: [type]
    :param dcf_p_target: [description]
    :type dcf_p_target: [type]
    :param dcf_c_fn: [description]
    :type dcf_c_fn: [type]
    :param dcf_c_fp: [description]
    :type dcf_c_fp: [type]

    :returns: fnrs, fprs, thresholds, min_cdet, min_cdet_threshold, eer, eer_threshold

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
    """Filter dataframe df by a demographic subgroup defined by filter_dict.

    :param df: dataframe with speaker verification predictions that has columns, df['sc'] scores, df['lab'] binary labels (0=False, 1=True)
    :type df: dataframe
    :param filter_dict: filter specifying column names and unique values by which to filter e.g. {'nationality':'India', 'sex':'female'} will select all rows where df['nationality']=='India' and df['sex']=='female'
    
    :returns: Subgroup dataframe with prediction scores ['sc'], labels ['lab'] and filter columns

    """

    filters = ' & '.join([f"{k}=='{v}'" for k, v in filter_dict.items()])
    sg = df.query(filters)

    return(sg[['sc','lab']+list(filter_dict.keys())])



def fpfnth_metrics(df, dcf_p_target=0.05, dcf_c_fn=1, dcf_c_fp=1):
    """
    This function returns false negative rates, false positive rates and the corresponding 
    threshold values for scores in dataset df.  
    
    :param df: dataframe with speaker verification predictions that has columns `df['sc']: scores` and `df['lab']: binary labels (0=False, 1=True)`
    :type df: pandas.DataFrame
    :param dcf_p_target: detection cost function target (default = 0.05)
    :type dcf_p_target: float
    :param dcf_c_fn: detection cost function false negative weight (default = 1)
    :type dcf_c_fn: float
    :param dcf_c_fp: detection cost function  false positive weight (default = 1)
    :type dcf_c_fp: float

    :returns: fpfnth, metrics -- dataframe with the following columns: 'fnrs','fprs','thresholds' and minimum detection cost and equal error rate metrics
    :rtype: dataframe, dict

    """

    if len(df)>0:
        df_eval = _evaluate_sv(df['sc'], df['lab'], dcf_p_target, dcf_c_fn, dcf_c_fp)
        df_fpfnth = pd.DataFrame(data={'fnrs':df_eval[0],
                                       'fprs':df_eval[1],
                                       'thresholds':df_eval[2]})
        
        df_metrics = {'min_cdet':df_eval[3],'min_cdet_threshold':df_eval[4],'eer':df_eval[5],'eer_threshold':df_eval[6]}       

        return(df_fpfnth, df_metrics)

    else:
        return(None)  #TODO: silent pass --> consider response
    
    
    
def sg_fpfnth_metrics(df, filter_keys:list, dcf_p_target=0.05, dcf_c_fn=1, dcf_c_fp=1):
    """
    This function returns false negative rates, false positive rates and threshold values
    for subgroups in dataframe df. Subgroups are defined by the column names specified in 
    the filter_keys argument.

    :param df: dataframe with speaker verification predictions that has columns, df['sc'] scores, df['lab'] binary labels (0=False, 1=True)
    :type df: dataframe
    :param filter_keys: list of column names in df that define subgroups, e.g. ['nationality','sex'], Creates subgroups from all unique value pairs in the designated columns. Currently supports lists with 1, 2 or 3 items.
    :type filter_keys: list
    :param dcf_p_target: detection cost function target (default = 0.05)
    :type dcf_p_target: float
    :param dcf_c_fn: detection cost function false negative weight (default = 1)
    :type dcf_c_fn: float
    :param dcf_c_fp: detection cost function  false positive weight (default = 1)
    :type dcf_c_fp: float

    :returns: fpfnth, metrics dataframe with the following columns 'fnrs','fprs','thresholds','subgroup' and minimum detection cost and equal error rate metrics for each subgroup
    :rtype: dataframe, dict

    """

    # Create a dictionary to construct subgroups from filter_keys {filter_key: [unique filter_key values], }
    filter_dict = {}
    
    for f_key in filter_keys:
        f_vals = list(df[f_key].unique())
        f_vals.sort()
        filter_dict[f_key] = f_vals

    # Create a list of dictionaries of potential subgroups from all combinations of values in filter_dict
    # [{filter_keys[0]: filter_keys[0] first unique value, filter_keys[1]: filter_keys[1] first unique value}, 
    #  {filter_keys[0]: filter_keys[0] second unique value, filter_keys[1]: filter_keys[1] first unique value}, ...]
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

    fpfnth_list = []
    metrics = {}

    for fi in filter_items:
        try:
            sg = _subgroup(df, fi)
            sg_fpfnth, sg_metrics = fpfnth_metrics(sg, dcf_p_target, dcf_c_fn, dcf_c_fp)
            for key, val in fi.items():
                sg_fpfnth[key] = val
            sg_name = '_'.join([v.replace(" ", "").lower() for v in fi.values()])
            sg_fpfnth['subgroup'] = sg_name
            fpfnth_list.append(sg_fpfnth)
            metrics[sg_name] = sg_metrics
        except:
            pass

    fpfnth = pd.concat(fpfnth_list, ignore_index=True)

    return(fpfnth, metrics)



def fpfn_min_threshold(df, threshold, ppf_norm=False):
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

    # Find the index in df that is closest to the SUBGROUP minimum threshold value
    sg_threshold_diff = np.array([abs(i - threshold) for i in df['thresholds']])
    
    if ppf_norm == True:
        min_threshold_fpr = sp.stats.norm.ppf(df['fprs'])[np.ndarray.argmin(sg_threshold_diff)]
        min_threshold_fnr = sp.stats.norm.ppf(df['fnrs'])[np.ndarray.argmin(sg_threshold_diff)]
    else:
        min_threshold_fpr = df['fprs'].iloc[np.ndarray.argmin(sg_threshold_diff)]  
        min_threshold_fnr = df['fnrs'].iloc[np.ndarray.argmin(sg_threshold_diff)]      

    norm_threshold = [min_threshold_fpr, min_threshold_fnr]

    return norm_threshold



def cdet_ratio(fpfnth, metrics, metrics_baseline, filter_keys:list, dcf_p_target=0.05, dcf_c_fn=1, dcf_c_fp=1):
    """[summary]

    :param fpfnth: [description]
    :type fpfnth: [type]
    :param metrics: [description]
    :type metrics: [type]
    :param metrics_baseline: [description]
    :type metrics_baseline: [type]
    :param filter_keys: [description]
    :type filter_keys: list
    :param dcf_p_target: [description]. Defaults to 0.05.
    :type dcf_p_target: (float, optional)
    :param dcf_c_fn: [description]. Defaults to 1.
    :type dcf_c_fn: (int, optional)
    :param dcf_c_fp: [description]. Defaults to 1.
    :type dcf_c_fp: (int, optional)

    :returns: [description]
    :rtype: [type]

    """    

    fpfn_list = []
    for k in metrics.keys():

        # Calculate cdet of subgroup at the overall minimum threshold value
        sg = k.split('_')
        sg_fpfnth = fpfnth[(fpfnth[filter_keys[0]].apply(lambda x: x.replace(" ", "").lower())==sg[0]) & 
                              (fpfnth[filter_keys[1]]==sg[1])]
        sg_fpfn_overall_min_cdet = fpfn_min_threshold(sg_fpfnth, metrics_baseline['min_cdet_threshold'])
        sg_cdet_overall_min = sg_fpfn_overall_min_cdet[0]*dcf_c_fp*(1-dcf_p_target) + sg_fpfn_overall_min_cdet[1]*dcf_c_fn*dcf_p_target        
 
        fpfn_list.append([k, sg_cdet_overall_min, metrics[k]['min_cdet']])

    fpfn_df = pd.DataFrame(fpfn_list, columns=['subgroup','sg_cdet_overall_min','sg_min_cdet'])
    
    # Calculate cdet ratios for subgroups
    fpfn_df['overall_cdet_ratio'] = fpfn_df['sg_cdet_overall_min']/metrics_baseline['min_cdet']
    fpfn_df['sg_cdet_ratio'] = fpfn_df['sg_min_cdet']/fpfn_df['sg_cdet_overall_min']    
    
    return fpfn_df



def fpfn_ratio(fpfnth, metrics, metrics_baseline, filter_keys:list, threshold_type):
    """[summary]

    :param fpfnth: [description]
    :type fpfnth: [type]
    :param metrics: [description]
    :type metrics: [type]
    :param metrics_baseline: [description]
    :type metrics_baseline: [type]
    :param filter_keys: [description]
    :type filter_keys: list
    :param threshold_type: [description]
    :type threshold_type: [type]

    :returns: [description]
    :rtype: [type]

    """

    fpfn_list = []
    for k in metrics.keys():
        
        # Calculate FPR/FNR at the overall minimum threshold value
        overall_min_cdet = fpfn_min_threshold(fpfnth, metrics_baseline[threshold_type])

        # Calculate FPR/FNR of subgroup at the overall minimum threshold value
        sg = k.split('_')
        sg_fpfnth = fpfnth[(fpfnth[filter_keys[0]].apply(lambda x: x.replace(" ", "").lower())==sg[0]) & 
                              (fpfnth[filter_keys[1]]==sg[1])]
        sg_fpfn_overall_min_cdet = fpfn_min_threshold(sg_fpfnth, metrics_baseline[threshold_type])
        
        # Calculate FPR/FNR of subgroup at the subgroup minimum threshold value
        sg_fpfn_min_cdet = fpfn_min_threshold(sg_fpfnth, metrics[k][threshold_type])

        fpfn_list.append([k, sg_fpfn_overall_min_cdet[0], sg_fpfn_overall_min_cdet[1], sg_fpfn_min_cdet[0], sg_fpfn_min_cdet[1]])

    fpfn_df = pd.DataFrame(fpfn_list, columns=['subgroup','overall_fpr','overall_fnr','sg_fpr','sg_fnr'])
    
    fpfn_df['overall_fpr_ratio'] = fpfn_df['overall_fpr']/sg_fpfn_overall_min_cdet[0]
    fpfn_df['overall_fnr_ratio'] = fpfn_df['overall_fnr']/sg_fpfn_overall_min_cdet[1]
    fpfn_df['sg_fpr_ratio'] = fpfn_df['sg_fpr']/fpfn_df['overall_fpr']
    fpfn_df['sg_fnr_ratio'] = fpfn_df['sg_fnr']/fpfn_df['overall_fnr']
    
    return fpfn_df



def compare_experiments(experiment_dict:dict, comparison:str):
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
        min_sc = df[(df['subgroup']==subgroup) & (df['lab']==1)]['sc'].min()
        max_sc = df[(df['subgroup']==subgroup) & (df['lab']==0)]['sc'].max()
        overlap_fn = df[(df['subgroup']==subgroup) & (df['lab']==1) & (df['sc'] >= min_sc) & (df['sc'] <= eer_threshold)]['sc'].count()
        overlap_fp = df[(df['subgroup']==subgroup) & (df['lab']==0) & (df['sc'] >= eer_threshold) & (df['sc'] <= max_sc)]['sc'].count()
        overlap_total = overlap_fn + overlap_fp
        total_instances = df[df['subgroup']==subgroup]['sc'].count()
        overlap_probability = overlap_total / total_instances
        
        score_overlap[subgroup] = overlap_probability
        
    return score_overlap