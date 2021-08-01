#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 18-05-2021
@author: wiebket
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from .evaluate import fpfn_min_threshold



def plot_det_curves(fpfnth, **kwargs):
    """
    
    ARGUMENTS
    ---------
    fpfnth [dataframe]: dataframe, must contain false negative rates ['fnrs'] and false positive rates ['fprs']. 
                        If **kwargs are specified, then dataframe must contain columns with names used for values in hue, style and col (see below)
    **kwargs: valid options: hue, style, col, palette and linewidth. Passed to seaborn.relplot()
    """

    kwargs_hue = kwargs.get('hue', None)
    kwargs_style = kwargs.get('style', None)
    kwargs_col = kwargs.get('col', None)
    kwargs_palette= kwargs.get('palette', 'colorblind')
    kwargs_lw = kwargs.get('linewidth', 2)
    try:
        n_kwargs_col = len(fpfnth[kwargs_col].unique())
        n_col_wrap = 3 if n_kwargs_col >=3 else n_kwargs_col
    except KeyError:
        n_col_wrap = None

    g = sns.relplot(x=sp.stats.norm.ppf(fpfnth['fprs']), 
                    y=sp.stats.norm.ppf(fpfnth['fnrs']), 
                    hue=kwargs_hue,
                    style=kwargs_style,
                    col=kwargs_col,
                    col_wrap=n_col_wrap,
                    palette=kwargs_palette,
                    linewidth= kwargs_lw,
                    height=4, aspect=1.3, 
                    kind='line', 
                    facet_kws=dict(sharex=False,sharey=False),
                    data=fpfnth)

    ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
    tick_locations = sp.stats.norm.ppf(ticks)
    tick_labels = [
      '{:.0%}'.format(s) if (100*s).is_integer() else '{:.1%}'.format(s)
      for s in ticks
    ]

    for ax in g.axes.flat:
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(-3.5, 1)
        ax.set_yticks(tick_locations)
        ax.set_yticklabels(tick_labels)
        ax.set_ylim(-3.5, 0.5)

    g.set_axis_labels('false positive rate', 'false negative rate', labelpad=10)

    return g



def plot_det_baseline(g, fpfnth_baseline, metrics_baseline, threshold_type, **kwargs):
    """
    Add a baseline DET curve to every DET curve subplot of an existing seaborn FacetGrid.
    
    ARGUMENTS
    ---------
    g [FacetGrid]: created for example with plot_det_curves()
    fpfnth_baseline [dataframe]: dataframe, must contain false negative rates ['fnrs'], false positive rates ['fprs'] and threshold values ['thresholds']
    threshold_value [float]: score at threshold 'min_cdet_threshold' or 'eer_threshold'  
    **kwargs: valid options: c (colour='black'), s (size=75), marker (='^'), label (='threshold'). Passed to g.axes.flat.scatter()
    
    OUTPUT
    ------
    Returns FacetGrid object with baseline added
    """
    
    norm_fpfn = fpfn_min_threshold(fpfnth_baseline, metrics_baseline[threshold_type], ppf_norm=True)
    
    kwargs_c = kwargs.get('c', 'black')
    kwargs_s = kwargs.get('s', 75)
    kwargs_marker = kwargs.get('marker', '^')
    kwargs_label = kwargs.get('label', 'threshold')
    
    for ax in g.axes.flat:
        ax.plot(sp.stats.norm.ppf(fpfnth_baseline['fprs']), sp.stats.norm.ppf(fpfnth_baseline['fnrs']), c=kwargs_c, ls='dotted')
        ax.scatter(x=norm_fpfn[0], y=norm_fpfn[1], label=kwargs_label, c=kwargs_c, s=kwargs_s, marker=kwargs_marker)
    
    return g



def plot_thresholds(g, fpfnth, metrics, threshold_type, min_threshold=['subgroup','overall'], metrics_baseline=None, **kwargs): #TO DO: change to thresholds
    """
    Add subgroup and overall thresholds at minimium threshold_type to every DET curve subplot of an existing seaborn FacetGrid.
    
    ARGUMENTS
    ---------
    g [FacetGrid]: created for example with plot_det_curves()
    fpfnth [dataframe]: output of sveva.evaluate.fpfnth [all_fpfnth dataframe, all_metrics dictionary] OR a list with a dataframe   
    metrics []:
    threshold_type []:
    metrics_baseline []:
    min_threshold [list]: valid items are 'subgroup', 'overall'. Plot thresholds at subgroup and overall minimum threshold value respectively.
    **kwargs: valid options: s (size=100), marker (='x'). Passed to g.axes.flat.scatter()
    
    OUTPUT
    ------    
    Returns FacetGrid object with thresholds added
    """
    
    kwargs_s = kwargs.get('s', 100)
    kwargs_marker = kwargs.get('marker', 'x')
    
    assert isinstance(min_threshold, list), 'type min_threshold must be list'
    
    #TO DO: update to work for multi-level legend (i.e. hue & style). At the moment only works for hue & multi-column

    filter_by_1 = g.legend.get_title().get_text()
    colors = [i.get_c() for i in g.legend.get_lines()]
    if type(colors[0]) is not str:
        colors = [np.array([c]) for c in colors]
    lines = [i.get_text() for i in g.legend.get_texts()] 
    line_colors = dict(zip(lines, colors))
    
    if len(g.axes_dict.keys()) != 0:
        for col, subplot in g.axes_dict.items():
            filter_by_2 = subplot.get_title().split(' = ')[0]
            for li in lines:
                try:
                    if 'subgroup' in min_threshold:
                        metrics_key = [key for key in [key for key in metrics.keys() if li.replace(" ", "").lower() in key.lower()
                                                      ] if col.replace(" ", "").lower() in key.lower()]
                        sg_fpfn_min_cdet = fpfn_min_threshold(fpfnth[(fpfnth[filter_by_1]==li) & (fpfnth[filter_by_2]==col)],
                                                                metrics[metrics_key[0]][threshold_type], ppf_norm=True)
                        subplot.scatter(x=sg_fpfn_min_cdet[0], y=sg_fpfn_min_cdet[1], label=threshold_type+' for '+col.lower()+'_'+li, 
                                        c=line_colors[li], s=kwargs_s, marker=kwargs_marker)
                        
                    if 'overall' in min_threshold:
                        sg_fpfn_overall_min_cdet = fpfn_min_threshold(fpfnth[(fpfnth[filter_by_1]==li) & (fpfnth[filter_by_2]==col)],
                                                                 metrics_baseline[threshold_type], ppf_norm=True)
                        subplot.scatter(x=sg_fpfn_overall_min_cdet[0], y=sg_fpfn_overall_min_cdet[1], label=threshold_type+' for all', 
                                        c=line_colors[li], s=75, marker='^')
                except:
                    pass
                
    else:
        for li in lines:
            fpfn_overall_min_cdet = fpfn_min_threshold(fpfnth[fpfnth[filter_by_1]==li], metrics[li][threshold_type], ppf_norm=True)
            g.axes[0][0].scatter(x=fpfn_overall_min_cdet[0], y=fpfn_overall_min_cdet[1], label=threshold_type+' for '+li, 
                                 c=line_colors[li], s=75, marker='^')
        
    return g



def plot_score_distribution():
    
    return



def plot_cdet_ratios():
    
    return