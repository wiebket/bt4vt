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



def _norm_fnfp_min_threshold(fnfpth, threshold):
    """
    Calculate the normalised false positive rate (FPR) and false negative rate (FNR) at the minimum threshold value.
    
    ARGUMENTS
    ---------
    fnfpth [dataframe]: dataframe, must contain false negative rates ['fnrs'], false positive rates ['fprs'] and threshold values ['thresholds']
    threshold [float]: score at threshold 'min_cdet_threshold' or 'eer_threshold'
    
    OUTPUT
    ------
    list: [FPR, FNR] at minimum threshold value
    """

    # Find the index in fnfpth that is closest to the SUBGROUP minimum threshold value
    sg_threshold_diff = np.array([abs(i - threshold) for i in fnfpth['thresholds']])
    min_threshold_fnr = sp.stats.norm.ppf(fnfpth['fnrs'])[np.ndarray.argmin(sg_threshold_diff)]
    min_threshold_fpr = sp.stats.norm.ppf(fnfpth['fprs'])[np.ndarray.argmin(sg_threshold_diff)]

    norm_threshold = [min_threshold_fpr, min_threshold_fnr]

    return norm_threshold



def plot_det_curves(fnfpth, **kwargs):
    """
    
    ARGUMENTS
    ---------
    fnfpth [dataframe]: dataframe, must contain false negative rates ['fnrs'] and false positive rates ['fprs']. 
                        If **kwargs are specified, then dataframe must contain columns with names used for values in hue, style and col (see below)
    **kwargs: valid options: hue, style, col and palette. Passed to seaborn.relplot()
    """

    kwargs_hue = kwargs.get('hue', None)
    kwargs_style = kwargs.get('style', None)
    kwargs_col = kwargs.get('col', None)
    kwargs_palette= kwargs.get('palette', 'colorblind')
    try:
        n_kwargs_col = len(fnfpth[kwargs_col].unique())
        n_col_wrap = 3 if n_kwargs_col >=3 else n_kwargs_col
    except KeyError:
        n_col_wrap = None

    sns.set_context(context='notebook', font_scale=1)
    sns.set_style('ticks')

    g = sns.relplot(x=sp.stats.norm.ppf(fnfpth['fprs']), 
                    y=sp.stats.norm.ppf(fnfpth['fnrs']), 
                    hue=kwargs_hue,
                    style=kwargs_style,
                    col=kwargs_col,
                    col_wrap=n_col_wrap,
                    palette=kwargs_palette,
                    height=4, aspect=1.3, linewidth=2.5,
                    kind='line', 
                    facet_kws=dict(sharex=False,sharey=False),
                    data=fnfpth)

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

    g.set_xlabels('false positive rate')
    g.set_ylabels('false negative rate')

    return g



def plot_det_baseline(g, fnfpth_baseline, metrics_baseline, threshold_type, **kwargs):
    """
    Add a baseline DET curve to every DET curve subplot of an existing seaborn FacetGrid.
    
    ARGUMENTS
    ---------
    g [FacetGrid]: created for example with plot_det_curves()
    fnfpth_baseline [dataframe]: dataframe, must contain false negative rates ['fnrs'], false positive rates ['fprs'] and threshold values ['thresholds']
    threshold_value [float]: score at threshold 'min_cdet_threshold' or 'eer_threshold'  
    **kwargs: valid options: c (colour='black'), s (size=75), marker (='^'), label (='threshold'). Passed to g.axes.flat.scatter()
    
    OUTPUT
    ------
    FacetGrid object with baseline added
    """
    
    norm_fnfp = _norm_fnfp_min_threshold(fnfpth_baseline, metrics_baseline[threshold_type])
    
    kwargs_c = kwargs.get('c', 'black')
    kwargs_s = kwargs.get('s', 75)
    kwargs_marker = kwargs.get('marker', '^')
    kwargs_label = kwargs.get('label', 'threshold')
    
    for ax in g.axes.flat:
        ax.plot(sp.stats.norm.ppf(fnfpth_baseline['fprs']), sp.stats.norm.ppf(fnfpth_baseline['fnrs']), c=kwargs_c, ls='dotted')
        ax.scatter(x=norm_fnfp[0], y=norm_fnfp[1], label=kwargs_label, c=kwargs_c, s=kwargs_s, marker=kwargs_marker)
    
    return g



def plot_thresholds(g, fnfpth, metrics, threshold_type, metrics_baseline=None, **kwargs): #TO DO: change to thresholds
    """
    Add system and subgroup thresholds to every DET curve subplot of an existing seaborn FacetGrid.
    
    ARGUMENTS
    ---------
    g [FacetGrid]: created for example with plot_det_curves()
    fnfpth_list [list]: output of sveva.evaluate.fnfpth [all_fnfpth dataframe, all_metrics dictionary] OR a list with a dataframe   
    **kwargs: valid options: s (size=75), marker (='^'). Passed to g.axes.flat.scatter()
    """
    
    kwargs_s = kwargs.get('s', 100)
    kwargs_marker = kwargs.get('marker', 'x')
    
    #TO DO: update to work for multi-level legend (i.e. hue & style). At the moment only works for hue & multi-column

    filter_by_1 = g.legend.get_title().get_text()
    colors = [list(i.get_c()) for i in g.legend.get_lines()]
    lines = [i.get_text() for i in g.legend.get_texts()] 
    line_colors = dict(zip(lines, colors))
    
    if len(g.axes_dict.keys()) != 0:
        for col, subplot in g.axes_dict.items():
            filter_by_2 = subplot.get_title().split(' = ')[0]
            for li in lines:
                try:
                    sg_norm_fnfp = _norm_fnfp_min_threshold(fnfpth[(fnfpth[filter_by_1]==li) & (fnfpth[filter_by_2]==col)],
                                                            metrics[col.replace(" ", "").lower()+'_'+li][threshold_type])
                    subplot.scatter(x=sg_norm_fnfp[0], y=sg_norm_fnfp[1], label=threshold_type+' for '+col.lower()+'_'+li, 
                                    c=np.array([line_colors[li]]), s=kwargs_s, marker=kwargs_marker)

                    all_norm_fnfp = _norm_fnfp_min_threshold(fnfpth[(fnfpth[filter_by_1]==li) & (fnfpth[filter_by_2]==col)],
                                                             metrics_baseline[threshold_type])
                    subplot.scatter(x=all_norm_fnfp[0], y=all_norm_fnfp[1], label=threshold_type+' for all', 
                                    c=np.array([line_colors[li]]), s=75, marker='^')
                except:
                    pass
                
    else:
        for li in lines:
            norm_fnfp = _norm_fnfp_min_threshold(fnfpth[fnfpth[filter_by_1]==li], metrics[li][threshold_type])
            g.axes[0][0].scatter(x=norm_fnfp[0], y=norm_fnfp[1], label=threshold_type+' for '+li, 
                                 c=np.array([line_colors[li]]), s=75, marker='^')
        
    return g