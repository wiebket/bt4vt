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
from .evaluate import mincdet_fnfp


def plot_det_curves(fnfpth_list):

    df = fnfpth_list[0]

    sns.set_context(context='notebook', font_scale=1)
    sns.set_style('ticks')

    g = sns.relplot(x=sp.stats.norm.ppf(df['fprs']), 
                    y=sp.stats.norm.ppf(df['fnrs']), 
                    hue='ref_gender',
                    style='ref_gender',
                    col='ref_nationality',
                    col_wrap=3,
                    height=4, aspect=1.3, linewidth=2.5,
                    kind='line', 
                    data=df)

    ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
    tick_locations = sp.stats.norm.ppf(ticks)
    tick_labels = [
      '{:.0%}'.format(s) if (100*s).is_integer() else '{:.1%}'.format(s)
      for s in ticks
    ]

    p = mincdet_fnfp(fnfpth_list)

    for ax in g.axes.flat:
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(-3.5, 1)
        ax.set_yticks(tick_locations)
        ax.set_yticklabels(tick_labels)
        ax.set_ylim(-3.5, 0.5)
        col = ax.get_title().split(' = ')[1].lower()
        print(col)
        h, lines = ax.get_legend_handles_labels()
        for l in lines:
            sg = col+'_'+l #TO DO: Looks like lines only returns an object once
            print(sg)
            try:
                ax.scatter(x=p[sg]['all'][1], y=p[sg]['all'][0], c='black', s=75, marker='^')
                ax.scatter(x=p[sg]['subgroup'][1], y=p[sg]['subgroup'][0], c='black', s=100, marker='*')
            except:
                pass
        ax.scatter(x=p['all']['all'][1], y=p['all']['all'][0], label='system threshold', c='black', s=75, marker='^')
        ax.plot(sp.stats.norm.ppf(df[df['subgroup']=='all']['fprs']), 
                sp.stats.norm.ppf(df[df['subgroup']=='all']['fnrs']), 
                c='k', ls='dotted')

    g.set_xlabels('false positive rate')
    g.set_ylabels('false negative rate')

    #legend = plt.legend(loc='upper right')

    return g