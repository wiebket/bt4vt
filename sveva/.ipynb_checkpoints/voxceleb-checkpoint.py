#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 18-05-2021
@author: wiebket
"""

import pandas as pd
import numpy as np
import scipy as sp


def voxceleb_scores_with_demographics(score_file, meta_file, **kwargs):
    """
    score_file [str]: path to file with scores
    meta_file [str]: path to file with demographic metadata
    **kwargs: passed to pd.read_csv() 
    """

    # Get the scores for each pair of segments
    df = pd.read_csv(score_file)

    df['ref_id'] = df['ref_file'].apply(lambda x: x.split('/')[0])
    df['ref_video'] = df['ref_file'].apply(lambda x: x.split('/')[1])
    df['ref_seg'] = df['ref_file'].apply(lambda x: x.split('/')[2].split('.')[0])
    df['com_id'] = df['com_file'].apply(lambda x: x.split('/')[0])
    df['com_video'] = df['com_file'].apply(lambda x: x.split('/')[1])
    df['com_seg'] = df['com_file'].apply(lambda x: x.split('/')[2].split('.')[0])

    # Get demographic metadata
    v1_meta = pd.read_csv(meta_file, **kwargs)

    # Merge the scores with demographic metadata
    demo_ref = pd.merge(
      left=df, 
      right=v1_meta, 
      left_on='ref_id', 
      right_on='VoxCeleb1 ID', 
      how='left', 
      sort=False
      ).drop(labels='VoxCeleb1 ID', axis=1)

    demo_ref.rename({
      'VGGFace1 ID':'ref_vggface1', 
      'Gender':'ref_gender',
      'Nationality':'ref_nationality',
      'Set':'ref_set'}, 
      axis=1, inplace=True)

    demo_df = pd.merge(
      left=demo_ref, 
      right=v1_meta, 
      left_on='com_id', 
      right_on='VoxCeleb1 ID', 
      how='left', 
      sort=False
      ).drop(labels='VoxCeleb1 ID', axis=1)

    demo_df.rename({
      'VGGFace1 ID':'com_vggface1', 
      'Gender':'com_gender',
      'Nationality':'com_nationality',
      'Set':'com_set'}, 
      axis=1, inplace=True)

    demo_df['same_gen'] = np.where(demo_df['ref_gender']==demo_df['com_gender'], 1, 0)
    demo_df['same_nat'] = np.where(demo_df['ref_nationality']==demo_df['com_nationality'], 1, 0)

    return demo_df