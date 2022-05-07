#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 01-05-2022
# @author: wiebket, AnnaLesch

import pandas as pd
from datetime import datetime
from pathlib import Path
from .dataio import load_config, load_data
from .evaluate import evaluate_scores
from .groups import split_scores_by_speaker_groups
from .metrics import compute_metrics_ratios

class BiasTest:

    def __init__(self):

        return self

    def audit(self):

        return

    def plot(self):

        # TODO: implement method

        return

    def evaluate_dataset(self):

        # TODO: implement method

        return


class SpeakerBiasTest(BiasTest):

    def __init__(self, scores,
                 config_file):

        self.fprs = pd.DataFrame()
        self.fnrs = pd.DataFrame()
        self.thresholds = pd.DataFrame()
        self.metrics = pd.DataFrame()

        self.config = load_config(config_file)
        scores_input = load_data(scores)
        speaker_metadata_input = load_data(self.config['speaker_metadata_file'])

        self.check_input(scores_input, speaker_metadata_input)

        # columns selection, reordering and renaming
        self.scores = scores_input
        # TODO: select, reorder, rename score file: "label", "ref", "test", "score"

        self.speaker_metadata = speaker_metadata_input
        # TODO: select and reorder metadata_file: rename to "id" first,

        config_file_name = Path(config_file).stem
        if isinstance(scores, str):
            scores_file_name = Path(scores).stem
        elif isinstance(scores, pd.DataFrame):
            date = datetime.now()
            scores_file_name = date.strftime("%d_%m_%Y_%H_%M_%S")
        else:
            # TODO: error handling
            scores_file_name = None
        self.biastest_results_file = config_file_name + "_" + scores_file_name


    def check_input(self, scores_input, speaker_metadata_input):
        # TODO: config_file
        # TODO: check id_column exsists, select columns exist and speaker groups exist
        # TODO: check that speaker groups are part of select columns

        # TODO: score_input: check that all columns are in data frame as specified in config

        # TODO: speaker_metadata_input: check that ID and select columns are in dataframe as specified in config

        return

    def audit(self): #note: check if weights is the right term to use, or if cost -- look at function

        # Calculate overall metrics
        fprs, fnrs, thresholds, metric_scores, metric_thresholds = evaluate_scores(self.scores['score'], self.scores['label'], self.config['dcf_costs'])
        self.fprs['overall'] = fprs
        self.fnrs['overall'] = fnrs
        self.thresholds['overall'] = thresholds
        self.metrics['thresholds'] = metric_thresholds
        self.metrics['overall'] = metric_scores

        # for metrics first row is eer, after that follow order of self.config.dcf_costs

        # Calculate metrics for each group
        scores_by_speaker_groups = split_scores_by_speaker_groups(self.scores, self.speaker_metadata, self.config['speaker_groups'])
        # maybe create dictionary with groups and categories
        for group in scores_by_speaker_groups:
            for category in group:
                label = category[0]
                score = category[1]
                fprs, fnrs, thresholds, metric_scores = evaluate_scores(score, label, self.config['dcf_costs'], threshold_values=self.metrics['thresholds'])
                self.fprs[category] = fprs
                self.fnrs[category] = fnrs
                self.thresholds[category] = thresholds
                self.metrics[category] = metric_scores

                # for metrics first row is eer, after that follow order of self.config.dcf_costs

        # do bias test
        metrics_ratios = compute_metrics_ratios(self.metrics)

        return

    def plot(self):

        # TODO: implement method

        return

    def evaluate_dataset(self):

        # TODO: implement method

        return