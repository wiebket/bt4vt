#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 01-05-2022
# @author: wiebket, AnnaLesch

import pandas as pd
import numpy as np
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
        speaker_metadata_input = load_data(self.config['speaker_metadata_file'], delim_whitespace=True)

        self.check_input(scores_input, speaker_metadata_input)

        # scores_input columns selection, reordering and renaming
        scores_input = scores_input[[self.config["label_column"],
                                     self.config["reference_filepath_column"],
                                     self.config["test_filepath_column"],
                                     self.config["scores_column"]]]
        self.scores = scores_input.rename(columns={self.config["label_column"]: "label",
                                                   self.config["reference_filepath_column"]: "ref",
                                                   self.config["test_filepath_column"]: "test",
                                                   self.config["scores_column"]: "score"})
        # speaker_metadata_input column selection, reordering, renaming id
        metadata_selection_list = self.config["select_columns"]
        metadata_selection_list.insert(0, self.config["id_column"])
        speaker_metadata_input = speaker_metadata_input[metadata_selection_list]
        self.speaker_metadata = speaker_metadata_input.rename(columns={self.config["id_column"]: "id"})

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

        # TODO error handling, at the moment only simple print messages
        # check config file
        try:
            self.config["id_column"]
        except KeyError:
            print("id_column is missing in config file")

        try:
            self.config["select_columns"]
        except KeyError:
            print("select_columns is missing in config file")

        try:
            self.config["speaker_groups"]
        except KeyError:
            print("speaker_groups is missing in config file")

        speaker_group_list = [speaker_group for group_sublist in self.config["speaker_groups"] for speaker_group in group_sublist]
        speaker_group_list = np.unique(speaker_group_list)
        for speaker_group in speaker_group_list:
            if speaker_group not in self.config["select_columns"]:
                print(speaker_group + " not in select_columns")

        # check scores_input
        if self.config["reference_filepath_column"] not in scores_input.columns:
            print("reference file name as specified in config file was not found in scores")
        if self.config["test_filepath_column"] not in scores_input.columns:
            print("test file name as specified in config file was not found in scores")
        if self.config["label_column"] not in scores_input.columns:
            print("label as specified in config file was not found in scores")
        if self.config["scores_column"] not in scores_input.columns:
            print("scores as specified in config file were not found in scores")

        # check metadata_input
        if self.config["id_column"] not in speaker_metadata_input.columns:
            print("id_column as specified in config file was not found in metadata file")

        for select_column in self.config["select_columns"]:
            if select_column not in speaker_metadata_input.columns:
                print(select_column + " as specified in config file was not found in metadata file")

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