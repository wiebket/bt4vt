#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 01-05-2022
# @author: wiebket, AnnaLesch

import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from .dataio import load_config, load_data, write_data
from .evaluate import evaluate_scores
from .groups import split_scores_by_speaker_groups
from .metrics import compute_metrics_ratios
from .dataset_evaluate import evaluate_scores_by_speaker_groups
#from .plot import plot_det_curves


class BiasTest:

    def __init__(self):

        return self

    def run_tests(self):

        return

    def plot(self):

        # TODO: implement method

        return

    def evaluate_dataset(self):

        # TODO: implement method

        return


class SpeakerBiasTest(BiasTest):
    """ The primary purpose of the SpeakerBiasTest class is the implementation of the run_tests() method, which performs the bias tests.

        :param scores: Either path to csv or txt file or a Pandas DataFrame that includes information on the reference and test utterances as well as corresponding labels and scores
        :type scores: str or DataFrame
        :param config_file: path to yaml config file
        :type config_file: str

    """

    def __init__(self, scores,
                 config_file):
        """Constructor method
        """
        self.error_rates_by_speaker_group = dict()
        self.metrics = pd.DataFrame()

        self.config = load_config(config_file)
        scores_input = load_data(scores)
        speaker_metadata_input = load_data(self.config['speaker_metadata_file'])

        self._check_input(scores_input, speaker_metadata_input)

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

        # check if results directory exists
        results_dir = self.config["results_dir"]
        if not os.path.isdir(os.path.expanduser(results_dir)):
            os.mkdir(os.path.expanduser(results_dir))

        # dataset_evaluation will later be turned into a results file rather than a log file
        if self.config["dataset_evaluation"]:
            self.__dataset_eval_log_file = "dataset_eval_" + config_file_name + "_" + scores_file_name + ".log"
        else:
            self.__dataset_eval_log_file = None

        self.__biastest_results_file = "biastest_results_" + config_file_name + "_" + scores_file_name + ".csv"

    def _check_input(self, scores_input, speaker_metadata_input):
        """ Check that requirements for performing evaluation are fulfilled e.g. parameters of scores, speaker metadata and config are specified correctly

            :param scores_input: DataFrame that contains reference and test utterances and corresponding labels and scores
            :type scores_input: DataFrame
            :param speaker_metadata_input: DataFrame that contains speaker metadata with speaker ids and speaker groups attributes as specified in config file
            :type speaker_metadata_input: DataFrame

        """

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

    def run_tests(self):
        """ Main method of the SpeakerBiasTest class which performs bias evaluation and tests.
        This function calls :py:func:`evaluate.evaluate_scores` from :py:mod:`evaluate.py` for the overall dataset.
        Later subgroups are constructed using :py:func:`groups.split_scores_by_speaker_groups` from :py:mod:`groups.py`.
        These subgroup scores are again evaluated using :py:func:`evaluate.evaluate_scores`.
        Lastly metric ratios are computed calling :py:func:`metrics.compute_metrics_ratios` from :py:mod:`metrics.py`.

        :returns: biastest_results_file to the results directory as specified in config.yaml, the name of the file contains the config filename and the scores filename. If a scores dataframe was provided instead of a scores filename the results file contains the date and time of the evaluation
        :rtype: csv_file

        """

        print("Running bias test on scores")

        # Calculate overall metrics
        fprs, fnrs, thresholds, metric_scores, metric_thresholds = evaluate_scores(self.scores['score'], self.scores['label'], self.config['dcf_costs'])
        self.error_rates_by_speaker_group.update({"overall": pd.DataFrame({'FPRS': fprs, 'FNRS': fnrs, 'Thresholds': thresholds})})
        # add string to prepare for SpeakerGroup row
        self.metrics['thresholds'] = ["thresholds"] + metric_thresholds
        self.metrics['overall'] = ["overall"] + metric_scores

        # for metrics first row is eer, after that follow order of self.config.dcf_costs

        # Calculate metrics for each group
        self.scores_by_speaker_groups = split_scores_by_speaker_groups(self.scores, self.speaker_metadata, self.config['speaker_groups'])
        for group in self.scores_by_speaker_groups:
            for subgroup in self.scores_by_speaker_groups[group]:
                label_score_list = self.scores_by_speaker_groups[group][subgroup]
                labels, scores = zip(*label_score_list)
                if any(np.isnan(labels)) or any(np.isnan(scores)):
                    continue

                fprs, fnrs, thresholds, metric_scores = evaluate_scores(scores, labels, self.config['dcf_costs'], threshold_values=self.metrics['thresholds'])

                # TODO: Wiebke to check on different dimensions for subgroups for fprs, fnrs, thresholds
                # if group in keys add to existing DataFrame otherwise create new key
                if group in self.error_rates_by_speaker_group.keys():
                    self.error_rates_by_speaker_group[group] = pd.concat([self.error_rates_by_speaker_group[group], pd.DataFrame({'Subgroup': subgroup, 'FPRS': fprs, 'FNRS': fnrs, 'Thresholds': thresholds})])
                else:
                    self.error_rates_by_speaker_group.update({group: pd.DataFrame({'Subgroup': subgroup, 'FPRS': fprs, 'FNRS': fnrs, 'Thresholds': thresholds})})

                # for metrics first row is eer, after that follow order of self.config.dcf_costs
                self.metrics[subgroup] = [group] + metric_scores

        # do bias test and bring metrics ratios in long format
        metrics_ratios = compute_metrics_ratios(self.metrics)
        metrics_ratios = metrics_ratios.transpose()
        metrics_ratios.index.name = 'Subgroup'
        metrics_ratios.columns = ["Speaker_Group", "EER"] + ["DCF" + str(cost) for cost in self.config["dcf_costs"]]
        metrics_ratios.reset_index(inplace=True)

        # write metrics ratios to biastest results file
        write_data(metrics_ratios, self.config["results_dir"] + self.__biastest_results_file)

        # calculate a bias test score: function in metrics which takes output of compute_metrics_ratios

        print("Bias test finished. Results saved to " + self.__biastest_results_file)

        return

    def plot(self):
        """ Plotting of bias test results
        """

        #plot_det_curves(self.fprs, self.fnrs, subgroup="f_India")

        return

    def evaluate_dataset(self):

        # TODO: implement method
        evaluate_scores_by_speaker_groups(self.scores_by_speaker_groupsm, self.__dataset_eval_log_file)

        return