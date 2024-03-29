#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on 01-05-2022
# @author: wiebket, AnnaLesch

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path
from .dataio import load_config, load_data, write_data
from .evaluate import evaluate_scores
from .groups import split_scores_by_speaker_groups
from .metrics import compute_metrics_ratios
from .dataset_evaluate import evaluate_scores_by_speaker_groups


class BiasTest:

    """ Elementary Class for implementing Bias tests.
    """

    def __init__(self):

        """
        Constructor method
        """

        return self

    def run_tests(self):

        """
        Runs bias tests. This is an empty method that needs to be implemented by subclasses
        """

        return

    def plot(self):

        """
        This is an empty method that can be implemented by subclasses.
        """

        return

    def evaluate_dataset(self):

        """
        This is an empty method that can be implemented by subclasses.
        """

        return


class SpeakerBiasTest(BiasTest):
    """ The primary purpose of the SpeakerBiasTest class is the implementation of the run_tests() method, which performs
     the bias tests.

        :param scores: Either path to csv or txt file or a Pandas DataFrame that includes information on the reference
        and test utterances as well as corresponding labels and scores; labels have to be either {-1,1} or {0,1}
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
        try:
            self.config["id_delimiter"]
        except KeyError:
            self.id_delimiter = "/"
        else:
            self.id_delimiter = self.config["id_delimiter"]

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
        self.scores = self.scores.astype({"ref": "str", "test": "str"})
        # speaker_metadata_input column selection, reordering, renaming id
        metadata_selection_list = self.config["select_columns"]
        metadata_selection_list.insert(0, self.config["id_column"])
        speaker_metadata_input = speaker_metadata_input[metadata_selection_list]
        # delete metadata rows that include NaN, None or Empty Strings in selected columns
        speaker_metadata_input.replace(' ', np.nan, inplace=True)
        if speaker_metadata_input.isnull().values.any():
            print("Selected Columns in Metadata File contain NaNs or Empty Cells. We recommend a dataset evaluation.")
        speaker_metadata_input.dropna(inplace=True)

        self.speaker_metadata = speaker_metadata_input.rename(columns={self.config["id_column"]: "id"})
        self.speaker_metadata = self.speaker_metadata.astype({"id": "str"})

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
            os.makedirs(os.path.expanduser(results_dir))

        # dataset_evaluation will later be turned into a results file rather than a log file
        if self.config["dataset_evaluation"]:
            self._dataset_eval_log_file = "dataset_eval_" + config_file_name + "_" + scores_file_name + ".log"
        else:
            self._dataset_eval_log_file = None

        self._biastest_results_file = "biastest_results_" + config_file_name + "_" + scores_file_name + ".csv"

    def _check_input(self, scores_input, speaker_metadata_input):
        """ Check that requirements for performing evaluation are fulfilled e.g. parameters of scores, speaker metadata and config are specified correctly

            :param scores_input: DataFrame that contains reference and test utterances and corresponding labels and scores
            :type scores_input: DataFrame
            :param speaker_metadata_input: DataFrame that contains speaker metadata with speaker ids and speaker groups attributes as specified in config file
            :type speaker_metadata_input: DataFrame

        """

        # check config file
        try:
            self.config["id_column"]
        except KeyError:
            print("Error: id_column not specified in config file")
            sys.exit(1)

        try:
            self.config["select_columns"]
        except KeyError:
            print("Error: select_columns not specified in config file")
            sys.exit(1)

        try:
            self.config["speaker_groups"]
        except KeyError:
            print("Error: speaker_groups not specified in config file")
            sys.exit(1)

        if not isinstance(self.config["select_columns"], list):
            raise ValueError("Select Columns in config file must be a list")

        if not all(isinstance(el, list) for el in self.config["speaker_groups"]):
            raise ValueError("Speaker Groups in config file must be a list of lists")

        speaker_group_list = [speaker_group for group_sublist in self.config["speaker_groups"] for speaker_group in group_sublist]
        speaker_group_list = np.unique(speaker_group_list)
        for speaker_group in speaker_group_list:
            try:
                self.config["select_columns"].index(speaker_group)
            except ValueError:
                print("Error: " + speaker_group + " not found in select_columns as specified in config file")
                sys.exit(1)

        # check if dcf costs PTarget is between 0 and 1
        for dcf_costs in self.config["dcf_costs"]:
            if (dcf_costs[0] <= 0.0) | (dcf_costs[0] >= 1.0):
                raise Exception("PTarget in DCF Costs needs to be between 0 and 1")

        # check scores_input
        try:
            list(scores_input.columns).index(self.config["reference_filepath_column"])
        except ValueError:
            print("Error: reference filepath column '" + self.config["reference_filepath_column"] + "' as specified in config file not found in scores file")
            sys.exit(1)

        try:
            list(scores_input.columns).index(self.config["test_filepath_column"])
        except ValueError:
            print("Error: test filepath column '" + self.config["test_filepath_column"] + "' as specified in config file not found in scores file")
            sys.exit(1)

        try:
            list(scores_input.columns).index(self.config["label_column"])
        except ValueError:
            print("Error: label column '" + self.config["label_column"] + "' as specified in config file not found in scores file")
            sys.exit(1)

        try:
            list(scores_input.columns).index(self.config["scores_column"])
        except ValueError:
            print("Error: scores column '" + self.config["scores_column"] + "' as specified in config file not found in scores file")
            sys.exit(1)

        # check metadata_input
        try:
            list(speaker_metadata_input.columns).index(self.config["id_column"])
        except ValueError:
            print("Error: id column '" + self.config["id_column"] + "' as specified in config file not found in metadata file")
            sys.exit(1)

        for select_column in self.config["select_columns"]:
            try:
                list(speaker_metadata_input.columns).index(select_column)
            except ValueError:
                print("Error: '" + select_column + "' in select_columns as specified in config file not found in metadata file")
                sys.exit(1)

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

        # Calculate average metrics
        fprs, fnrs, thresholds, metric_scores, metric_thresholds = evaluate_scores(self.scores['score'], self.scores['label'], self.config['dcf_costs'])
        self.error_rates_by_speaker_group.update({"average": pd.DataFrame({'FPRS': fprs, 'FNRS': fnrs, 'Thresholds': thresholds})})
        # add string to prepare for SpeakerGroup row
        self.metrics['thresholds'] = ["thresholds"] + metric_thresholds
        self.metrics['average'] = ["average"] + metric_scores

        # for metrics first row is EER, after that follow order of self.config.dcf_costs

        # Calculate metrics for each group
        self.scores_by_speaker_groups = split_scores_by_speaker_groups(self.scores, self.speaker_metadata, self.config['speaker_groups'], id_delimiter=self.id_delimiter)
        for group in self.scores_by_speaker_groups:
            for subgroup in self.scores_by_speaker_groups[group]:
                label_score_list = self.scores_by_speaker_groups[group][subgroup]
                labels, scores = zip(*label_score_list)
                if all(np.isnan(labels)) or all(np.isnan(scores)):
                    fprs = []
                    fnrs = []
                    thresholds = []
                    metric_scores = np.empty((len(self.config["dcf_costs"]) + 1))
                    metric_scores[:] = np.nan
                    metric_scores = metric_scores.tolist()
                else:
                    fprs, fnrs, thresholds, metric_scores = evaluate_scores(scores, labels, self.config['dcf_costs'], threshold_values=self.metrics['thresholds'])

                # if group in keys add to existing DataFrame otherwise create new key
                if group in self.error_rates_by_speaker_group.keys():
                    self.error_rates_by_speaker_group[group] = pd.concat([self.error_rates_by_speaker_group[group], pd.DataFrame({'Subgroup': subgroup, 'FPRS': fprs, 'FNRS': fnrs, 'Thresholds': thresholds})])
                else:
                    self.error_rates_by_speaker_group.update({group: pd.DataFrame({'Subgroup': subgroup, 'FPRS': fprs, 'FNRS': fnrs, 'Thresholds': thresholds})})

                # for metrics first row is eer, after that follow order of self.config.dcf_costs
                #self.metrics[subgroup] = [group] + metric_scores -> use concat to avoid performance issues
                self.metrics = pd.concat([self.metrics, pd.Series([group] + metric_scores).rename(subgroup)], axis=1)


        # format metrics and metrics ratios
        metrics_ratios = compute_metrics_ratios(self.metrics).T
        metrics_ratios.columns = ["speaker_groups", "EER ratio"] + ["DCF ratio " + str(cost) for cost in self.config["dcf_costs"]]

        metrics_out = self.metrics.T
        metrics_out.columns = ["speaker_groups", "EER"] + ["DCF " + str(cost) for cost in self.config["dcf_costs"]]
        output = metrics_out.rename_axis('group_name').reset_index().merge(metrics_ratios.rename_axis('group_name').reset_index())

        # write metrics and metrics ratios to biastest results file
        write_data(output, os.path.join(self.config["results_dir"], self._biastest_results_file))

        # calculate a bias test score: function in metrics which takes output of compute_metrics_ratios

        print("Bias test finished. Results saved to " + self.config["results_dir"]+self._biastest_results_file)

        return

    def evaluate_dataset(self):

        # TODO: implement method
        evaluate_scores_by_speaker_groups(self.scores_by_speaker_groups, self._dataset_eval_log_file)

        return
