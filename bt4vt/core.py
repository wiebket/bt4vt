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
            scores_file_name = date.strftime("%d%m%Y-%H%M%S")
        else:
            # TODO: error handling
            scores_file_name = None

        # create results directory if it does not exists
        results_dir = self.config["results_dir"]
        if not os.path.isdir(os.path.expanduser(results_dir)):
            os.makedirs(os.path.expanduser(results_dir))

        # dataset_evaluation will later be turned into a results file rather than a log file
        if self.config["dataset_evaluation"]:
            self._dataset_eval_log_file = "dataset_eval_" + config_file_name + "_" + scores_file_name + ".log"
        else:
            self._dataset_eval_log_file = None

        self._biastest_results_file = config_file_name + "_" + scores_file_name + ".csv"

    def _check_input(self, scores_input, speaker_metadata_input):
        """ Check that requirements for performing evaluation are fulfilled e.g. parameters of scores, speaker metadata and config are specified correctly

            :param scores_input: DataFrame that contains reference and test utterances and corresponding labels and scores
            :type scores_input: DataFrame
            :param speaker_metadata_input: DataFrame that contains speaker metadata with speaker ids and speaker groups attributes as specified in config file
            :type speaker_metadata_input: DataFrame

        """

        # check config file
        # check fpr_values (float) and dcf_costs (list of lists)
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
            self.config["group_names"]
        except KeyError:
            print("Error: group_names not specified in config file")
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

    def run_tests(self, write_output=True):
        """ Main method of the SpeakerBiasTest class which performs bias evaluation and tests.
        This function calls :py:func:`evaluate.evaluate_scores` from :py:mod:`evaluate.py` for the overall dataset.
        Later subgroups are constructed using :py:func:`groups.split_scores_by_speaker_groups` from :py:mod:`groups.py`.
        These subgroup scores are again evaluated using :py:func:`evaluate.evaluate_scores`.
        Lastly bias measures are computed calling :py:func:`metrics.BiasMeasures` from :py:mod:`metrics.py`.

        :returns: biastest_results_file to the results directory as specified in config.yaml, the name of the file contains the config filename and the scores filename. If a scores dataframe was provided instead of a scores filename the results file contains the date and time of the evaluation
        :rtype: csv_file

        """

        print("Running bias test on scores")

        metrics = dict()

        # Calculate average metrics
        fprs, fnrs, thresholds, metric_scores_dict, metric_thresholds_dict = evaluate_scores(self.scores['score'], self.scores['label'],
                                                                                   self.config['fpr_values'], self.config['dcf_costs'],
                                                                                   threshold_values=None)
        # add string to prepare for SpeakerGroup row
        metrics['thresholds'] = {'thresholds': metric_thresholds_dict}
        metrics['average'] = {'average': metric_scores_dict}

        # Calculate metrics for each group at the 'average' thresholds
        self.scores_by_speaker_groups = split_scores_by_speaker_groups(self.scores, self.speaker_metadata, self.config['group_names'])
        for group in self.scores_by_speaker_groups:
            metrics[group] = dict()
            for subgroup in self.scores_by_speaker_groups[group]:
                label_score_list = self.scores_by_speaker_groups[group][subgroup]
                labels, scores = zip(*label_score_list)
                if any(np.isnan(labels)) or any(np.isnan(scores)):
                    continue
                fprs, fnrs, thresholds, metric_scores = evaluate_scores(scores, labels, self.config['fpr_values'], self.config['dcf_costs'],
                                                                        threshold_values=metrics['thresholds']['thresholds'])
                metrics[group][subgroup] = metric_scores

        metrics_list = []
        for group_name in metrics.keys():
            for group_category in metrics[group_name].keys():
                m_cols = ['group_name', 'group_category'] # currently not using m_cols; should construct dict and then append to dataframe based on keys & col names
                m = [group_name, group_category]
                for k,v in metrics[group_name][group_category].items():
                    m_cols.append(k)
                    m.append(v)
                metrics_list.append(m)

        # write metrics to biastest results file
        output = pd.DataFrame(metrics_list, columns=['group_name', 'group_category']+list(metrics['thresholds']['thresholds'].keys()))
        if write_output is True:
            write_data(output, os.path.join(self.config["results_dir"], self._biastest_results_file))
            print("Bias test finished. Results saved to " + self.config["results_dir"]+self._biastest_results_file)
        self.metrics = output

        # calculate a bias measures
        # do this separately

        return

    def evaluate_dataset(self):

        # TODO: implement method
        evaluate_scores_by_speaker_groups(self.scores_by_speaker_groups, self._dataset_eval_log_file)

        return
