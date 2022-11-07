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
        self.metrics = dict()

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
        self.scores = self.scores.astype({"ref":"str","test":"str"})
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

        # TODO error handling, at the moment only simple print messages
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

        speaker_group_list = [speaker_group for group_sublist in self.config["speaker_groups"] for speaker_group in group_sublist]
        speaker_group_list = np.unique(speaker_group_list)
        for speaker_group in speaker_group_list:
            try:
                self.config["select_columns"].index(speaker_group)
            except ValueError:
                print("Error: " + speaker_group + " not found in select_columns as specified in config file")
                sys.exit(1)

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
        fprs, fnrs, thresholds, metric_scores_dict, metric_thresholds_dict = evaluate_scores(self.scores['score'], self.scores['label'], 
                                                                                   self.config['fpr_values'], self.config['dcf_costs'],  
                                                                                   threshold_values=None)
        self.error_rates_by_speaker_group.update({"average": pd.DataFrame({'FPRS': fprs, 'FNRS': fnrs, 'Thresholds': thresholds})})
        # add string to prepare for SpeakerGroup row
        self.metrics['thresholds'] = {'thresholds': metric_thresholds_dict}
        self.metrics['average'] = {'average': metric_scores_dict}

        # Calculate metrics for each group at the 'average' thresholds
        self.scores_by_speaker_groups = split_scores_by_speaker_groups(self.scores, self.speaker_metadata, self.config['speaker_groups'])
        for group in self.scores_by_speaker_groups:
            self.metrics[group] = dict()
            for subgroup in self.scores_by_speaker_groups[group]:
                label_score_list = self.scores_by_speaker_groups[group][subgroup]
                labels, scores = zip(*label_score_list)
                if any(np.isnan(labels)) or any(np.isnan(scores)):
                    continue
                
                fprs, fnrs, thresholds, metric_scores = evaluate_scores(scores, labels, self.config['fpr_values'], self.config['dcf_costs'], 
                                                                        threshold_values=self.metrics['thresholds']['thresholds'])

                # if group in keys add to existing DataFrame, otherwise create new key
                if group in self.error_rates_by_speaker_group.keys():
                    self.error_rates_by_speaker_group[group] = pd.concat([self.error_rates_by_speaker_group[group], 
                                                                          pd.DataFrame({'Subgroup': subgroup, 'FPRS': fprs, 'FNRS': fnrs, 'Thresholds': thresholds})])
                else:
                    self.error_rates_by_speaker_group.update({group: pd.DataFrame({'Subgroup': subgroup, 'FPRS': fprs, 'FNRS': fnrs, 'Thresholds': thresholds})})

                # for metrics first row is eer, after that follow order of self.config.dcf_costs
                self.metrics[group][subgroup] = metric_scores

        # format metrics and metrics ratios
        # metrics_ratios = compute_metrics_ratios(self.metrics).T # --> remove and do this separately
        # metrics_ratios.columns = ["speaker_groups", "EER ratio"] + ["minCDet ratio " + str(cost) for cost in self.config["dcf_costs"]]

        metrics_out = self.metrics.T
        metrics_out.columns = ["speaker_groups", "EER"] + ["minCDet" + str(cost) for cost in self.config["dcf_costs"]]
        output = metrics_out.rename_axis('group_name').reset_index()#.merge(metrics_ratios.rename_axis('group_name').reset_index())

        # write metrics and metrics ratios to biastest results file
        write_data(output, os.path.join(self.config["results_dir"], self._biastest_results_file))

        # calculate a bias test score: function in metrics which takes output of compute_metrics_ratios
        # do this separately
        print("Bias test finished. Results saved to " + self.config["results_dir"]+self._biastest_results_file)

        return

    def plot(self):
        """ Plotting of bias test results
        """

        #plot_det_curves(self.fprs, self.fnrs, subgroup="f_India")

        return

    def evaluate_dataset(self):

        # TODO: implement method
        evaluate_scores_by_speaker_groups(self.scores_by_speaker_groups, self._dataset_eval_log_file)

        return
