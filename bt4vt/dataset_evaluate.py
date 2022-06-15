import logging
import numpy as np


def evaluate_scores_by_speaker_groups(scores_by_speaker_groups, log_file):

    for group in scores_by_speaker_groups:
        for category in scores_by_speaker_groups[group]:
            label_score_list = scores_by_speaker_groups[group][category]
            labels, scores = zip(*label_score_list)
            if any(np.isnan(labels)) or any(np.isnan(scores)):
                logging.basicConfig(filename=log_file, level=logging.INFO)
                logging.info("No scores available either because no ids in speaker metadata for subgroup or subgroup exists in metadata but no scores are given" + str(combination))