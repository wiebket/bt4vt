=======
Example
=======

Below is an example for using ``bt4vt``. All code and data for reproducing the example are contained in the ``bias_tests_4_voice_tech`` folder in your home directory. The example evaluates the fairness of models released with the `VoxCeleb Trainer <https://github.com/clovaai/voxceleb_trainer>`_ benchmark.

Here's the folder structure containing an example project:

| bias_tests_4_voice_tech
| ├── example
| │   ├── config.yaml
| │   ├── resnetse34l_H-eval_scores.csv
| │   ├── resnetse34v2_H-eval_scores.csv
| │   ├── vox1_meta.csv
| │   └── voxceleb1h_test_list.txt
| ├── results
| │   └── biastest_results_config_resnetse34v2_H-eval_scores.csv
|
|

In the folder structure above:

- ``bias_tests_4_voice_tech`` is the folder we get when the package is installed.
- ``bias_tests_4_voice_tech/example`` is the directory where all example files can be found.
- ``bias_tests_4_voice_tech/example/config.yaml`` is an example config file that is needed to run bias tests. More information on how to create a config file can be found below.
- ``bias_tests_4_voice_tech/example/resnetse34v2_H-eval_scores.csv`` and ``bias_tests_4_voice_tech/example/resnetse34l_H-eval_scores.csv`` are the evaluation scores for two pretrained baseline models made available by the VoxCeleb Trainer. You can find more information on ResNetSE34V2 `here <https://arxiv.org/abs/2009.14153>`_ and on ResNetSE34L `here <https://doi.org/10.21437/Interspeech.2020-1064>`_.
- ``bias_tests_4_voice_tech/example/vox1_meta.csv`` contains the VoxCeleb Dataset Metadata
- ``bias_tests_4_voice_tech/results`` will be created once the example was run for the first time.
- ``bias_tests_4_voice_tech/results/biastest_results_config_resnetse34v2_H-eval_scores.csv`` is the file where the results of the example will be saved to. It contains metrics ratios for the metrics and speaker groups specified in the config file and evaluated for ResNetSE34V2 scores.


Run Bias Tests for Speaker Verification
_______________________________________

1. Create config file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A template for the ``config.yaml`` file is provided in the ``bias_tests_4_voice_tech/example/`` folder in your home directory::

    speaker_metadata_file: "~/bias_tests_4_voice_tech/example/vox1_meta.csv"
    results_dir: "~/bias_tests_4_voice_tech/results/"

    # for metadata
    id_column: "VoxCeleb1 ID"
    select_columns: ["Gender", "Nationality"]
    speaker_groups: [["Gender"], ["Nationality"], ["Gender", "Nationality"]]

    # for scores
    reference_filepath_column: "ref_file"
    test_filepath_column: "com_file"
    label_column: "lab"
    scores_column: "sc"

    # for dataset evaluation

    dataset_evaluation: True

    # for audit
    dcf_costs: [[0.05, 1, 1]]


2. Run the bias tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Import ``bt4vt`` and specify your score and config file path.

Pass the score and config file path to the :py:class:`bt4vt.core.SpeakerBiasTest` class and run the :py:func:`bt4vt.core.SpeakerBiasTest.audit` function::

    import bt4vt

    score_file = "~/bias_tests_4_voice_tech/example/resnetse34v2_H-eval_scores.csv"
    config_file = "~/bias_tests_4_voice_tech/example/config.yaml"
    test1 = bt4vt.core.SpeakerBiasTest(score_file, config_file)
    test1.audit()

Test results will be stored in ``~/bias_tests_4_voice_tech/results``. The results file contains metrics ratios for the metrics and speaker groups specified in the config file.
