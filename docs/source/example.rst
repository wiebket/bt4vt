=======
Example
=======

Below is an example for using ``bt4vt``. All data for reproducing the example is provided as a package resource. The example evaluates the fairness of models released with the `Clova AI VoxCeleb Trainer <https://github.com/clovaai/voxceleb_trainer>`_ benchmark.

The package resources are:

- ``data/config.yaml`` is an example config file that is needed to run bias tests. More information on how to create a config file can be found below.
- ``data/resnetse34v2_H-eval_scores.csv`` and ``data/resnetse34l_H-eval_scores.csv`` are the evaluation scores for two pretrained baseline models made available by the VoxCeleb Trainer. You can find more information on ResNetSE34V2 `here <https://arxiv.org/abs/2009.14153>`_ and on ResNetSE34L `here <https://doi.org/10.21437/Interspeech.2020-1064>`_.
- ``data/vox1_meta.csv`` contains the VoxCeleb Dataset Metadata

When running the example for the first time, ``bt4vt`` will create a ``bias_tests_4_voice_tech/results`` folder in your ``home`` directory to store the results files.

| bias_tests_4_voice_tech
| ├── results
| │   └── biastest_results_config_resnetse34v2_H-eval_scores.csv

- ``bias_tests_4_voice_tech/results/biastest_results_config_resnetse34v2_H-eval_scores.csv`` is the file where the results of the example will be saved to. It contains metrics ratios and metric results for the metrics and speaker groups specified in the config file and evaluated for ResNetSE34V2 scores.


Run Bias Tests for Speaker Verification
_______________________________________

1. Create config file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A template for the ``config.yaml`` file is provided as a package resource::

    speaker_metadata_file: "vox1_meta.csv"
    results_dir: "~/bias_tests_4_voice_tech/results/"

    # for metadata
    id_column: "VoxCeleb1 ID"
    select_columns: ["Gender", "Nationality"]
    speaker_groups: [["Gender"], ["Nationality"], ["Gender", "Nationality"]]
    # optional attributes
    # id_delimiter: "-" (default is "/")

    # for scores
    reference_filepath_column: "ref_file"
    test_filepath_column: "com_file"
    label_column: "lab"
    scores_column: "sc"

    # for dataset evaluation

    dataset_evaluation: True

    # for run_tests
    dcf_costs: [[0.05, 1, 1]]


2. Run the bias tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Import ``bt4vt`` and access the score and config file provided as package resources.

Pass the score and config file path to the :py:class:`bt4vt.core.SpeakerBiasTest` class and run the :py:func:`bt4vt.core.SpeakerBiasTest.run_tests` function::

    import bt4vt
    import importlib.resources

    with importlib.resources.path("bt4vt.data", "resnetse34v2_H-eval_scores.csv") as path:
        score_file = str(path)

    with importlib.resources.path("bt4vt.data", "config.yaml") as path:
        config_file = str(path)

    test = bt4vt.core.SpeakerBiasTest(score_file, config_file)
    test.run_tests()

Test results will be stored in ``~/bias_tests_4_voice_tech/results``. The results file contains metrics ratios for the metrics and speaker groups specified in the config file.
