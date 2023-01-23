=======
Example
=======

Below is an example for using ``bt4vt``. All data for reproducing the example can be copied by using :py:func:`bt4vt.dataio.copy_example`. The example evaluates the fairness of models released with the `Clova AI VoxCeleb Trainer <https://github.com/clovaai/voxceleb_trainer>`_ benchmark.

For now, we copy the resources to ``~/bias_tests_4_voice_tech/example/``::

    import bt4vt

    bt4vt.dataio.copy_example("~/bias_tests_4_voice_tech/example/")


The following files are now copied to the folder:

bias_tests_4_voice_tech
├── example
| ├── config.yaml
| ├── resnetse34l_H-eval_scores.csv
| ├── resnetse34v2_H-eval_scores.csv
| └── vox1_meta.csv

- ``config.yaml`` is an example config file that is needed to run bias tests. More information on how to create a config file can be found below.
- ``resnetse34v2_H-eval_scores.csv`` and ``resnetse34l_H-eval_scores.csv`` are the evaluation scores for two pretrained baseline models made available by the VoxCeleb Trainer. You can find more information on ResNetSE34V2 `here <https://arxiv.org/abs/2009.14153>`_ and on ResNetSE34L `here <https://doi.org/10.21437/Interspeech.2020-1064>`_.
- ``vox1_meta.csv`` contains the VoxCeleb Dataset Metadata

When running the example, ``bt4vt`` will create a ``bias_tests_4_voice_tech/results`` folder to store the results files.

bias_tests_4_voice_tech
├── example
├── results
│   └── biastest_results_config_resnetse34v2_H-eval_scores.csv

- ``bias_tests_4_voice_tech/results/biastest_results_config_resnetse34v2_H-eval_scores.csv`` is the file where the results of the example will be saved to. It contains metrics ratios and metric results for the metrics and speaker groups specified in the config file and evaluated for ResNetSE34V2 scores.


Run Bias Tests for Speaker Verification
_______________________________________

1. Create config file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A template for the ``config.yaml`` file is now provided in the ``~/bias_tests_4_voice_tech/example/`` folder.
If you copied the files to a different folder you need to adjust the path to the ``speaker_metadata_file`` and ``results_dir``.::

    speaker_metadata_file: "~/bias_tests_4_voice_tech/example/vox1_meta.csv"
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

    score_file = "~/bias_tests_4_voice_tech/example/resnetse34v2_H-eval_scores.csv"
    config_file = "~/bias_tests_4_voice_tech/example/config.yaml"

    test = bt4vt.core.SpeakerBiasTest(score_file, config_file)
    test.run_tests()

Test results will be stored in ``~/bias_tests_4_voice_tech/results``. The results file contains metrics ratios for the metrics and speaker groups specified in the config file.
