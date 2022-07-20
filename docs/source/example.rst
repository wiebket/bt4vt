=======
Example
=======

Below is an example for using **bt4vt**. All code and data for reproducing the example are contained in the *bias_tests_4_voice_tech* folder in your home directory. The example evaluates the fairness of models released with the `VoxCeleb Trainer <https://github.com/clovaai/voxceleb_trainer>`_ benchmark.

Run Bias Tests for Speaker Verification
_______________________________________

1. Create config file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A template for the config.yaml file is provided in the *bias_tests_4_voice_tech/example/* folder in your home directory::

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
Import **bt4vt** and specify your score and config file path.

Pass the score and config file path to the :py:class:`bt4vt.core.SpeakerBiasTest` class and run the :py:func:`bt4vt.core.SpeakerBiasTest.audit` function::

    import bt4vt

    score_file = "~/bias_tests_4_voice_tech/example/resnetse34v2_H-eval_scores.csv"
    config_file = "~/bias_tests_4_voice_tech/example/config.yaml"
    test1 = bt4vt.core.SpeakerBiasTest(score_file, config_file)
    test1.audit()

Test results will be stored in *~/bias_tests_4_voice_tech/results*. The results file contains metrics ratios for the metrics and speaker groups specified in the config file.
