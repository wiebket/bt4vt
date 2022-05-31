=======
Example
=======

Below is an example for using bt4vt. All code and data for reproducing the example are contained in the [example](example) directory.
The example evaluates the fairness of models released with the `VoxCeleb Trainer <https://github.com/clovaai/voxceleb_trainer>`_ benchmark.
For convenience, we have included a `voxceleb` module for generating the appropriate input dataframe for bt4vt from the VoxCeleb Trainer results and metadata.
The results can be reproduced with the code and models made available by voxceleb trainer.
For evaluation we use scores for the speaker pairs from the `H` test set:

Evaluate speaker verification performance across subgroups
__________________________________________________________

1. Create config file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A template for the config.yaml file is provided in the example folder::

    speaker_metadata_file: "vox1_meta.csv"

    # for metadata
    id_column: "VoxCeleb1 ID"
    select_columns: ["Gender", "Nationality"]
    speaker_groups: [["Gender"], ["Nationality"], ["Gender", "Nationality"]]

    # for scores
    reference_filepath_column: "ref_file"
    test_filepath_column: "com_file"
    label_column: "lab"
    scores_column: "sc"

    # for audit
    dcf_costs: [[0.05, 1, 1]]


2. Run the bias test audit
^^^^^^^^^^^^^^^^^^^^^^^
::

    import bt4vt

    score_file = "resnetse34v2_H/eval_scores.csv"
    config_file = "config.yaml"
    test1 = bt4vt.core.SpeakerBiasTest(score_file, config_file)
    test1.audit()

