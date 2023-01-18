# Bias Tests for Voice Technologies (bt4vt)

## About this package

`bt4vt` is a python library to diagnose performance discrepancies (i.e. bias) in speaker verification models. The library provides evaluation measures and visualisations to interrogate model performance and can be integrated into development pipelines to test for bias. We plan to extend the library to other speech processing tasks in future. Speak to us if you're interested to help.

<a href="https://bt4vt.readthedocs.io/en/latest/" target="_blank">Read the docs</a>

The development of this open source library is part of the <a href="https://www.faireva.org/" target="_blank">Fair EVA</a>
project and has been supported by the Mozilla Technology Fund.

## Setup instructions
You need `python 3` to use this library. The easiest way to use the library is to install it with pip.
```
$ pip install bt4vt
```
To use the library in development mode, install it as follows:

1. Clone this repository from github and navigate to the project's root directory (`bt4vt\`)
    ```
    $ git clone https://github.com/wiebket/bt4vt.git
    ```
2. Install the project.

    ```
    $ pip install . -e
    ```
The installation will create the `bias_tests_4_voice_tech` folder in your home directory. It provides an example how to use `bt4vt` and can be deleted if not required. 

## Usage

Import the library into your project
    ```
    import bt4vt
    ```

Below is an example for using `bt4vt`. All code and data for reproducing the example are contained in the `bias_tests_4_voice_tech` folder in your `home` directory. The example evaluates the fairness of models released with the <a href="https://github.com/clovaai/voxceleb_trainer" target="_blank">Clova AI VoxCeleb Trainer</a>.

### Run Bias Tests for Speaker Verification

#### 1. Create config file

A template for the `config.yaml` file is provided in the `~/bias_tests_4_voice_tech/example/` folder after package installation.

```
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

    # for run_tests
    dcf_costs: [[0.05, 1, 1]]
```

#### 2. Run the bias tests 

Import `bt4vt` and specify your score and config file. Pass the score and config file path to the `SpeakerBiastTest` class and run the `run_tests()` function.

```
import bt4vt

score_file = "resnetse34v2_H-eval_scores.csv"
config_file = "config.yaml"

test1 = bt4vt.core.SpeakerBiasTest(score_file, config_file)

test1.run_tests()
```

Test results will be stored in `~/bias_tests_4_voice_tech/results`. The results file contains *metrics ratios* for the metrics and speaker groups specified in the config file. 

The *metrics ratio* is calculated as ```speaker group metric / average metric```.

### Under Development

The project is under continuous development and we appreciate contributions! Planned enhancements include:
* advanced plotting of test results
* implementation of further metrics and fairness measures
* inclusive evaluation dataset generators

If you'd like to get involved, have a look at: https://www.faireva.org/get-involved 

### Resources and Citation

Early versions of this library were prototyped in the following research:

*Wiebke Toussaint Hutiri and Aaron Yi Ding. 2022. Bias in Automated Speaker Recognition. In 2022 ACM Conference on Fairness, Accountability, and Transparency (FAccT '22). Association for Computing Machinery, New York, NY, USA, 230–247. https://doi-org.tudelft.idm.oclc.org/10.1145/3531146.3533089* 

### License

This code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public License for details.

You should have received a copy of the GNU General Public License along with this source code. If not, go the following link: http://www.gnu.org/licenses/.
