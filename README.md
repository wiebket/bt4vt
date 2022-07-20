# bt4vt - Bias Tests for Voice Technologies: Building fair voice systems

## About this package

Voice-based systems are widely adopted in users' devices, homes and vehicles as well as in the commerical field, i.e. in call centers or banks. Thus, it is important that they work reliably for all users. Speaker verification is one of the largely deployed voice technologies within these systems that relies on machine learning.
Fair machine learning is necessary for these technologies to be free from bias and discrimnation against users with protected attributes, like race and ethnicity, sex, religion and belief, age, disability, or sexual orientation.

`bt4vt` is an actionable and model-agnostic framework for evaluating the fairness of voice technology components (currently limited to speaker verification). 
This python framework provides evaluation measures and visualisations to interrogate model performance and can be integrated into your development pipeline to troubleshoot unreliable performance.

Link to the full documentation: 

The development of this framework is part of the [Fair EVA open source project](https://www.faireva.org/). 

## Setup instructions
You need python 3 to use this library.

1. Clone this repository from github and navigate to root directory (`bt4vt`)
    ```
    $ git clone https://github.com/wiebket/bt4vt.git
    ```
2. Install all the requirements and the project.

    The installation will create the `bias_tests_4_voice_tech` folder in your home directory. It provides an example how to use `bt4vt` and can be deleted if not required. 
    ```
    $ pip install -r requirements.txt
    $ python setup.py install
    ```
3. Import bt4vt into your project
    ```
    import bt4vt
    ```

## Usage
Below is an example for using `bt4vt`. All code and data for reproducing the example are contained in the `bias_tests_4_voice_tech` folder in your home directory. The example evaluates the fairness of models released with the <a href="https://github.com/clovaai/voxceleb_trainer" target="_blank">VoxCeleb Trainer</a> benchmark.

### Run Bias Tests for Speaker Verification

Import `bt4vt` and specify your score and config file. 
More on the config file can be found here (TODO add link to documentation).

Pass the score and config file to the `SpeakerBiastTest` class and run the `audit()` (TODO Link to documentation) function.

```
import bt4vt

score_file = "resnetse34v2_H-eval_scores.csv"
config_file = "config.yaml"

test1 = bt4vt.core.SpeakerBiasTest(score_file, config_file)

test1.audit()
```

Test results will be stored in `~/bias_tests_4_voice_tech/results`. The results file contains metrics ratios for the metrics and speaker groups specified in the config file. 

### Under Development

The project is under constant development and planned enhancements will include advanced plotting of test results and metrics implementation. 

If you'd like to get involved, have a look at this page: https://www.faireva.org/get-involved 

### Resources and Citation

This work is based on SVEva Fair. Full details on SVEva Fair and a case study that evaluates the fairness of benchmark models trained with the VoxCeleb Trainer benchmark are available here:

*Toussaint, W. and Ding, A. 2021. “SVEva Fair: A Framework for Evaluating Fairness in Speaker Verification.”, https://arxiv.org/abs/2107.12049* 

### License

This code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public License for details.

You should have received a copy of the GNU General Public License along with this source code. If not, go the following link: http://www.gnu.org/licenses/.