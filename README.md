# SVEva Fair: EVAluating fairness of Speaker Verification
\*SVEva stands for **S**peaker **V**erification **Eva**luation

## About this package

SVEva Fair is an actionable and model-agnostic framework for evaluating the fairness of speaker verification components in voice assistants. The framework provides evaluation measures and visualisations to interrogate model performance across speaker subgroups and compare fairness between models. This repository contains a python library that implements SVEva Fair. You can integrate SVEva Fair into your embedded ML development pipeline to troubleshoot unreliable speaker verification performance, and select high impact approaches for mitigating fairness challenges.

SVEva Fair equips you to interrogate your speaker verification models to answer two questions:

1. Fairness: Does speaker verification performance vary across speaker subgroups for a particular model?
2. Comparison: How does fairness compare across speaker verification models?

Full details on SVEva Fair and a case study that evaluates the fairness of benchmark models trained with the VoxCeleb Trainer benchmark are available here:

*Toussaint, W. and Ding, A. 2021. “SVEva Fair: A Framework for Evaluating Fairness in Speaker Verification.”* 

### Fair Speaker Evaluation

#### Why does fairness matter?

Fair machine learning requires machine learning systems and products to be free from bias and discrimnation against users with protected attributes, like race and ethnicity, sex, religion and belief, age, disability, or sexual orientation. Given the large-scale deployment of speaker verification in voice assistants, and the important function that voice-based systems have as a user interface in wearables, smart homes and autonomous vehicles, it is necessary that they work reliably for all users. Building discrimantory systems can also carry legal consequences and reputational damage for developers.

#### Defining appropriate fairness metrics
The quantitative fairness evaluation that SVEva Fair facilitates is based on the equalised odds fairness metric. Equalised odds requires protected and unprotected subgroups to have equal true and false positive rates, which is equivalent to equal false negative and false positive rates across subgroups. This definition naturally fits with the false negative and false positive rate trade-offs that speaker verification applications must make. In particular, the detection cost function which is the dominant evaluation metric in speaker verification, can be viewed as a weighted equalised odds ratio. SVEva Fair adopts the detection cost as a proxy for fairness.

## Setup instructions
You need python 3 to use this library.

1. Clone this repository from github.
2. Navigate to the root directory (`sveva_fair`) and run `python setup.py install`.
3. Import the sveva_fair `evaluate` and `plot` modules into your project
"""
import sveva_fair.evaluate as sveva_evaluate
import sveva_fair.plot as sveva_plot
"""

## Usage

### Evaluate 


"""
fpfnth_baseline = sveva_evaluate.fpfnth_metrics(df)
fpfnth = sveva_evaluate.sg_fpfnth_metrics(df, ['nationality','sex'])
"""

### Plot

"""
g = sveva_plot.plot_det_curves(fpfnth, hue='sex', style='sex', col='nationality') 
g = sveva_plot.plot_det_baseline(g, fpfnth_baseline[0], fpfnth_baseline[1], 'min_cdet_threshold')
g = sveva_plot.plot_thresholds(g, fpfnthv21_filter, fpfnthv21[1], 'min_cdet_threshold', metrics_baseline=fpfnthv21_baseline[1])
"""

## Case Study with VoxCeleb Trainer
