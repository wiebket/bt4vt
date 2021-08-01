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
```
import sveva_fair.evaluate as sveva_evaluate
import sveva_fair.plot as sveva_plot
```

## Usage
Below is an example for using SVEva Fair. All code and data for reproducing the example are contained in the [\example](\example) directory. The example evaluates the fairness of models released with the [VoxCeleb Trainer]().

### Evaluate speaker verification performance across subgroups

#### 1. Generate speaker verification results with metadata

For convenience, we have included a `voxceleb` module for generating the appropriate input dataframe for SVEva Fair from the VoxCeleb Trainer results and metadata. The results can be reproduced with the code and models made available by voxceleb trainer. For evaluation we use scores for the speaker pairs from the 'H' test set.

```
import sveva_fair.voxceleb as sveva_vox

score_file = 'example/resnetse34v2_H/eval_scores.csv'
meta_file = 'example/vox1_meta.csv'
df = sveva_vox.voxceleb_scores_with_demographics(score_file, meta_file, sep='\t')
df.rename(columns={"ref_gender": "sex", "ref_nationality":"nationality"}, inplace=True)
```

#### 2. Evaluate the results 
The evaluation functions return false negative rates (FNR), false positive rates (FPR) and the corresponding threshold values for speaker verification scores stored in dataset `df`. `df` must have a column with scores `df['sc']` and a column with labels `df['lab']`. The `sg_` function determines these values for subgroups that are specified in a list of column names passed to the filter_keys argument.

```
all_fpfnth, all_metrics = sveva_evaluate.fpfnth_metrics(df)
sg_fpfnth, sg_metrics = sveva_evaluate.sg_fpfnth_metrics(df, filter_keys=['nationality','sex'])
```

In addition to the FPR, FNR and threshold values, the functions also return a dictionary with metrics: `min_cdet`, the minimum of the detection cost function, and `eer`, the equal error rate. The following optional arguments can be passed to the evaluation functions to adjust the detection cost to the application requirements:
```
dcf_p_target [float]: detection cost function target (default = 0.05)
dcf_c_fn [float]: detection cost function false negative weight (default = 1)
dcf_c_fp [float]: detection cost function  false positive weight (default = 1)
```

#### 3. Plot Detection Error Trade-off curves

```
g = sveva_plot.plot_det_curves(sg_fpfnth, hue='sex', style='sex', col='nationality') 
g = sveva_plot.plot_det_baseline(g, all_fpfnth, all_metrics, threshold_type='min_cdet_threshold')
g = sveva_plot.plot_thresholds(g, sg_fpfnth, sg_metrics, threshold_type='min_cdet_threshold', metrics_baseline=all_metrics)
```
FIG

