============
bt4vt - Bias Tests for Voice Tech
============

About this Package
==================
bt4vt is an actionable and model-agnostic framework for evaluating the fairness of speaker verification components in voice assistants. The framework provides evaluation measures and visualisations to interrogate model performance across speaker subgroups and compare fairness between models. This repository contains a python library that implements bt4vt. You can integrate bt4vt into your embedded ML development pipeline to troubleshoot unreliable speaker verification performance, and select high impact approaches for mitigating fairness challenges.

bt4vt equips you to interrogate your speaker verification models to answer two questions:

1. Fairness: Does speaker verification performance vary across speaker subgroups for a particular model?
2. Comparison: How does fairness compare across speaker verification models?

Full details on bt4vt and a case study that evaluates the fairness of benchmark models trained with the VoxCeleb Trainer benchmark are available here:

Toussaint, W. and Ding, A. 2021. “SVEva Fair: A Framework for Evaluating Fairness in Speaker Verification.”, `<https://arxiv.org/abs/2107.12049>`_

Fair Speaker Evaluation
=======================

Why does fairness matter?
_________________________

Fair machine learning requires machine learning systems and products to be free from bias and discrimnation against users with protected attributes, like race and ethnicity, sex, religion and belief, age, disability, or sexual orientation. Given the large-scale deployment of speaker verification in voice assistants, and the important function that voice-based systems have as a user interface in wearables, smart homes and autonomous vehicles, it is necessary that they work reliably for all users. Building discrimantory systems can also carry legal consequences and reputational damage for developers.

Defining appropriate fairness metrics
_____________________________________

The quantitative fairness evaluation that bt4vt facilitates is based on the equalised odds fairness metric. Equalised odds requires protected and unprotected subgroups to have equal true and false positive rates, which is equivalent to equal false negative and false positive rates across subgroups. This definition naturally fits with the false negative and false positive rate trade-offs that speaker verification applications must make. In particular, the detection cost function which is the dominant evaluation metric in speaker verification, can be viewed as a weighted equalised odds ratio. bt4vt adopts the detection cost as a proxy for fairness.
