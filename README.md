# Uncertainty in Repeated Implicit Feedback as a Measure of Reliability

This repository provides our Python code to reproduce the experiments from the paper **Uncertainty in Repeated Implicit Feedback as a Measure of
Reliability**. The paper was submitted to ACM UMAP 2025. 

## Abstract
Recommender systems rely heavily on user feedback to learn effective user and item representations. Despite their widespread adoption, limited attention has been given to the uncertainty inherent in the feedback used to train these systems. Both implicit and explicit feedback are prone to noise due to the variability in human interactions, with implicit feedback being particularly challenging. In collaborative filtering, the reliability of interaction signals is critical, as these signals determine user and item similarities. Thus, deriving accurate confidence measures from implicit feedback is essential for ensuring the reliability of these signals.

A common assumption in academia and industry is that repeated interactions indicate stronger user interest, increasing confidence in preference estimates. However, in domains such as music streaming, repeated consumption can shift user preferences over time due to factors like satiation and exposure. While literature on repeated consumption acknowledges these dynamics, they are often overlooked when deriving confidence scores for implicit feedback.

This paper addresses this gap by focusing on music streaming, where repeated interactions are frequent and quantifiable. We analyze how repetition patterns intersect with key factors influencing user interest and develop methods to quantify the associated uncertainty. These uncertainty measures are then integrated as consistency metrics in a recommendation task. Our empirical results show that incorporating uncertainty into user preference models yields more accurate and relevant recommendations. Key contributions include a comprehensive analysis of uncertainty in repeated consumption patterns, the release of a novel dataset, and a Bayesian model for implicit listening feedback.


## Dataset
We will release the data soon.

## Running the code
Running implicit_ALS.py will create the Beta posteriors interpolation files, with the set number of recency bins and the selected prior and will compute the recommendation results for a given weighing scheme. The data used for the experiments has to be split into train, validation and test, with both validation and test containing 2 items per user. 

## Environment

- python 3.9.16
- scipy 1.11.4
- pandas 2.2.3
- numpy 1.26.4
- implicit 0.7.2


Please cite our paper if you use this code in your own work:

```
@inproceedings{sguerra2025uncertainty,
  title={Uncertainty in Repeated Implicit Feedback as a Measure of Reliability},
  author={Sguerra, Bruno and Tran, Viet-Anh and Hennequin, Romain and Moussallam, Manuel},
  booktitle = {Proceedings of the 33rd ACM Conference on User Modeling, Adaptation and Personalization},
  year = {2025}
}
```
