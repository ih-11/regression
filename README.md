# regression

Machine learning experiments for predicting PR_var from transcript sequence features.

## Data

Three plant species:

- AT21
- NB21
- OS21

Each dataset contains:

- var_id
- trans_id
- gene_id
- PR_var
- 5'UTR
- CDS
- 3'UTR

## Goal

Predict PR_var from sequence-derived features and evaluate within-species and cross-species performance.

# Benchmark

Benchmark multiple machine learning models for predicting transcript-level
translation activity (PR_var) from sequence-encoded features derived from the
5′ UTR, CDS, and 3′ UTR.

The benchmark evaluates:

- Linear vs. nonlinear modeling strategies.
- Default vs. hyperparameter-tuned models.
- Cross-species performance across Arabidopsis (AT21), Nicotiana (NB21), and Oryza (OS21).

The goal is not only to identify the best predictive model, but also to assess
whether translation is primarily explained by linear or nonlinear relationships
between sequence features and PR_var.